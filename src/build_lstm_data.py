from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    PipelineConfig,
    ensure_output_dir,
    get_samples_path,
    get_sequence_array_paths,
    get_sequence_feature_names_path,
    load_samples_frame,
    print_section,
    require_columns,
    resolve_existing_path,
    save_json,
)


def load_orders_and_samples(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 데이터 로드
    print_section("[INFO] LSTM 데이터 로드")
    orders_path = resolve_existing_path(config.orders_path)
    sample_path = get_samples_path(config)
    if not sample_path.exists():
        raise FileNotFoundError(f"전처리 결과가 없습니다: {sample_path}")

    orders = pd.read_csv(
        orders_path,
        usecols=[
            "user_id",
            "order_number",
            "days_since_prior_order",
            "order_dow",
            "order_hour_of_day",
        ],
        dtype={
            "user_id": "int32",
            "order_number": "int16",
            "days_since_prior_order": "float32",
            "order_dow": "float32",
            "order_hour_of_day": "float32",
        },
    )
    samples = load_samples_frame(config)

    require_columns(
        orders,
        ["user_id", "order_number", "days_since_prior_order", "order_dow", "order_hour_of_day"],
        "orders",
    )
    require_columns(
        samples,
        [
            "sample_id",
            "user_id",
            "target_order_number",
            "split",
            "label",
            "hist_total_orders_log",
            "hist_mean_gap",
            "hist_gap_mean_3",
            "hist_order_frequency",
        ],
        "samples",
    )

    orders["days_since_prior_order"] = orders["days_since_prior_order"].fillna(0.0)
    orders["order_dow"] = orders["order_dow"].fillna(0.0)
    orders["order_hour_of_day"] = orders["order_hour_of_day"].fillna(0.0)
    orders = orders.sort_values(["user_id", "order_number"]).reset_index(drop=True)

    print(f"[INFO] 주문 수: {len(orders)}")
    print(f"[INFO] 샘플 수: {len(samples)}")
    return orders, samples


def _build_order_lookup(orders: pd.DataFrame) -> dict[int, pd.DataFrame]:
    # 주문 인덱스 생성
    lookup: dict[int, pd.DataFrame] = {}
    for user_id, user_orders in orders.groupby("user_id", sort=False):
        lookup[int(user_id)] = user_orders.sort_values("order_number").reset_index(drop=True)
    return lookup


def build_sequence_tensor(
    orders: pd.DataFrame,
    samples: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[dict[str, np.ndarray], list[str]]:
    # 시퀀스 생성
    print_section("[INFO] 시퀀스 생성")
    order_lookup = _build_order_lookup(orders)
    split_x: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_y: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    split_ids: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    feature_names = [
        "seq_gap",
        "seq_gap_change",
        "seq_order_norm",
        "seq_order_dow",
        "seq_order_hour",
        "static_hist_total_orders_log",
        "static_hist_mean_gap",
        "static_hist_gap_mean_3",
        "static_hist_order_frequency",
    ]

    for row in samples.itertuples(index=False):
        user_orders = order_lookup.get(int(row.user_id))
        if user_orders is None:
            continue

        target_positions = np.where(user_orders["order_number"].to_numpy() == int(row.target_order_number))[0]
        if len(target_positions) == 0:
            continue
        target_idx = int(target_positions[0])
        if target_idx < config.seq_len:
            continue

        window = user_orders.iloc[target_idx - config.seq_len : target_idx].copy()
        gaps = window["days_since_prior_order"].to_numpy(dtype=np.float32)
        gap_change = np.diff(np.concatenate(([gaps[0]], gaps))).astype(np.float32)
        order_norm = (
            window["order_number"].to_numpy(dtype=np.float32)
            / max(float(window["order_number"].iloc[-1]), 1.0)
        )
        dows = window["order_dow"].to_numpy(dtype=np.float32)
        hours = window["order_hour_of_day"].to_numpy(dtype=np.float32)
        static_vector = np.array(
            [
                float(row.hist_total_orders_log),
                float(row.hist_mean_gap),
                float(row.hist_gap_mean_3),
                float(row.hist_order_frequency),
            ],
            dtype=np.float32,
        )
        static_window = np.repeat(static_vector.reshape(1, -1), config.seq_len, axis=0)
        sequence = np.column_stack([gaps, gap_change, order_norm, dows, hours]).astype(np.float32)
        sequence = np.concatenate([sequence, static_window], axis=1)

        split_x[row.split].append(sequence)
        split_y[row.split].append(int(row.label))
        split_ids[row.split].append(int(row.sample_id))

    arrays: dict[str, np.ndarray] = {}
    for split_name in ["train", "val", "test"]:
        if not split_x[split_name]:
            raise ValueError(f"{split_name} 시퀀스가 없습니다.")
        arrays[f"x_{split_name}"] = np.stack(split_x[split_name]).astype(np.float32)
        arrays[f"y_{split_name}"] = np.asarray(split_y[split_name], dtype=np.int32)
        arrays[f"{split_name}_sample_ids"] = np.asarray(split_ids[split_name], dtype=np.int32)
        print(f"[INFO] {split_name} shape: {arrays[f'x_{split_name}'].shape}")

    return arrays, feature_names


def save_sequence_data(arrays: dict[str, np.ndarray], feature_names: list[str], config: PipelineConfig) -> dict:
    # 배열 저장
    print_section("[INFO] LSTM 데이터 저장")
    output_dir = ensure_output_dir(config)
    array_paths = get_sequence_array_paths(config)

    np.save(array_paths["x_train"], arrays["x_train"])
    np.save(array_paths["x_val"], arrays["x_val"])
    np.save(array_paths["x_test"], arrays["x_test"])
    np.save(array_paths["y_train"], arrays["y_train"])
    np.save(array_paths["y_val"], arrays["y_val"])
    np.save(array_paths["y_test"], arrays["y_test"])
    np.save(array_paths["test_sample_ids"], arrays["test_sample_ids"])

    save_json(get_sequence_feature_names_path(config), {"feature_names": feature_names})
    save_json(
        output_dir / "lstm_data_summary.json",
        {
            "feature_names": feature_names,
            "train_shape": arrays["x_train"].shape,
            "val_shape": arrays["x_val"].shape,
            "test_shape": arrays["x_test"].shape,
        },
    )

    print(f"[INFO] 저장 경로: {output_dir}")
    return {key: str(value) for key, value in array_paths.items()}


def run_build_lstm_data(config: PipelineConfig) -> dict:
    # LSTM 데이터 생성
    orders, samples = load_orders_and_samples(config)
    arrays, feature_names = build_sequence_tensor(orders, samples, config)
    return save_sequence_data(arrays, feature_names, config)
