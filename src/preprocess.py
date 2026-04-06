from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    PipelineConfig,
    check_binary_labels,
    ensure_output_dir,
    get_samples_path,
    get_tabular_feature_names_path,
    print_section,
    require_columns,
    resolve_existing_path,
    resolve_feature_path,
    save_json,
)


ORDER_COLUMNS = [
    "user_id",
    "order_number",
    "days_since_prior_order",
    "order_dow",
    "order_hour_of_day",
]

FEATURE_COLUMNS = [
    "user_id",
    "total_orders",
]


def load_data(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 데이터 로드
    print_section("[INFO] 데이터 로드")
    orders_path = resolve_existing_path(config.orders_path)
    feature_path = resolve_feature_path(config)

    print(f"[INFO] orders 파일: {orders_path}")
    print(f"[INFO] feature 파일: {feature_path}")

    orders_header = pd.read_csv(orders_path, nrows=0)
    feature_header = pd.read_csv(feature_path, nrows=0)

    print(f"[INFO] orders 컬럼: {list(orders_header.columns)}")
    print(f"[INFO] feature 컬럼: {list(feature_header.columns)}")

    require_columns(orders_header, ["user_id", "order_number", "days_since_prior_order"], "orders")
    require_columns(feature_header, FEATURE_COLUMNS, "feature")

    use_order_columns = [column for column in ORDER_COLUMNS if column in orders_header.columns]
    use_feature_columns = [column for column in FEATURE_COLUMNS if column in feature_header.columns]
    order_dtypes = {
        "user_id": "int32",
        "order_number": "int16",
        "days_since_prior_order": "float32",
        "order_dow": "float32",
        "order_hour_of_day": "float32",
    }
    feature_dtypes = {
        "user_id": "int32",
        "total_orders": "float32",
    }

    orders = pd.read_csv(
        orders_path,
        usecols=use_order_columns,
        dtype={column: order_dtypes[column] for column in use_order_columns},
    )
    features = pd.read_csv(
        feature_path,
        usecols=use_feature_columns,
        dtype={column: feature_dtypes[column] for column in use_feature_columns},
    )

    return orders, features


def filter_high_frequency_users(
    orders: pd.DataFrame,
    features: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    # 고빈도 고객 선택
    print_section("[INFO] 고빈도 고객 필터링")
    threshold = float(features["total_orders"].quantile(config.high_value_quantile))
    high_value_users = features.loc[features["total_orders"] >= threshold, ["user_id"]].drop_duplicates()

    if high_value_users.empty:
        raise ValueError("고빈도 고객이 없습니다.")

    filtered_orders = orders[orders["user_id"].isin(high_value_users["user_id"])].copy()
    if filtered_orders.empty:
        raise ValueError("고빈도 고객 주문 데이터가 없습니다.")

    filtered_orders["days_since_prior_order"] = filtered_orders["days_since_prior_order"].fillna(0.0)
    if "order_dow" not in filtered_orders.columns:
        filtered_orders["order_dow"] = 0.0
    if "order_hour_of_day" not in filtered_orders.columns:
        filtered_orders["order_hour_of_day"] = 0.0
    filtered_orders["order_dow"] = filtered_orders["order_dow"].fillna(0.0)
    filtered_orders["order_hour_of_day"] = filtered_orders["order_hour_of_day"].fillna(0.0)
    filtered_orders = filtered_orders.sort_values(["user_id", "order_number"]).reset_index(drop=True)

    print(f"[INFO] total_orders 80분위수: {threshold:.2f}")
    print(f"[INFO] 고빈도 고객 수: {high_value_users['user_id'].nunique()}")
    print(f"[INFO] 필터 후 주문 수: {len(filtered_orders)}")
    return filtered_orders


def _build_user_samples(user_orders: pd.DataFrame, config: PipelineConfig) -> list[dict]:
    # 사용자 샘플 생성
    if config.max_recent_orders is not None and len(user_orders) > config.max_recent_orders:
        user_orders = user_orders.tail(config.max_recent_orders).reset_index(drop=True)

    if len(user_orders) < config.seq_len + 3:
        return []

    gaps = user_orders["days_since_prior_order"].to_numpy(dtype=np.float32)
    dows = user_orders["order_dow"].to_numpy(dtype=np.float32)
    hours = user_orders["order_hour_of_day"].to_numpy(dtype=np.float32)
    order_numbers = user_orders["order_number"].to_numpy(dtype=np.int32)

    sample_rows: list[dict] = []
    for target_idx in range(config.seq_len, len(user_orders)):
        history_orders = user_orders.iloc[:target_idx]
        positive_gaps = history_orders.loc[
            history_orders["days_since_prior_order"] > 0, "days_since_prior_order"
        ].to_numpy(dtype=np.float32)
        recent_gaps = positive_gaps[-3:]

        last_gap = float(gaps[target_idx - 1]) if target_idx >= 1 else 0.0
        prev_gap = float(gaps[target_idx - 2]) if target_idx >= 2 else 0.0
        total_gap_days = float(positive_gaps.sum()) if len(positive_gaps) else 0.0
        hist_total_orders = int(target_idx)
        hist_mean_gap = float(positive_gaps.mean()) if len(positive_gaps) else 0.0
        hist_std_gap = float(positive_gaps.std()) if len(positive_gaps) else 0.0
        hist_min_gap = float(positive_gaps.min()) if len(positive_gaps) else 0.0
        hist_max_gap = float(positive_gaps.max()) if len(positive_gaps) else 0.0
        hist_gap_mean_3 = float(recent_gaps.mean()) if len(recent_gaps) else 0.0
        hist_gap_std_3 = float(recent_gaps.std()) if len(recent_gaps) else 0.0
        hist_order_frequency = float(hist_total_orders / max(total_gap_days, 1.0))

        sample_rows.append(
            {
                "user_id": int(user_orders.iloc[0]["user_id"]),
                "target_order_number": int(order_numbers[target_idx]),
                "target_days_since_prior_order": float(gaps[target_idx]),
                "label": int(gaps[target_idx] > config.delay_threshold),
                "history_end_order_number": int(order_numbers[target_idx - 1]),
                "seq_start_order_number": int(order_numbers[target_idx - config.seq_len]),
                "hist_total_orders": hist_total_orders,
                "hist_total_orders_log": float(np.log1p(hist_total_orders)),
                "hist_mean_gap": hist_mean_gap,
                "hist_std_gap": hist_std_gap,
                "hist_min_gap": hist_min_gap,
                "hist_max_gap": hist_max_gap,
                "hist_last_gap": last_gap,
                "hist_gap_mean_3": hist_gap_mean_3,
                "hist_gap_std_3": hist_gap_std_3,
                "hist_gap_change_1": float(last_gap - prev_gap),
                "hist_order_span_days": total_gap_days,
                "hist_order_frequency": hist_order_frequency,
                "hist_mean_dow": float(dows[:target_idx].mean()),
                "hist_mean_hour": float(hours[:target_idx].mean()),
                "hist_last_dow": float(dows[target_idx - 1]),
                "hist_last_hour": float(hours[target_idx - 1]),
            }
        )
    return sample_rows


def build_time_aware_samples(orders: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    # 누수 방지 처리
    print_section("[INFO] 누수 방지 샘플 생성")
    sample_rows: list[dict] = []
    skipped_users = 0

    for _, user_orders in orders.groupby("user_id", sort=False):
        user_orders = user_orders.sort_values("order_number").reset_index(drop=True)
        rows = _build_user_samples(user_orders, config)
        if not rows:
            skipped_users += 1
            continue
        sample_rows.extend(rows)

    if not sample_rows:
        raise ValueError("생성된 샘플이 없습니다.")

    samples = pd.DataFrame(sample_rows)
    samples.insert(0, "sample_id", np.arange(len(samples), dtype=np.int32))
    print(f"[INFO] 생성 샘플 수: {len(samples)}")
    print(f"[INFO] 제외 사용자 수: {skipped_users}")
    return samples


def assign_time_splits(samples: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    # 시점 기준 분할
    print_section("[INFO] 시점 기준 분할")
    assigned_rows: list[pd.DataFrame] = []

    for _, user_samples in samples.groupby("user_id", sort=False):
        user_samples = user_samples.sort_values("target_order_number").reset_index(drop=True)
        count = len(user_samples)
        if count < 3:
            continue

        test_count = max(1, int(np.ceil(count * config.test_ratio)))
        val_count = max(1, int(np.ceil(count * config.val_ratio)))
        train_count = count - test_count - val_count
        if train_count < 1:
            train_count = count - 2
            val_count = 1
            test_count = 1
        if train_count < 1:
            continue

        split = ["train"] * train_count + ["val"] * val_count + ["test"] * test_count
        user_samples = user_samples.iloc[: len(split)].copy()
        user_samples["split"] = split
        assigned_rows.append(user_samples)

    if not assigned_rows:
        raise ValueError("train/val/test 분할이 가능한 샘플이 없습니다.")

    assigned = pd.concat(assigned_rows, ignore_index=True)

    for split_name in ["train", "val", "test"]:
        split_labels = assigned.loc[assigned["split"] == split_name, "label"].to_numpy(dtype=np.int32)
        if len(split_labels) == 0:
            raise ValueError(f"{split_name} 분할이 비어 있습니다.")
        check_binary_labels(split_labels, split_name)

    print(f"[INFO] split 분포: {assigned['split'].value_counts().to_dict()}")
    print(f"[INFO] label 분포: {assigned['label'].value_counts().sort_index().to_dict()}")
    return assigned


def save_preprocessed_data(samples: pd.DataFrame, config: PipelineConfig) -> dict:
    # 전처리 저장
    print_section("[INFO] 전처리 완료")
    output_dir = ensure_output_dir(config)
    sample_path = get_samples_path(config)
    samples.to_csv(sample_path, index=False)

    feature_columns = [column for column in samples.columns if column.startswith("hist_")]
    save_json(get_tabular_feature_names_path(config), {"feature_names": feature_columns})
    save_json(
        output_dir / "preprocess_summary.json",
        {
            "config": config.to_dict(),
            "sample_count": int(len(samples)),
            "user_count": int(samples["user_id"].nunique()),
            "split_counts": {key: int(value) for key, value in samples["split"].value_counts().to_dict().items()},
            "label_counts": {
                str(key): int(value)
                for key, value in samples["label"].value_counts().sort_index().to_dict().items()
            },
        },
    )

    print(f"[INFO] 샘플 저장: {sample_path}")
    return {
        "sample_path": str(sample_path),
        "feature_columns": feature_columns,
    }


def run_preprocess(config: PipelineConfig) -> dict:
    # 전처리 실행
    orders, features = load_data(config)
    filtered_orders = filter_high_frequency_users(orders, features, config)
    samples = build_time_aware_samples(filtered_orders, config)
    samples = assign_time_splits(samples, config)
    return save_preprocessed_data(samples, config)
