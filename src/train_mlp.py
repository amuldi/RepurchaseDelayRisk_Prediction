from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.evaluate import (
    compute_metrics,
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_metrics_report,
    save_prediction_frame,
)
from src.utils import (
    PipelineConfig,
    check_binary_labels,
    ensure_output_dir,
    get_tabular_feature_columns,
    load_samples_frame,
    print_section,
    set_global_seed,
)


def _import_tensorflow():
    # 텐서플로 확인
    try:
        import tensorflow as tf

        return tf
    except ImportError as error:
        raise ImportError("MLP 학습에는 tensorflow가 필요합니다.") from error


def load_tabular_splits(config: PipelineConfig) -> dict:
    # 데이터 로드
    print_section("[INFO] MLP 데이터 로드")
    samples = load_samples_frame(config)
    feature_columns = get_tabular_feature_columns(samples)

    train_frame = samples.loc[samples["split"] == "train"].copy()
    val_frame = samples.loc[samples["split"] == "val"].copy()
    test_frame = samples.loc[samples["split"] == "test"].copy()

    x_train_raw = train_frame[feature_columns].to_numpy(dtype=np.float32)
    x_val_raw = val_frame[feature_columns].to_numpy(dtype=np.float32)
    x_test_raw = test_frame[feature_columns].to_numpy(dtype=np.float32)

    mean = x_train_raw.mean(axis=0, keepdims=True)
    std = x_train_raw.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    x_train = (x_train_raw - mean) / std
    x_val = (x_val_raw - mean) / std
    x_test = (x_test_raw - mean) / std

    y_train = train_frame["label"].to_numpy(dtype=np.int32)
    y_val = val_frame["label"].to_numpy(dtype=np.int32)
    y_test = test_frame["label"].to_numpy(dtype=np.int32)

    check_binary_labels(y_train, "mlp_train")
    check_binary_labels(y_val, "mlp_val")
    check_binary_labels(y_test, "mlp_test")

    return {
        "feature_columns": feature_columns,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "test_sample_ids": test_frame["sample_id"].to_numpy(dtype=np.int32),
        "scaler_mean": mean.astype(np.float32),
        "scaler_std": std.astype(np.float32),
    }


def build_model(input_dim: int, learning_rate: float):
    # 모델 정의
    tf = _import_tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def _make_class_weight(y_train: np.ndarray) -> dict[int, float]:
    # 가중치 생성
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def _permutation_importance(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_columns: list[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    # 중요도 계산
    rng = np.random.default_rng(seed)
    baseline_prob = model.predict(x_test, verbose=0).ravel()
    baseline_auc = compute_metrics(y_test, baseline_prob)["roc_auc"]
    rows: list[dict] = []

    for feature_index, feature_name in enumerate(feature_columns):
        drops = []
        for _ in range(repeats):
            shuffled = x_test.copy()
            rng.shuffle(shuffled[:, feature_index])
            shuffled_prob = model.predict(shuffled, verbose=0).ravel()
            shuffled_auc = compute_metrics(y_test, shuffled_prob)["roc_auc"]
            drops.append(baseline_auc - shuffled_auc)
        rows.append(
            {
                "feature": feature_name,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def run_mlp_training(config: PipelineConfig) -> dict:
    # MLP 실행
    tf = _import_tensorflow()
    split_data = load_tabular_splits(config)
    output_dir = ensure_output_dir(config)

    print_section("[INFO] MLP 학습 시작")
    class_weight = _make_class_weight(split_data["y_train"])
    run_results: list[dict] = []
    best_bundle: dict | None = None
    best_val_auc = -np.inf

    for seed in config.run_seeds:
        set_global_seed(seed, use_tensorflow=True)
        print(f"[INFO] MLP seed: {seed}")
        model = build_model(split_data["x_train"].shape[1], config.learning_rate)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=4,
                restore_best_weights=True,
                verbose=0,
            )
        ]
        history = model.fit(
            split_data["x_train"],
            split_data["y_train"],
            validation_data=(split_data["x_val"], split_data["y_val"]),
            epochs=config.mlp_epochs,
            batch_size=config.batch_size,
            verbose=0,
            class_weight=class_weight,
            callbacks=callbacks,
        )
        val_prob = model.predict(split_data["x_val"], verbose=0).ravel()
        test_prob = model.predict(split_data["x_test"], verbose=0).ravel()
        val_metrics = compute_metrics(split_data["y_val"], val_prob)
        test_metrics = compute_metrics(split_data["y_test"], test_prob)
        run_record = {
            "seed": seed,
            "epochs_ran": int(len(history.history["loss"])),
            "val_roc_auc": val_metrics["roc_auc"],
            **test_metrics,
        }
        run_results.append(run_record)

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_bundle = {
                "seed": seed,
                "model": model,
                "history": history.history,
                "test_prob": test_prob,
                "metrics": test_metrics,
            }

    if best_bundle is None:
        raise ValueError("MLP 학습 결과가 없습니다.")

    model_path = output_dir / "mlp_model.keras"
    best_bundle["model"].save(model_path)
    np.save(output_dir / "mlp_scaler_mean.npy", split_data["scaler_mean"])
    np.save(output_dir / "mlp_scaler_std.npy", split_data["scaler_std"])

    prediction_path = output_dir / "mlp_test_predictions.csv"
    save_prediction_frame(
        prediction_path,
        split_data["test_sample_ids"],
        split_data["y_test"],
        best_bundle["test_prob"],
    )
    save_confusion_matrix_plot(
        split_data["y_test"],
        best_bundle["test_prob"],
        output_dir / "confusion_matrix_mlp.png",
        title="MLP Confusion Matrix",
        cmap="Blues",
    )

    importance_frame = _permutation_importance(
        best_bundle["model"],
        split_data["x_test"],
        split_data["y_test"],
        split_data["feature_columns"],
        repeats=config.permutation_repeats,
        seed=best_bundle["seed"],
    )
    importance_frame.to_csv(output_dir / "mlp_feature_importance.csv", index=False)
    save_feature_importance_plot(
        importance_frame,
        output_dir / "mlp_feature_importance.png",
        title="MLP Feature Importance",
    )

    save_metrics_report(
        model_name="MLP",
        run_results=run_results,
        best_result=best_bundle["metrics"] | {"seed": best_bundle["seed"]},
        output_path=output_dir / "mlp_metrics.json",
        extra={
            "prediction_path": "mlp_test_predictions.csv",
            "feature_importance_path": "mlp_feature_importance.csv",
            "history": {
                "loss": [float(value) for value in best_bundle["history"]["loss"]],
                "val_loss": [float(value) for value in best_bundle["history"]["val_loss"]],
                "auc": [float(value) for value in best_bundle["history"]["auc"]],
                "val_auc": [float(value) for value in best_bundle["history"]["val_auc"]],
            },
        },
    )

    print(f"[INFO] MLP 저장: {model_path}")
    return {
        "model_name": "MLP",
        "metrics_path": str(output_dir / "mlp_metrics.json"),
        "prediction_path": str(prediction_path),
    }
