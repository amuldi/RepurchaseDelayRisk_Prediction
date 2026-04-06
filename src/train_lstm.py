import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.evaluate import compute_metrics, save_confusion_matrix_plot, save_metrics_report, save_prediction_frame
from src.utils import (
    PipelineConfig,
    check_binary_labels,
    ensure_output_dir,
    get_sequence_array_paths,
    print_section,
    set_global_seed,
)


def _import_tensorflow():
    # 텐서플로 확인
    try:
        import tensorflow as tf

        return tf
    except ImportError as error:
        raise ImportError("LSTM 학습에는 tensorflow가 필요합니다.") from error


def load_sequence_splits(config: PipelineConfig) -> dict:
    # 시퀀스 로드
    print_section("[INFO] LSTM 시퀀스 로드")
    paths = get_sequence_array_paths(config)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"LSTM 입력 파일이 없습니다: {missing}")

    split_data = {
        "x_train": np.load(paths["x_train"]).astype(np.float32),
        "x_val": np.load(paths["x_val"]).astype(np.float32),
        "x_test": np.load(paths["x_test"]).astype(np.float32),
        "y_train": np.load(paths["y_train"]).astype(np.int32),
        "y_val": np.load(paths["y_val"]).astype(np.int32),
        "y_test": np.load(paths["y_test"]).astype(np.int32),
        "test_sample_ids": np.load(paths["test_sample_ids"]).astype(np.int32),
    }

    check_binary_labels(split_data["y_train"], "lstm_train")
    check_binary_labels(split_data["y_val"], "lstm_val")
    check_binary_labels(split_data["y_test"], "lstm_test")

    print(f"[INFO] train shape: {split_data['x_train'].shape}")
    print(f"[INFO] val shape: {split_data['x_val'].shape}")
    print(f"[INFO] test shape: {split_data['x_test'].shape}")
    return split_data


def build_model(input_shape: tuple[int, int], learning_rate: float):
    # 모델 정의
    tf = _import_tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
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
    # 가중치 계산
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def run_lstm_training(config: PipelineConfig) -> dict:
    # LSTM 실행
    tf = _import_tensorflow()
    split_data = load_sequence_splits(config)
    output_dir = ensure_output_dir(config)

    print_section("[INFO] LSTM 학습 시작")
    class_weight = _make_class_weight(split_data["y_train"])
    run_results: list[dict] = []
    best_bundle: dict | None = None
    best_val_auc = -np.inf

    for seed in config.run_seeds:
        set_global_seed(seed, use_tensorflow=True)
        print(f"[INFO] LSTM seed: {seed}")
        model = build_model(
            (split_data["x_train"].shape[1], split_data["x_train"].shape[2]),
            config.learning_rate,
        )
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
            epochs=config.lstm_epochs,
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
        raise ValueError("LSTM 학습 결과가 없습니다.")

    model_path = output_dir / "lstm_model.keras"
    best_bundle["model"].save(model_path)
    prediction_path = output_dir / "lstm_test_predictions.csv"
    save_prediction_frame(
        prediction_path,
        split_data["test_sample_ids"],
        split_data["y_test"],
        best_bundle["test_prob"],
    )
    save_confusion_matrix_plot(
        split_data["y_test"],
        best_bundle["test_prob"],
        output_dir / "confusion_matrix_lstm.png",
        title="LSTM Confusion Matrix",
        cmap="Greens",
    )

    save_metrics_report(
        model_name="LSTM",
        run_results=run_results,
        best_result=best_bundle["metrics"] | {"seed": best_bundle["seed"]},
        output_path=output_dir / "lstm_metrics.json",
        extra={
            "prediction_path": "lstm_test_predictions.csv",
            "interpretation_note": "최근 주문 간격과 주문 시퀀스 흐름을 직접 반영한 모델입니다.",
            "history": {
                "loss": [float(value) for value in best_bundle["history"]["loss"]],
                "val_loss": [float(value) for value in best_bundle["history"]["val_loss"]],
                "auc": [float(value) for value in best_bundle["history"]["auc"]],
                "val_auc": [float(value) for value in best_bundle["history"]["val_auc"]],
            },
        },
    )

    print(f"[INFO] LSTM 저장: {model_path}")
    return {
        "model_name": "LSTM",
        "metrics_path": str(output_dir / "lstm_metrics.json"),
        "prediction_path": str(prediction_path),
    }
