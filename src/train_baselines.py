import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluate import compute_metrics, save_confusion_matrix_plot, save_metrics_report, save_prediction_frame
from src.utils import (
    PipelineConfig,
    ensure_output_dir,
    get_tabular_feature_columns,
    load_samples_frame,
    print_section,
    set_global_seed,
)


def load_tabular_splits(config: PipelineConfig) -> dict:
    # 탭형 데이터 로드
    print_section("[INFO] Baseline 데이터 로드")
    samples = load_samples_frame(config)
    feature_columns = get_tabular_feature_columns(samples)

    split_frames = {
        split_name: samples.loc[samples["split"] == split_name].copy()
        for split_name in ["train", "val", "test"]
    }

    scaler = StandardScaler()
    x_train = scaler.fit_transform(split_frames["train"][feature_columns].to_numpy(dtype=np.float32))
    x_val = scaler.transform(split_frames["val"][feature_columns].to_numpy(dtype=np.float32))
    x_test = scaler.transform(split_frames["test"][feature_columns].to_numpy(dtype=np.float32))

    return {
        "feature_columns": feature_columns,
        "scaler": scaler,
        "x_train": x_train,
        "y_train": split_frames["train"]["label"].to_numpy(dtype=np.int32),
        "x_val": x_val,
        "y_val": split_frames["val"]["label"].to_numpy(dtype=np.int32),
        "x_test": x_test,
        "y_test": split_frames["test"]["label"].to_numpy(dtype=np.int32),
        "test_sample_ids": split_frames["test"]["sample_id"].to_numpy(dtype=np.int32),
    }


def _train_dummy(split_data: dict) -> dict:
    # 더미 학습
    model = DummyClassifier(strategy="prior")
    model.fit(split_data["x_train"], split_data["y_train"])
    y_prob = model.predict_proba(split_data["x_test"])[:, 1]
    metrics = compute_metrics(split_data["y_test"], y_prob)
    return {
        "model": model,
        "metrics": metrics,
        "y_prob": y_prob,
    }


def _train_logistic(split_data: dict, seed: int) -> dict:
    # 로지스틱 학습
    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(split_data["x_train"], split_data["y_train"])
    y_prob = model.predict_proba(split_data["x_test"])[:, 1]
    val_prob = model.predict_proba(split_data["x_val"])[:, 1]
    test_metrics = compute_metrics(split_data["y_test"], y_prob)
    val_metrics = compute_metrics(split_data["y_val"], val_prob)
    return {
        "model": model,
        "metrics": test_metrics,
        "val_metrics": val_metrics,
        "y_prob": y_prob,
    }


def run_baseline_training(config: PipelineConfig) -> dict:
    # baseline 실행
    split_data = load_tabular_splits(config)
    output_dir = ensure_output_dir(config)
    results: list[dict] = []

    print_section("[INFO] Baseline 학습 시작")
    dummy_result = _train_dummy(split_data)
    dummy_prediction_name = "dummy_test_predictions.csv"
    save_prediction_frame(
        output_dir / dummy_prediction_name,
        split_data["test_sample_ids"],
        split_data["y_test"],
        dummy_result["y_prob"],
    )
    save_confusion_matrix_plot(
        split_data["y_test"],
        dummy_result["y_prob"],
        output_dir / "confusion_matrix_dummy.png",
        title="Dummy Confusion Matrix",
        cmap="Greys",
    )
    results.append(
        {
            "model_name": "DummyClassifier",
            "run_count": 1,
            "aggregate": {
                "accuracy_mean": dummy_result["metrics"]["accuracy"],
                "accuracy_std": 0.0,
                "precision_mean": dummy_result["metrics"]["precision"],
                "precision_std": 0.0,
                "recall_mean": dummy_result["metrics"]["recall"],
                "recall_std": 0.0,
                "f1_score_mean": dummy_result["metrics"]["f1_score"],
                "f1_score_std": 0.0,
                "roc_auc_mean": dummy_result["metrics"]["roc_auc"],
                "roc_auc_std": 0.0,
            },
            "best_run": dummy_result["metrics"],
            "runs": [dummy_result["metrics"]],
            "prediction_path": dummy_prediction_name,
            "roc_color": "dimgray",
        }
    )

    logistic_runs: list[dict] = []
    best_logistic: dict | None = None
    best_score = -np.inf

    for seed in config.run_seeds:
        set_global_seed(seed)
        print(f"[INFO] LogisticRegression seed: {seed}")
        result = _train_logistic(split_data, seed)
        run_metrics = {
            "seed": seed,
            **result["metrics"],
            "val_roc_auc": result["val_metrics"]["roc_auc"],
        }
        logistic_runs.append(run_metrics)
        if result["val_metrics"]["roc_auc"] > best_score:
            best_score = result["val_metrics"]["roc_auc"]
            best_logistic = result | {"seed": seed}

    if best_logistic is None:
        raise ValueError("LogisticRegression 학습 결과가 없습니다.")

    logistic_prediction_name = "logistic_regression_test_predictions.csv"
    save_prediction_frame(
        output_dir / logistic_prediction_name,
        split_data["test_sample_ids"],
        split_data["y_test"],
        best_logistic["y_prob"],
    )
    save_confusion_matrix_plot(
        split_data["y_test"],
        best_logistic["y_prob"],
        output_dir / "confusion_matrix_logistic_regression.png",
        title="Logistic Regression Confusion Matrix",
        cmap="Purples",
    )
    with (output_dir / "logistic_regression_model.pkl").open("wb") as file:
        pickle.dump(
            {
                "model": best_logistic["model"],
                "scaler": split_data["scaler"],
                "feature_columns": split_data["feature_columns"],
            },
            file,
        )

    coefficient_frame = pd.DataFrame(
        {
            "feature": split_data["feature_columns"],
            "coefficient": best_logistic["model"].coef_.ravel(),
            "absolute_coefficient": np.abs(best_logistic["model"].coef_.ravel()),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficient_frame.to_csv(output_dir / "logistic_regression_coefficients.csv", index=False)

    save_metrics_report(
        model_name="LogisticRegression",
        run_results=logistic_runs,
        best_result=best_logistic["metrics"] | {"seed": best_logistic["seed"]},
        output_path=output_dir / "logistic_regression_metrics.json",
        extra={"prediction_path": logistic_prediction_name},
    )

    logistic_payload = {
        "model_name": "LogisticRegression",
        "run_count": len(logistic_runs),
        "aggregate": {
            "accuracy_mean": float(np.mean([run["accuracy"] for run in logistic_runs])),
            "accuracy_std": float(np.std([run["accuracy"] for run in logistic_runs])),
            "precision_mean": float(np.mean([run["precision"] for run in logistic_runs])),
            "precision_std": float(np.std([run["precision"] for run in logistic_runs])),
            "recall_mean": float(np.mean([run["recall"] for run in logistic_runs])),
            "recall_std": float(np.std([run["recall"] for run in logistic_runs])),
            "f1_score_mean": float(np.mean([run["f1_score"] for run in logistic_runs])),
            "f1_score_std": float(np.std([run["f1_score"] for run in logistic_runs])),
            "roc_auc_mean": float(np.mean([run["roc_auc"] for run in logistic_runs])),
            "roc_auc_std": float(np.std([run["roc_auc"] for run in logistic_runs])),
        },
        "best_run": best_logistic["metrics"] | {"seed": best_logistic["seed"]},
        "runs": logistic_runs,
        "prediction_path": logistic_prediction_name,
        "roc_color": "darkorange",
    }
    results.append(logistic_payload)

    baseline_payload = {
        "models": results,
        "feature_columns": split_data["feature_columns"],
        "test_size": int(len(split_data["y_test"])),
    }
    with (output_dir / "baseline_metrics.json").open("w", encoding="utf-8") as file:
        import json

        json.dump(baseline_payload, file, ensure_ascii=False, indent=2)

    print(f"[INFO] baseline 저장: {output_dir / 'baseline_metrics.json'}")
    return baseline_payload
