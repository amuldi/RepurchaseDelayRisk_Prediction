import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import PipelineConfig, ensure_output_dir, load_json, print_section, save_json, save_text


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    # 성능 계산
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int32)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def aggregate_run_metrics(run_results: list[dict]) -> dict:
    # 반복 결과 집계
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    summary: dict[str, float] = {}
    for name in metric_names:
        values = np.asarray([result[name] for result in run_results], dtype=np.float32)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=0))
    return summary


def save_metrics_report(
    model_name: str,
    run_results: list[dict],
    best_result: dict,
    output_path: Path,
    extra: dict | None = None,
) -> None:
    # 성능 저장
    payload = {
        "model_name": model_name,
        "run_count": len(run_results),
        "aggregate": aggregate_run_metrics(run_results),
        "best_run": best_result,
        "runs": run_results,
    }
    if extra:
        payload.update(extra)
    save_json(output_path, payload)


def save_prediction_frame(path: Path, sample_ids: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    # 예측 저장
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids.astype(int),
            "y_true": y_true.astype(int),
            "y_prob": y_prob.astype(float),
        }
    )
    frame.to_csv(path, index=False)


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str,
    threshold: float = 0.5,
    cmap: str = "Blues",
) -> None:
    # 혼동행렬 저장
    y_pred = (np.asarray(y_prob) >= threshold).astype(np.int32)
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(matrix).plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_feature_importance_plot(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    # 중요도 저장
    ordered = frame.sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered["feature"], ordered["importance_mean"], xerr=ordered["importance_std"], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("ROC-AUC Drop")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_final_outputs(config: PipelineConfig) -> pd.DataFrame:
    # 최종 비교 생성
    print_section("[RESULT] 최종 비교")
    output_dir = ensure_output_dir(config)

    metric_sources = [
        output_dir / "baseline_metrics.json",
        output_dir / "mlp_metrics.json",
        output_dir / "lstm_metrics.json",
    ]
    for path in metric_sources:
        if not path.exists():
            raise FileNotFoundError(f"평가 파일이 없습니다: {path}")

    baseline_payload = load_json(metric_sources[0])
    mlp_payload = load_json(metric_sources[1])
    lstm_payload = load_json(metric_sources[2])

    rows: list[dict] = []
    roc_sources: list[tuple[str, Path, str]] = []

    for baseline in baseline_payload["models"]:
        rows.append(
            {
                "model_name": baseline["model_name"],
                "accuracy": baseline["aggregate"]["accuracy_mean"],
                "precision": baseline["aggregate"]["precision_mean"],
                "recall": baseline["aggregate"]["recall_mean"],
                "f1_score": baseline["aggregate"]["f1_score_mean"],
                "roc_auc": baseline["aggregate"]["roc_auc_mean"],
                "run_count": baseline["run_count"],
                "metrics_path": "outputs/baseline_metrics.json",
            }
        )
        roc_sources.append(
            (
                baseline["model_name"],
                output_dir / baseline["prediction_path"],
                baseline["roc_color"],
            )
        )

    for payload, prediction_name, color in [
        (mlp_payload, "mlp_test_predictions.csv", "royalblue"),
        (lstm_payload, "lstm_test_predictions.csv", "seagreen"),
    ]:
        rows.append(
            {
                "model_name": payload["model_name"],
                "accuracy": payload["aggregate"]["accuracy_mean"],
                "precision": payload["aggregate"]["precision_mean"],
                "recall": payload["aggregate"]["recall_mean"],
                "f1_score": payload["aggregate"]["f1_score_mean"],
                "roc_auc": payload["aggregate"]["roc_auc_mean"],
                "run_count": payload["run_count"],
                "metrics_path": f"outputs/{payload['model_name'].lower()}_metrics.json",
            }
        )
        roc_sources.append((payload["model_name"], output_dir / prediction_name, color))

    summary = pd.DataFrame(rows).sort_values(["f1_score", "roc_auc"], ascending=False).reset_index(drop=True)
    summary_path = output_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, prediction_path, color in roc_sources:
        frame = pd.read_csv(prediction_path)
        fpr, tpr, _ = roc_curve(frame["y_true"], frame["y_prob"])
        roc_auc = roc_auc_score(frame["y_true"], frame["y_prob"])
        ax.plot(fpr, tpr, linewidth=2, color=color, label=f"{model_name} (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    best_row = summary.iloc[0]
    explanation_lines = [
        "실험 공정성:",
        "- 모든 모델은 같은 고빈도 고객 집단을 사용했습니다.",
        "- 모든 모델은 같은 label 정의를 사용했습니다.",
        "- 모든 모델은 같은 시점 기준 split을 사용했습니다.",
        "",
        "최종 결론:",
        f"- 가장 높은 F1-score 모델: {best_row['model_name']}",
        f"- Accuracy: {best_row['accuracy']:.4f}",
        f"- Precision: {best_row['precision']:.4f}",
        f"- Recall: {best_row['recall']:.4f}",
        f"- F1-score: {best_row['f1_score']:.4f}",
        f"- ROC-AUC: {best_row['roc_auc']:.4f}",
        "",
        "해석:",
        "- DummyClassifier 대비 개선되면 학습 모델이 의미 있다고 볼 수 있습니다.",
        "- LogisticRegression과 MLP는 탭형 과거 요약 피처를 사용합니다.",
        "- LSTM은 최근 주문 흐름을 직접 사용하므로 시퀀스 패턴 반영 여부를 볼 수 있습니다.",
    ]
    save_text(output_dir / "final_summary.txt", "\n".join(explanation_lines))

    print(summary.to_string(index=False))
    print(f"[RESULT] 비교 저장: {summary_path}")
    print(f"[RESULT] ROC 저장: {roc_path}")
    print(f"[RESULT] 요약 저장: {output_dir / 'final_summary.txt'}")
    return summary
