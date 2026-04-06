from pathlib import Path

from src.build_lstm_data import run_build_lstm_data
from src.evaluate import create_final_outputs
from src.preprocess import run_preprocess
from src.train_baselines import run_baseline_training
from src.train_lstm import run_lstm_training
from src.train_mlp import run_mlp_training
from src.utils import PipelineConfig, ensure_output_dir, print_section, save_json


def main() -> None:
    # 전체 실행
    config = PipelineConfig()
    output_dir = ensure_output_dir(config)
    save_json(output_dir / "run_config.json", config.to_dict())

    try:
        run_preprocess(config)
        run_build_lstm_data(config)
        run_baseline_training(config)
        run_mlp_training(config)
        run_lstm_training(config)
        summary = create_final_outputs(config)
    except Exception as error:
        print_section("[ERROR] 실행 실패")
        print(f"[ERROR] {error}")
        raise

    print_section("[RESULT] 최종 비교")
    for row in summary.to_dict(orient="records"):
        print(
            f"{row['model_name']:<20} "
            f"accuracy={row['accuracy']:.4f} "
            f"precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} "
            f"f1={row['f1_score']:.4f} "
            f"roc_auc={row['roc_auc']:.4f} "
            f"path={row['metrics_path']}"
        )
    print(f"[RESULT] outputs 경로: {Path(config.output_dir).resolve()}")


if __name__ == "__main__":
    main()
