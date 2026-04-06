import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PipelineConfig:
    # 기본 설정
    orders_path: str = "data/orders.csv"
    feature_path_candidates: tuple[str, ...] = (
        "data/customer_features.csv",
        "data/script_job_3ea69affb682fff4480bec5a3efe5361_1.csv",
    )
    output_dir: str = "outputs"
    high_value_quantile: float = 0.8
    delay_threshold: int = 15
    seq_len: int = 5
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    run_seeds: list[int] = field(default_factory=lambda: [42])
    max_recent_orders: int | None = None
    mlp_epochs: int = 30
    lstm_epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 0.001
    cv_splits: int = 3
    cv_sample_size: int = 30000
    permutation_repeats: int = 3

    def to_dict(self) -> dict:
        # 설정 변환
        return asdict(self)


def print_section(title: str) -> None:
    # 구간 출력
    line = "=" * 50
    print(f"\n{line}\n{title}\n{line}")


def ensure_output_dir(config: PipelineConfig) -> Path:
    # 출력 경로 생성
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir = output_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))
    return output_dir


def set_global_seed(seed: int, use_tensorflow: bool = False) -> None:
    # 시드 고정
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    if use_tensorflow:
        try:
            import tensorflow as tf

            tf.keras.utils.set_random_seed(seed)
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
        except ImportError:
            raise ImportError("tensorflow가 설치되어 있지 않습니다.")


def resolve_existing_path(path_text: str) -> Path:
    # 파일 확인
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path_text}")
    return path


def resolve_feature_path(config: PipelineConfig) -> Path:
    # 피처 파일 확인
    for candidate in config.feature_path_candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    joined = ", ".join(config.feature_path_candidates)
    raise FileNotFoundError(f"피처 파일을 찾을 수 없습니다: {joined}")


def require_columns(frame: pd.DataFrame, required: Iterable[str], context: str) -> None:
    # 컬럼 확인
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} 필수 컬럼이 없습니다: {missing}")


def save_json(path: Path, payload: dict) -> None:
    # JSON 저장
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    # JSON 로드
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_text(path: Path, text: str) -> None:
    # 텍스트 저장
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def check_binary_labels(labels: np.ndarray, context: str) -> None:
    # 라벨 확인
    unique_values = np.unique(labels)
    if len(unique_values) < 2:
        raise ValueError(f"{context} 라벨이 한 클래스만 있습니다: {unique_values.tolist()}")


def get_samples_path(config: PipelineConfig) -> Path:
    # 샘플 경로
    return Path(config.output_dir) / "samples.csv"


def get_tabular_feature_names_path(config: PipelineConfig) -> Path:
    # 피처명 경로
    return Path(config.output_dir) / "tabular_feature_names.json"


def get_sequence_feature_names_path(config: PipelineConfig) -> Path:
    # 시퀀스 피처명 경로
    return Path(config.output_dir) / "sequence_feature_names.json"


def load_samples_frame(config: PipelineConfig) -> pd.DataFrame:
    # 샘플 로드
    sample_path = get_samples_path(config)
    if not sample_path.exists():
        raise FileNotFoundError(f"전처리 결과가 없습니다: {sample_path}")
    return pd.read_csv(sample_path)


def get_tabular_feature_columns(samples: pd.DataFrame) -> list[str]:
    # 탭형 피처 선택
    columns = [column for column in samples.columns if column.startswith("hist_")]
    if not columns:
        raise ValueError("탭형 피처 컬럼이 없습니다.")
    return columns


def get_sequence_array_paths(config: PipelineConfig) -> dict[str, Path]:
    # 배열 경로
    output_dir = Path(config.output_dir)
    return {
        "x_train": output_dir / "X_train_seq.npy",
        "x_val": output_dir / "X_val_seq.npy",
        "x_test": output_dir / "X_test_seq.npy",
        "y_train": output_dir / "y_train.npy",
        "y_val": output_dir / "y_val.npy",
        "y_test": output_dir / "y_test.npy",
        "test_sample_ids": output_dir / "test_sample_ids.npy",
    }


def get_split_sizes(samples: pd.DataFrame) -> dict[str, int]:
    # 분할 개수
    return samples["split"].value_counts().sort_index().to_dict()
