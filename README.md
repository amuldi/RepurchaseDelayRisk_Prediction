# High-Frequency Customer Repurchase Delay Risk Prediction

고빈도 고객의 재구매 지연 위험을 예측하고 `DummyClassifier`, `LogisticRegression`, `MLP`, `LSTM`을 비교하는 프로젝트입니다.

## 데이터

- 출처: Kaggle
- 데이터셋: Instacart Online Grocery Basket Analysis
- 링크: [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis)

사용 파일:

- `data/orders.csv`
- `data/customer_features.csv`

## 파일 구조

```text
.
├── 01_prepare_data.ipynb
├── 02_compare_models.ipynb
├── README.md
├── requirements.txt
└── data
    ├── customer_features.csv
    └── orders.csv
```

## 노트북 설명

### `01_prepare_data.ipynb`

- 고빈도 고객 추출
- 누수 없는 탭형 피처 생성
- 시점 기준 `train / val / test` 분할
- LSTM용 시퀀스 배열 생성
- 결과를 `outputs/`에 저장

### `02_compare_models.ipynb`

- 같은 샘플 기준으로 모델 비교
- 비교 모델:
  - `DummyClassifier`
  - `LogisticRegression`
  - `MLP`
  - `LSTM`
- ROC Curve, Confusion Matrix 표시
- 비교 결과를 `outputs/`에 저장

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 전처리 실행

Jupyter에서 아래 노트북을 먼저 실행합니다.

```text
01_prepare_data.ipynb
```

이 노트북이 끝나면 `outputs/`에 전처리 결과가 저장됩니다.

### 3. 모델 비교 실행

그다음 아래 노트북을 실행합니다.

```text
02_compare_models.ipynb
```

이 노트북이 끝나면:

- 모델별 성능표
- ROC Curve
- Confusion Matrix
- 최종 비교 결과

를 바로 화면에서 볼 수 있습니다.

## 결과 저장 위치

주요 결과는 `outputs/`에 자동 저장됩니다.

예시:

- `outputs/samples.csv`
- `outputs/X_train_seq.npy`
- `outputs/X_val_seq.npy`
- `outputs/X_test_seq.npy`
- `outputs/comparison_summary.csv`
- `outputs/baseline_metrics.json`
- `outputs/mlp_metrics.json`
- `outputs/lstm_metrics.json`
- `outputs/final_summary.txt`
- `outputs/roc_curve.png`

## 해석 기준

이 프로젝트는 위험 고객을 놓치지 않는 것이 중요하므로 아래 지표를 중심으로 봅니다.

- `Recall`
- `F1-score`
- `ROC-AUC`

해석 방식:

- `DummyClassifier`보다 높아야 학습 모델이 의미가 있습니다.
- `LogisticRegression`, `MLP`는 과거 요약 피처를 사용합니다.
- `LSTM`은 최근 주문 흐름을 직접 사용합니다.
- `LSTM`이 더 좋다면 시계열 정보가 중요한 문제라고 해석할 수 있습니다.

## 주의

- 반드시 `01_prepare_data.ipynb`를 먼저 실행하세요.
- `tensorflow`가 설치되어 있어야 `MLP`, `LSTM`이 실행됩니다.
- GitHub에는 실행 결과가 포함된 노트북 형태로 업로드할 수 있습니다.
