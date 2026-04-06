# High-Frequency Customer Repurchase Delay Risk Prediction

고빈도 고객의 재구매 지연 위험을 예측하고 `MLP`, `LSTM`을 중심으로 비교하는 프로젝트입니다.

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
├── 02_train_mlp.ipynb
├── 03_train_lstm.ipynb
├── 04_compare_models.ipynb
├── README.md
├── requirements.txt
└── data
    ├── customer_features.csv
    └── orders.csv
```

## 노트북 역할

### `01_prepare_data.ipynb`

- 고빈도 고객 추출
- 누수 없는 탭형 피처 생성
- 시점 기준 `train / val / test` 분할
- LSTM용 시퀀스 배열 생성
- 전처리 결과를 `outputs/`에 저장

### `02_train_mlp.ipynb`

- 전처리 결과를 사용해 `MLP` 학습
- `MLP` 성능 저장
- 학습 곡선과 혼동행렬 표시

### `03_train_lstm.ipynb`

- 전처리 결과를 사용해 `LSTM` 학습
- `LSTM` 성능 저장
- 학습 곡선과 혼동행렬 표시

### `04_compare_models.ipynb`

- `DummyClassifier`
- `LogisticRegression`
- 저장된 `MLP`
- 저장된 `LSTM`

위 네 모델을 같은 테스트셋 기준으로 비교합니다.

## 실행 방법

아래 순서대로 실행하면 됩니다.

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 전처리 실행

```text
01_prepare_data.ipynb
```

이 노트북이 끝나면 `outputs/`에 학습용 데이터가 저장됩니다.

### 3. MLP 학습

```text
02_train_mlp.ipynb
```

이 노트북이 끝나면 `MLP` 결과가 `outputs/`에 저장됩니다.

### 4. LSTM 학습

```text
03_train_lstm.ipynb
```

이 노트북이 끝나면 `LSTM` 결과가 `outputs/`에 저장됩니다.

### 5. 모델 비교

```text
04_compare_models.ipynb
```

이 노트북에서 baseline, `MLP`, `LSTM`을 한 번에 비교합니다.

## 주요 결과 파일

모든 주요 결과는 `outputs/`에 자동 저장됩니다.

예시:

- `outputs/samples.csv`
- `outputs/X_train_seq.npy`
- `outputs/X_val_seq.npy`
- `outputs/X_test_seq.npy`
- `outputs/mlp_metrics.json`
- `outputs/lstm_metrics.json`
- `outputs/baseline_metrics.json`
- `outputs/comparison_summary.csv`
- `outputs/final_summary.txt`

## 해석 기준

이 프로젝트는 위험 고객을 놓치지 않는 것이 중요하므로 아래 지표를 중심으로 봅니다.

- `Recall`
- `F1-score`
- `ROC-AUC`

해석 방식:

- `DummyClassifier`보다 높아야 학습 모델이 의미가 있습니다.
- `LogisticRegression`과 `MLP`는 과거 요약 피처를 사용합니다.
- `LSTM`은 최근 주문 흐름을 직접 사용합니다.
- `LSTM`이 더 좋다면 시계열 정보가 중요한 문제라고 해석할 수 있습니다.

## 주의

- 반드시 `01_prepare_data.ipynb`를 먼저 실행하세요.
- `02_train_mlp.ipynb`, `03_train_lstm.ipynb`를 실행한 뒤 `04_compare_models.ipynb`를 실행하세요.
- GitHub에는 실행 결과가 포함된 노트북 형태로 업로드할 수 있습니다.
