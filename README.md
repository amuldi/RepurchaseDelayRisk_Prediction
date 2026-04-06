# High-Frequency Customer Repurchase Delay Risk Prediction

고빈도 고객의 재구매 지연 위험을 예측하고 `DummyClassifier`, `LogisticRegression`, `MLP`, `LSTM`을 공정하게 비교하는 프로젝트입니다.

## 프로젝트 목표

- 대상: 주문 횟수가 많은 고빈도 고객
- 라벨: `days_since_prior_order > 15`
- 비교 포인트: 단순 기준 모델 대비 `MLP`와 `LSTM`이 실제로 더 적합한지 검증

## 데이터

- 출처: Kaggle
- 데이터셋: Instacart Online Grocery Basket Analysis
- 링크: [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis)

사용 파일:

- `data/orders.csv`
- `data/customer_features.csv`

## 실험 설계 핵심

### 1. 누수 방지

- 예측 시점 `t`의 피처는 `t` 이전 주문 이력만 사용합니다.
- `total_orders`, `avg_days_between_orders` 같은 고객 요약값은 원본 파일 값을 그대로 쓰지 않고, 각 샘플 시점 직전 이력으로 다시 계산합니다.
- `customer_features.csv`는 고빈도 고객 집단을 정의하는 용도로만 사용합니다.

### 2. 시점 기준 분할

- 무작위 split을 쓰지 않습니다.
- 각 고객의 주문 시퀀스를 시간순으로 정렬한 뒤:
  - 앞 구간은 `train`
  - 중간 구간은 `val`
  - 마지막 구간은 `test`
- 따라서 미래 주문이 학습에 섞이지 않습니다.

### 3. 공정한 비교

- 모든 모델은 같은 고빈도 고객 집단을 사용합니다.
- 모든 모델은 같은 라벨 정의를 사용합니다.
- 모든 모델은 같은 `train/val/test` 샘플을 사용합니다.
- 차이는 입력 표현만 다릅니다.
  - `LogisticRegression`, `MLP`: 과거 이력 기반 탭형 피처
  - `LSTM`: 최근 주문 시퀀스 피처

### 4. 반복 실험

- `run_seeds` 설정으로 반복 실험이 가능하도록 구성했습니다.
- 결과는 평균과 표준편차로 저장됩니다.

## 프로젝트 구조

```text
.
├── run_all.py
├── requirements.txt
├── README.md
├── src
│   ├── __init__.py
│   ├── utils.py
│   ├── preprocess.py
│   ├── build_lstm_data.py
│   ├── train_baselines.py
│   ├── train_mlp.py
│   ├── train_lstm.py
│   └── evaluate.py
├── data
│   ├── customer_features.csv
│   └── orders.csv
└── outputs
```

## 실행 방법

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 전체 실행

```bash
python run_all.py
```

## 주요 출력 파일

모든 주요 결과는 `outputs/` 아래에 자동 저장됩니다.

- `outputs/samples.csv`
- `outputs/X_train_seq.npy`
- `outputs/X_val_seq.npy`
- `outputs/X_test_seq.npy`
- `outputs/baseline_metrics.json`
- `outputs/logistic_regression_metrics.json`
- `outputs/mlp_metrics.json`
- `outputs/lstm_metrics.json`
- `outputs/comparison_summary.csv`
- `outputs/roc_curve.png`
- `outputs/confusion_matrix_dummy.png`
- `outputs/confusion_matrix_logistic_regression.png`
- `outputs/confusion_matrix_mlp.png`
- `outputs/confusion_matrix_lstm.png`
- `outputs/mlp_feature_importance.csv`
- `outputs/final_summary.txt`

## 모델 해석 기준

이 과제는 위험 고객을 놓치지 않는 것이 중요하므로 아래 지표를 중심으로 해석합니다.

- `Recall`
- `F1-score`
- `ROC-AUC`

해석 예시는 아래와 같습니다.

- `DummyClassifier`보다 좋아야 학습 모델이 의미가 있습니다.
- `LogisticRegression`보다 `MLP`가 좋으면 비선형 탭형 모델이 더 적합하다고 볼 수 있습니다.
- `MLP`보다 `LSTM`이 좋으면 최근 주문 흐름과 간격 변화가 중요하다고 해석할 수 있습니다.

## 실행 순서

`run_all.py`는 아래 순서로 전체 파이프라인을 실행합니다.

1. 고빈도 고객 필터링
2. 누수 방지 탭형 샘플 생성
3. LSTM 시퀀스 배열 생성
4. `DummyClassifier`, `LogisticRegression` 학습
5. `MLP` 학습
6. `LSTM` 학습
7. 최종 비교표와 요약 저장

## 참고

- `tensorflow`가 없으면 `MLP`, `LSTM` 학습 단계에서 오류가 발생합니다.
- 기본 설정은 `src/utils.py`의 `PipelineConfig`에서 변경할 수 있습니다.
- 반복 실험을 늘리고 싶으면 `run_seeds`를 늘리면 됩니다.
