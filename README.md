# High-Frequency Customer Repurchase Delay Prediction

고빈도 고객의 재구매 지연 위험을 예측하고, `MLP`와 `LSTM`의 성능을 비교하는 프로젝트입니다.

## 프로젝트 목표

- 대상: 주문 활동이 많은 고빈도 고객
- 문제: 재구매가 평소보다 늦어지는 위험 상태 예측
- 비교 모델: `DummyClassifier`, `MLP`, `LSTM`

## 위험 정의

재구매 지연 위험은 아래 기준으로 정의합니다.

```python
label = 1 if days_since_prior_order > 15 else 0
```

즉, 이전 주문 이후 15일을 초과하면 위험 고객으로 봅니다.

## 데이터

- 출처: Kaggle
- 데이터셋: Instacart Online Grocery Basket Analysis
- 링크: [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis)

사용 파일:

- `data/orders.csv`
- `data/customer_features.csv`

## 사용 모델

### 1. MLP

- 입력: `customer_features.csv`의 고객 집계 피처
- 특징: 정적 피처 기반 분류 모델
- 목적: 간단한 기준 모델

사용 피처 예시:

- `total_orders`
- `avg_days_between_orders`
- `max_order_number`

### 2. LSTM

- 입력: `orders.csv`로 만든 주문 시퀀스 데이터
- 특징: 최근 주문 흐름을 반영하는 시계열 모델
- 목적: 재구매 지연 패턴 학습

시퀀스 피처 예시:

- `days_since_prior_order`
- `order_number`
- `order_dow`
- `order_hour_of_day`
- 고객 정적 피처 반복 결합

### 3. DummyClassifier

- 입력: 동일한 분류 문제
- 목적: 학습 모델이 단순 기준보다 실제로 나은지 확인

## 프로젝트 구조

```text
.
├── build_lstm_data.ipynb
├── train_mlp.ipynb
├── train_lstm.ipynb
├── compare_models.ipynb
├── requirements.txt
├── data
│   ├── customer_features.csv
│   └── orders.csv
└── outputs
```

## 노트북 설명

### `build_lstm_data.ipynb`

- `orders.csv`를 LSTM용 시퀀스 데이터로 변환
- 고빈도 고객만 필터링
- rolling window 방식으로 입력 시퀀스 생성

생성 파일:

- `outputs/X_seq.npy`
- `outputs/y.npy`
- `outputs/lstm_sequence_metadata.csv`

### `train_mlp.ipynb`

- 고객 집계 피처로 `MLP` 학습
- 성능 지표 계산
- 결과를 노트북 셀에서 바로 시각화

### `train_lstm.ipynb`

- 시퀀스 데이터로 `LSTM` 학습
- 성능 지표 계산
- 결과를 노트북 셀에서 바로 시각화

### `compare_models.ipynb`

- `DummyClassifier`, `MLP`, `LSTM` 성능 비교
- 단일 테스트셋 비교 수행
- `StratifiedKFold` 교차검증 수행
- 결과를 표와 시각화로 바로 확인

## 비교 방법

단일 split 결과만으로 모델이 좋다고 판단하지 않기 위해 두 가지 방식으로 비교합니다.

### 1. 단일 테스트셋 비교

- 같은 테스트셋에서 세 모델 성능 비교
- 실제 발표용 직관적인 결과 확인

### 2. StratifiedKFold 교차검증

- 클래스 비율을 유지한 상태로 여러 fold 평가
- 단일 split의 우연이 아니라는 점 확인

## 주요 평가 지표

이 과제는 위험 고객 탐지가 중요하므로 아래 지표를 중심으로 해석합니다.

- `Recall`
- `F1-score`
- `ROC-AUC`

## 결과 해석 방법

이 프로젝트에서 가장 중요한 질문은 아래입니다.

- `MLP`와 `LSTM` 중 어떤 모델이 더 적합한가?

해석 기준은 다음과 같습니다.

- `Recall`이 높을수록 위험 고객을 더 놓치지 않습니다.
- `F1-score`가 높을수록 위험 고객 탐지 성능이 더 균형적입니다.
- `ROC-AUC`가 높을수록 전반적인 분류 성능이 더 좋습니다.

쉽게 말하면:

- `MLP`가 좋다면
  - 고객의 요약된 집계 피처만으로도 위험 예측이 충분하다는 뜻입니다.

- `LSTM`이 좋다면
  - 최근 주문 흐름과 재구매 패턴 같은 시계열 정보가 중요하다는 뜻입니다.

이 과제에서는 재구매 지연이 시간 흐름과 연결되어 있으므로, 보통 `LSTM`이 더 적합한 해석이 자연스럽습니다.

## 기대 결론

발표나 보고서에서는 아래 흐름으로 결론을 정리하면 됩니다.

1. `DummyClassifier`보다 `MLP`, `LSTM`이 좋아야 합니다.
2. `MLP`보다 `LSTM`의 `Recall`, `F1-score`, `ROC-AUC`가 높으면 시계열 정보가 실제로 도움이 된다고 볼 수 있습니다.
3. `StratifiedKFold` 평균에서도 같은 경향이 유지되면, 단일 split의 우연이 아니라는 근거가 됩니다.

즉, 최종 결론은 보통 아래처럼 정리할 수 있습니다.

`LSTM`은 최근 주문 순서와 간격 변화를 반영할 수 있기 때문에, 고빈도 고객의 재구매 지연 위험 예측에 더 적합한 모델이다.

## 시각화

비교 노트북에서는 아래 두 가지 시각화만 사용합니다.

- `ROC Curve`
- `Confusion Matrix`

모든 결과는 `png` 저장이 아니라 노트북 실행 후 셀 아래에서 바로 확인할 수 있습니다.

## 실행 순서

1. `build_lstm_data.ipynb`
2. `train_mlp.ipynb`
3. `train_lstm.ipynb`
4. `compare_models.ipynb`

## 실행 환경

필요 패키지는 `requirements.txt`에 정리되어 있습니다.

예시:

```bash
pip install -r requirements.txt
```

추가 참고:

- `MLP`는 `scikit-learn` 기반입니다.
- `LSTM`은 `tensorflow` 기반입니다.
- 노트북 실행 시 `tensorflow`가 설치된 커널을 사용하는 것이 안전합니다.
