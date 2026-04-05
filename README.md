# High-Frequency Customer Repurchase Delay Prediction

고빈도 고객의 재구매 지연 위험을 `MLP`와 `LSTM`으로 비교하는 프로젝트입니다.

## 주제

- 핵심 고객: 주문 횟수가 많은 고빈도 고객
- 위험 정의: `days_since_prior_order > 15`
- 목표: 재구매 지연 위험을 조기에 탐지

## 데이터

- `data/orders.csv`
- `data/customer_features.csv`

## 실행 순서

1. `build_lstm_data.ipynb`
2. `train_mlp.ipynb`
3. `train_lstm.ipynb`
4. `compare_models.ipynb`

## 노트북 설명

- `build_lstm_data.ipynb`
  - `orders.csv`를 시퀀스 데이터로 변환
  - `outputs/X_seq.npy`, `outputs/y.npy`, `outputs/lstm_sequence_metadata.csv` 생성

- `train_mlp.ipynb`
  - 집계 피처 기반 `MLP` 학습
  - 결과를 셀 안에서 바로 시각화

- `train_lstm.ipynb`
  - 시퀀스 기반 `LSTM` 학습
  - 결과를 셀 안에서 바로 시각화

- `compare_models.ipynb`
  - 같은 테스트 샘플 기준으로 `MLP`와 `LSTM` 성능 비교
  - 막대 그래프와 혼동행렬을 바로 표시

## 비교 기준

이 과제는 위험 고객 탐지가 중요하므로 아래 지표를 우선합니다.

- `Recall`
- `F1-score`
- `ROC-AUC`

## 참고

- `MLP`는 `scikit-learn` 기반으로 구성했습니다.
- `LSTM`은 `tensorflow`가 필요합니다.
- 노트북에서는 `png` 저장 대신 결과를 바로 화면에 표시합니다.
