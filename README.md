# iTransformer-MC-Dropout
iTransformer + MC Dropout + TSB-AD 조합에 대해 실험한 Repository입니다.

#MC_Dropout_time-series_variance_v3.py 전체 흐름도
CSV 데이터 로드
    │
    ▼
전처리 (StandardScaler로 정규화)
    │
    ▼
학습된 iTransformer 체크포인트 로드
    │
    ▼
테스트 구간을 슬라이딩 윈도우로 순회
    │   각 윈도우 위치에서:
    │     ┌─────────────────────────────────────────────────────┐
    │     │ 1. lookback 길이의 입력 구성                         │
    │     │ 2. 같은 입력을 MC_SAMPLES(100)번 복제                │
    │     │ 3. Dropout ON 상태로 한번에 forward pass             │
    │     │ 4. 100개 예측의 채널별 분산 계산                      │
    │     │ 5. 최대 pred len 만큼 쌓인 각 시점의 분산값 누적       │
    │     └─────────────────────────────────────────────────────┘
    │
    ▼
누적된 분산값을 mean 또는 max로 집계 → variance_ts (시점별 이상 점수)
    │
    ▼
채널별로 AUC-ROC 계산 → 최고 AUC를 가진 채널 선택
    │
    ▼
결과를 CSV에 append
