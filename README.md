# iTransformer-MC-Dropout
- iTransformer + MC Dropout + TSB-AD 조합에 대해 실험한 Repository입니다.

- 전체적인 간략한 흐름은 다음과 같습니다.
```
MC_Dropout_time-series_variance_v3.py에서 y축에 해당하는 AUC 값을 저장하는 csv파일을 만들고, MC_Dropout_scatter_v9.py가 csv를 받아와 y축을 그리고, x축은 내부에서 normal MAE, MSE를 구한다.
MC_Dropout_time-series_variance_v3.py → channel_wise_auc_roc_select_max.csv (y축) → MC_Dropout_scatter_v9.py → normal MAE, MSE (x축)
```


# MC_Dropout_time-series_variance_v3.py

- 전체 흐름도
```
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
    │     │ 4. 100개 예측의 채널별 분산 계산                     │
    │     │ 5. 최대 pred len 만큼 쌓인 각 시점의 분산값 누적      │
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
```


# MC_Dropout_scatter_v9.py

- 전체 흐름도
```
AUC-ROC CSV 로드 (variance_v3.py 산출물)
    │
    ▼
데이터셋별 순회
    │   각 데이터셋에서:
    │     ┌──────────────────────────────────────────────────────┐
    │     │ 1. 제외 family 필터링 (MSL, SMAP, TAO 등)            │
    │     │ 2. AUC-ROC 유효성 검사 (nan, 최솟값 필터)             │
    │     │ 3. pred.npy / true.npy 로드                          │
    │     │ 4. 원본 CSV 라벨로 정상 윈도우만 필터링                 │
    │     │    - lookback 구간 전체 label=0                       │
    │     │    - prediction 구간 전체 label=0                     │
    │     │ 5. 정상 윈도우만으로 채널별 MAE 또는 MSE 계산           │
    │     └──────────────────────────────────────────────────────┘
    │
    ▼
records 리스트 수집 완료
    │
    ├───────────────────────────┐
    ▼                           ▼
전체 Scatter Plot            Family별 Scatter Plot
 (Rank + Raw 2패널)           (family별 subplot + 회귀선)
    │                           │
    ├──── X_UPPER_CLIP 적용 ─────┤
    │     (outlier 제외 버전)     │
    ▼                           ▼
PNG / SVG / PDF 저장         PNG / SVG / PDF 저장
```
- 이때 X_UPPER_CLIP = 10으로써, outlier들을 때문에 나머지 데이터셋들의 분포 정도를 확인하기 어려워 이들을 제거했을 때 나머지 데이터셋들이 어떤 분포 형태가 띄는지 확인하기 위해서 추가한 로직입니다.
- MSL, SMAP 데이터셋은 Digital 성격의 채널 보유 비율이 너무 높고, TAO의 경우에는 너무 난잡하게 Anomaly 구간이 지정되어 있어서 올바른 결과를 얻기 힘들다 생각하여 제외하였습니다.
