"""
MC Dropout Time-Series Variance v3  (AUC-ROC only)

test set 전체에 sliding window MC Dropout variance 를 계산하고
데이터셋별 AUC-ROC 를 CSV 로 저장. 플롯은 생성하지 않음.

- variance_ts (full test-set sliding) → anomaly score
- labels[tr_count:]                   → ground truth
- NaN(lookback 미도달 구간) 마스킹 후 roc_auc_score 계산
- 결과를 {AGGREGATION_MODE}_auc_roc.csv 에 데이터셋별 1행씩 append
- 이미 CSV 에 기록된 데이터셋은 재계산 없이 skip
"""

import csv
import glob
import os
import re
import types

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model import iTransformer


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG – edit here
# ═════════════════════════════════════════════════════════════════════════════

RUN_MODE      = 'all_no_exclude'  # 'all' | 'all_no_exclude' | 'single'
SINGLE_CSV    = '001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv'
APPLY_EXCLUDE = RUN_MODE != 'all_no_exclude'  # all_no_exclude 이면 EXCLUDE_DATASETS 무시

_HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH    = os.path.join(_HERE, '../../dataset/TSB-AD-M/')
CHECKPOINTS  = os.path.join(_HERE, '../../checkpoints/')

LOOKBACK_WINDOW   = 192
PREDICTION_WINDOW = 96
LABEL_LEN         = 48
MC_SAMPLES        = 100

# ── Sliding window ────────────────────────────────────────────────────────────
SLIDE_STRIDE     = 1      # sliding step size (1 = fully overlapping)
AGGREGATION_MODE = 'mean'  # 'mean' or 'max' over accumulated variance values per time step
WINDOW_BATCH     = 20     # 한 번의 GPU forward pass 에서 처리할 window 수
                          # effective batch_size = WINDOW_BATCH × MC_SAMPLES

# ── Model architecture ────────────────────────────────────────────────────────
ENC_IN           = 7
DEC_IN           = 7
C_OUT            = 7
D_MODEL          = 512
N_HEADS          = 8
E_LAYERS         = 2
D_LAYERS         = 1
D_FF             = 2048
FACTOR           = 1
DROPOUT          = 0.1
EMBED            = 'timeF'
FREQ             = 'h'
ACTIVATION       = 'gelu'
USE_NORM         = 1
CLASS_STRATEGY   = 'projection'
OUTPUT_ATTENTION = False

USE_GPU = True
GPU     = 0

_LOOKBACK_MAP  = {192: '00_Lookback=192', 336: '01_Lookback=336', 720: '02_Lookback=720'}
_LOOKBACK_NAME = _LOOKBACK_MAP.get(LOOKBACK_WINDOW, f'Lookback={LOOKBACK_WINDOW}')

RESULT_DIR = os.path.join(
    _HERE,
    f'../01_MC_Dropout_results/03_timeseries_sliding/{_LOOKBACK_NAME}/{AGGREGATION_MODE}',
)
AUC_CSV = os.path.join(RESULT_DIR, 'channel_wise_auc_roc_select_max.csv')

# ── Exclude ───────────────────────────────────────────────────────────────────
EXCLUDE_DATASETS: set[str] = {
    # Genesis (digital_ratio 0.72)
    '001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv',
    # MSL (digital_ratio 0.98–1.0)
    '002_MSL_id_1_Sensor_tr_500_1st_900.csv',
    '003_MSL_id_2_Sensor_tr_883_1st_1238.csv',
    '004_MSL_id_3_Sensor_tr_530_1st_630.csv',
    '005_MSL_id_4_Sensor_tr_855_1st_2700.csv',
    '006_MSL_id_5_Sensor_tr_1150_1st_1250.csv',
    '007_MSL_id_6_Sensor_tr_980_1st_3550.csv',
    '008_MSL_id_7_Sensor_tr_656_1st_1630.csv',
    '009_MSL_id_8_Sensor_tr_714_1st_1390.csv',
    '010_MSL_id_9_Sensor_tr_554_1st_1172.csv',
    '011_MSL_id_10_Sensor_tr_1525_1st_4590.csv',
    '012_MSL_id_11_Sensor_tr_539_1st_940.csv',
    '013_MSL_id_12_Sensor_tr_554_1st_1200.csv',
    '014_MSL_id_13_Sensor_tr_1525_1st_4575.csv',
    '015_MSL_id_14_Sensor_tr_575_1st_1250.csv',
    '016_MSL_id_15_Sensor_tr_500_1st_780.csv',
    '017_MSL_id_16_Sensor_tr_512_1st_1850.csv',
    # SMAP (digital_ratio 0.96–1.0)
    '144_SMAP_id_1_Sensor_tr_2052_1st_5300.csv',
    '145_SMAP_id_2_Sensor_tr_2133_1st_5400.csv',
    '146_SMAP_id_3_Sensor_tr_2128_1st_5000.csv',
    '147_SMAP_id_4_Sensor_tr_2160_1st_6449.csv',
    '148_SMAP_id_5_Sensor_tr_2011_1st_5060.csv',
    '149_SMAP_id_6_Sensor_tr_2128_1st_5000.csv',
    '150_SMAP_id_7_Sensor_tr_2077_1st_5394.csv',
    '151_SMAP_id_8_Sensor_tr_1971_1st_4870.csv',
    '152_SMAP_id_9_Sensor_tr_2073_1st_5600.csv',
    '153_SMAP_id_10_Sensor_tr_1840_1st_4030.csv',
    '154_SMAP_id_11_Sensor_tr_2117_1st_4770.csv',
    '155_SMAP_id_12_Sensor_tr_1907_1st_4800.csv',
    '156_SMAP_id_13_Sensor_tr_1173_1st_2750.csv',
    '157_SMAP_id_14_Sensor_tr_2126_1st_5000.csv',
    '158_SMAP_id_15_Sensor_tr_2075_1st_5610.csv',
    '159_SMAP_id_16_Sensor_tr_1757_1st_2650.csv',
    '160_SMAP_id_17_Sensor_tr_1832_1st_5300.csv',
    '161_SMAP_id_18_Sensor_tr_2075_1st_5550.csv',
    '162_SMAP_id_19_Sensor_tr_1908_1st_4690.csv',
    '163_SMAP_id_20_Sensor_tr_2051_1st_4575.csv',
    '164_SMAP_id_21_Sensor_tr_1976_1st_4200.csv',
    '165_SMAP_id_22_Sensor_tr_2129_1st_5000.csv',
    '166_SMAP_id_23_Sensor_tr_1113_1st_1890.csv',
    '167_SMAP_id_24_Sensor_tr_2094_1st_5600.csv',
    '168_SMAP_id_25_Sensor_tr_1998_1st_2098.csv',
    '169_SMAP_id_26_Sensor_tr_1811_1st_4510.csv',
    '170_SMAP_id_27_Sensor_tr_2160_1st_4690.csv',
}

# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────── data helpers ─────────────────────────────────

def parse_tr_count(fname: str, n_total: int) -> int:
    m = re.search(r'_tr_(\d+)_', fname)
    return int(m.group(1)) if m else int(n_total * 0.6)


def get_anomaly_regions(labels: np.ndarray) -> list[tuple[int, int]]:
    regions, in_anom, start = [], False, 0
    for i, lbl in enumerate(labels):
        if lbl == 1 and not in_anom:
            start, in_anom = i, True
        elif lbl == 0 and in_anom:
            regions.append((start, i))
            in_anom = False
    if in_anom:
        regions.append((start, len(labels)))
    return regions


def load_csv(csv_path: str):
    """Return (scaled_data, labels, tr_count) or None."""
    df_raw    = pd.read_csv(csv_path)
    fname     = os.path.basename(csv_path)
    label_col = next((c for c in df_raw.columns if c.lower() == 'label'), None)
    if label_col is None:
        return None

    exclude     = {label_col, 'timestamp', 'Timestamp', 'datetime', 'date'}
    sensor_cols = [c for c in df_raw.columns if c not in exclude]

    labels   = df_raw[label_col].values.astype(int)
    raw_data = df_raw[sensor_cols].values.astype(np.float32)

    tr_count  = parse_tr_count(fname, len(df_raw))
    num_train = int(tr_count * 0.7)

    scaler = StandardScaler()
    scaler.fit(raw_data[:num_train])
    scaled = scaler.transform(raw_data)

    return scaled, labels, tr_count


# ─────────────────────────────── model helpers ────────────────────────────────

def enable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def find_checkpoint(csv_path: str, seq_len: int, pred_len: int) -> str | None:
    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    suffix   = f'_TSB-AD-M_seq{seq_len}_pred{pred_len}_'
    for name in os.listdir(CHECKPOINTS):
        if suffix not in name:
            continue
        if not os.path.isdir(os.path.join(CHECKPOINTS, name)):
            continue
        dataset_key = name[:name.index(suffix)]
        if csv_stem.startswith(dataset_key):
            return name
    return None


def build_model(device: torch.device, seq_len: int) -> nn.Module:
    args = types.SimpleNamespace(
        seq_len          = seq_len,
        pred_len         = PREDICTION_WINDOW,
        label_len        = LABEL_LEN,
        enc_in           = ENC_IN,
        dec_in           = DEC_IN,
        c_out            = C_OUT,
        d_model          = D_MODEL,
        n_heads          = N_HEADS,
        e_layers         = E_LAYERS,
        d_layers         = D_LAYERS,
        d_ff             = D_FF,
        factor           = FACTOR,
        dropout          = DROPOUT,
        embed            = EMBED,
        freq             = FREQ,
        activation       = ACTIVATION,
        use_norm         = USE_NORM,
        class_strategy   = CLASS_STRATEGY,
        output_attention = OUTPUT_ATTENTION,
        device           = device,
    )
    return iTransformer.Model(args).float().to(device)


def load_checkpoint(model: nn.Module, ckpt_name: str, device: torch.device) -> None:
    ckpt_path = os.path.join(CHECKPOINTS, ckpt_name, 'checkpoint.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))


def mc_dropout_predict(
    model:    nn.Module,
    x_inputs: np.ndarray,
    device:   torch.device,
) -> np.ndarray:
    """
    x_inputs : [W, lookback, n_features]
    returns  : [W, MC_SAMPLES, PREDICTION_WINDOW, n_features]
    """
    W, lookback, n_feat = x_inputs.shape
    total_batch = W * MC_SAMPLES

    x_enc = (torch.FloatTensor(x_inputs)
             .to(device)
             .unsqueeze(1)
             .expand(-1, MC_SAMPLES, -1, -1)
             .reshape(total_batch, lookback, n_feat))

    x_mark_enc = torch.zeros(total_batch, lookback, 4, device=device)
    dec_inp    = torch.zeros(total_batch, LABEL_LEN + PREDICTION_WINDOW, n_feat, device=device)
    x_mark_dec = torch.zeros(total_batch, LABEL_LEN + PREDICTION_WINDOW, 4, device=device)

    model.eval()
    enable_dropout(model)

    with torch.no_grad():
        out = model(x_enc, x_mark_enc, dec_inp, x_mark_dec)

    return out.cpu().numpy().reshape(W, MC_SAMPLES, PREDICTION_WINDOW, n_feat)


def var_per_channel_timestep(preds: np.ndarray) -> np.ndarray:
    """
    preds : [MC_SAMPLES, pred_len, n_channels]
    returns [pred_len, n_channels]  (variance over samples, per channel)
    """
    return preds.var(axis=0)  # [pred_len, n_channels]


# ──────────────── full test-set sliding window variance computation ───────────

def compute_sliding_variance(
    scaled:   np.ndarray,
    tr_count: int,
    model:    nn.Module,
    device:   torch.device,
    lookback: int,
) -> np.ndarray:
    """
    Returns variance_ts: [test_len] array, NaN where no window covered the step.
    """
    test_start = tr_count
    test_end   = len(scaled)
    test_len   = test_end - test_start

    n_ch = scaled.shape[1]
    if AGGREGATION_MODE == 'mean':
        accum_sum = np.zeros((test_len, n_ch), dtype=np.float64)
        accum_cnt = np.zeros(test_len, dtype=np.int64)
    else:
        accum_max = np.full((test_len, n_ch), -np.inf, dtype=np.float64)
        accum_hit = np.zeros(test_len, dtype=bool)

    positions = list(range(test_start, test_end - lookback - PREDICTION_WINDOW + 1, SLIDE_STRIDE))
    total     = len(positions)
    n_batches = (total + WINDOW_BATCH - 1) // WINDOW_BATCH
    log_every = max(1, n_batches // 20)

    for bi, batch_start in enumerate(range(0, total, WINDOW_BATCH)):
        batch_ts = positions[batch_start : batch_start + WINDOW_BATCH]

        if bi % log_every == 0:
            print(f"    sliding [{batch_start+1}/{total}]  t={batch_ts[0]}", flush=True)

        x_inputs = np.stack([scaled[t : t + lookback] for t in batch_ts])
        preds_w  = mc_dropout_predict(model, x_inputs, device)

        for j, t in enumerate(batch_ts):
            var_ch = var_per_channel_timestep(preds_w[j])  # [pred_len, n_channels]
            lo = (t + lookback) - test_start
            hi = lo + PREDICTION_WINDOW
            if AGGREGATION_MODE == 'mean':
                accum_sum[lo:hi] += var_ch
                accum_cnt[lo:hi] += 1
            else:
                np.maximum(accum_max[lo:hi], var_ch, out=accum_max[lo:hi])
                accum_hit[lo:hi] = True

    n_ch = scaled.shape[1]
    variance_ts = np.full((test_len, n_ch), np.nan, dtype=np.float64)
    if AGGREGATION_MODE == 'mean':
        valid = accum_cnt > 0          # [test_len]
        variance_ts[valid] = accum_sum[valid] / accum_cnt[valid, None]
    else:
        valid = accum_hit              # [test_len]
        variance_ts[valid] = accum_max[valid]

    return variance_ts


# ──────────────────────────────── AUC-ROC ─────────────────────────────────────

def auc_already_computed(name: str) -> bool:
    if not os.path.exists(AUC_CSV):
        return False
    df = pd.read_csv(AUC_CSV)
    return name in df['dataset'].values


def compute_auc(
    variance_ts: np.ndarray,
    labels:      np.ndarray,
    tr_count:    int,
) -> tuple[float, int, int, int]:
    """
    variance_ts : [test_len, n_channels]
    채널별로 AUC-ROC 를 계산하고 최댓값을 반환.

    Returns (auc_roc, n_valid, n_anomaly, best_channel).
    auc_roc is nan if no channel has both classes present.
    """
    test_labels = labels[tr_count : tr_count + variance_ts.shape[0]]
    # NaN 마스크는 채널 무관하게 동일 (어느 채널이든 커버 여부 동일)
    valid_mask  = ~np.isnan(variance_ts[:, 0])
    y_true      = test_labels[valid_mask]
    n_valid     = int(valid_mask.sum())
    n_anomaly   = int(y_true.sum())

    if n_anomaly == 0 or n_anomaly == n_valid:
        return float('nan'), n_valid, n_anomaly, -1

    best_auc = -np.inf
    best_ch  = -1
    for ch in range(variance_ts.shape[1]):
        scores = variance_ts[valid_mask, ch]
        auc    = float(roc_auc_score(y_true, scores))
        if auc > best_auc:
            best_auc = auc
            best_ch  = ch

    return best_auc, n_valid, n_anomaly, best_ch


def save_auc_result(name: str, auc: float, n_valid: int,
                    n_anomaly: int, best_channel: int) -> None:
    file_exists = os.path.exists(AUC_CSV)
    with open(AUC_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['dataset', 'auc_roc', 'n_valid', 'n_anomaly', 'best_channel'])
        auc_str = f'{auc:.6f}' if not np.isnan(auc) else 'nan'
        writer.writerow([name, auc_str, n_valid, n_anomaly, best_channel])


# ──────────────────────────────── main ────────────────────────────────────────

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    if USE_GPU and torch.cuda.is_available():
        device = torch.device(f'cuda:{GPU}')
        print(f'Use GPU: cuda:{GPU}')
    else:
        device = torch.device('cpu')
        print('Use CPU')

    model = build_model(device, LOOKBACK_WINDOW)

    if RUN_MODE == 'single':
        csv_files = [os.path.join(ROOT_PATH, SINGLE_CSV)]
    else:
        all_csv   = sorted(glob.glob(os.path.join(ROOT_PATH, '*.csv')))
        csv_files = [f for f in all_csv
                     if not APPLY_EXCLUDE or os.path.basename(f) not in EXCLUDE_DATASETS]

    if not csv_files:
        print(f"No CSV files found in {ROOT_PATH}")
        return

    print(f"\nMode: {RUN_MODE.upper()} | {len(csv_files)} candidate(s) | "
          f"lookback={LOOKBACK_WINDOW}, pred={PREDICTION_WINDOW}, "
          f"stride={SLIDE_STRIDE}, T={MC_SAMPLES}, agg={AGGREGATION_MODE}\n")

    for idx, csv_path in enumerate(csv_files):
        name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"[{idx+1:03d}/{len(csv_files)}] {name}")

        if auc_already_computed(name):
            print(f"  [SKIP] AUC already recorded.")
            continue

        result = load_csv(csv_path)
        if result is None:
            print(f"  [SKIP] No label column.")
            continue

        scaled, labels, tr_count = result

        if np.any(labels[:tr_count] == 1):
            print(f"  [SKIP] Anomaly found in train set.")
            continue

        all_regions  = get_anomaly_regions(labels)
        test_regions = [(s, e) for s, e in all_regions if s >= tr_count]
        if not test_regions:
            print(f"  [SKIP] No anomalies in test set.")
            continue

        ckpt_name = find_checkpoint(csv_path, LOOKBACK_WINDOW, PREDICTION_WINDOW)
        if ckpt_name is None:
            print(f"  [SKIP] No checkpoint found.")
            continue
        load_checkpoint(model, ckpt_name, device)

        variance_ts = compute_sliding_variance(
            scaled, tr_count, model, device, LOOKBACK_WINDOW
        )

        auc, n_valid, n_anomaly, best_ch = compute_auc(variance_ts, labels, tr_count)
        save_auc_result(name, auc, n_valid, n_anomaly, best_ch)
        print(f"  → AUC-ROC: {auc:.4f}  "
              f"(n_valid={n_valid}, n_anomaly={n_anomaly}, best_channel={best_ch})")

    print(f"\nDone. Results saved to: {AUC_CSV}")


if __name__ == '__main__':
    main()
