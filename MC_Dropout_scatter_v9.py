"""
MC Dropout Scatter Plot v9
  X축: Normal-only MAE or MSE on test set
       (pred.npy / true.npy에서 lookback + prediction 모두 label=0인 window만)
       window i 절대 위치 (CSV 기준):
         lookback  : CSV[tr_count - seq_len + i,  tr_count + i)
         prediction: CSV[tr_count + i,            tr_count + i + pred_len)
       두 구간 모두 label=0인 window만 포함.
       error = mean MAE or MSE over all valid windows
  Y축: AUC-ROC per dataset
       MC Dropout sliding variance → anomaly score → roc_auc_score vs. labels
       (variance_v3.py 에서 생성된 {AGGREGATION_MODE}_auc_roc.csv 사용)
"""

import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

# ─── IEEE style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'       : 'serif',
    'font.serif'        : ['Times New Roman', 'DejaVu Serif'],
    'font.size'         : 8,
    'axes.labelsize'    : 9,
    'axes.titlesize'    : 9,
    'xtick.labelsize'   : 7,
    'ytick.labelsize'   : 7,
    'legend.fontsize'   : 7,
    'axes.linewidth'    : 0.8,
    'xtick.major.width' : 0.8,
    'ytick.major.width' : 0.8,
    'xtick.direction'   : 'in',
    'ytick.direction'   : 'in',
    'lines.linewidth'   : 1.2,
    'grid.linewidth'    : 0.4,
    'grid.alpha'        : 0.4,
    'figure.dpi'        : 300,
})

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG – edit here
# ═════════════════════════════════════════════════════════════════════════════

# x축 지표 선택: 'MAE' or 'MSE'
ERROR_METRIC        = 'MSE'
# Lookback window 선택: 192, 336, or 720
SEQ_LEN             = 192
PRED_LEN            = 96
# variance_v3 에서 사용한 집계 모드: 'mean' or 'max'
AGGREGATION_MODE    = 'mean'

# X축 상한 클립 (이 값 초과인 데이터셋은 별도 figure에서 제외, None이면 클립 없음)
X_UPPER_CLIP        = 10
# AUC-ROC 최솟값 필터 (nan 제외 외 추가 필터 없으면 0.0)
MIN_AUC             = 0.0

# 제외할 데이터셋 family (dataset_family() 반환값 기준)
EXCLUDE_FAMILIES: set[str] = {'MSL', 'SMAP', 'TAO'}

# 저장 포맷: 'png' | 'svg' | 'pdf'
FMT                 = 'png'

SAVE_DIR            = None

# ═════════════════════════════════════════════════════════════════════════════

_HERE             = os.path.dirname(os.path.abspath(__file__))

_LOOKBACK_MAP     = {192: '00_Lookback=192', 336: '01_Lookback=336', 720: '02_Lookback=720'}
_LOOKBACK_NAME    = _LOOKBACK_MAP.get(SEQ_LEN, f'Lookback={SEQ_LEN}')

AUC_CSV           = os.path.join(
    _HERE,
    f'../01_MC_Dropout_results/03_timeseries_sliding/{_LOOKBACK_NAME}/{AGGREGATION_MODE}',
    f'{AGGREGATION_MODE}_auc_roc.csv',
)

_SCATTER_BASE     = os.path.join(_HERE, f'../01_MC_Dropout_results/01_scatter/08_v9')
_SCATTER_SAVE_DIR = os.path.join(_SCATTER_BASE, _LOOKBACK_NAME, AGGREGATION_MODE)

DATASET_DIR       = os.path.join(_HERE, '../../dataset/TSB-AD-M')
RESULTS_NPY_DIR   = os.path.join(_HERE, '../../results')   # pred.npy / true.npy


# ─────────────────────────────── helpers ─────────────────────────────────────

def dataset_family(name: str) -> str:
    parts = name.split('_')
    return parts[1] if len(parts) > 1 else name


def short_label(name: str) -> str:
    parts = name.split('_')
    if len(parts) >= 4:
        return f"{parts[1]}_{parts[2]}_{parts[3]}"
    return name


def build_color_map(families: list) -> dict:
    unique = sorted(set(families))
    cmap   = plt.get_cmap('Set1', len(unique))
    return {fam: cmap(i) for i, fam in enumerate(unique)}


def find_npy_dir(name: str) -> str | None:
    suffix = f'_TSB-AD-M_seq{SEQ_LEN}_pred{PRED_LEN}_0'
    candidates = [
        d for d in os.listdir(RESULTS_NPY_DIR)
        if d.endswith(suffix)
        and os.path.isdir(os.path.join(RESULTS_NPY_DIR, d))
    ]
    for ckpt in candidates:
        dataset_key = ckpt[:ckpt.index(suffix)]
        if name.startswith(dataset_key):
            return os.path.join(RESULTS_NPY_DIR, ckpt)
    return None


# ──────────────────────── X축: Normal-only error ─────────────────────────────

def compute_normal_only_error(name: str, metric: str) -> tuple[float | None, int]:
    """
    pred.npy / true.npy + 원본 CSV label을 이용해
    lookback + prediction 모두 anomaly-free인 window만으로 MAE/MSE 계산.

    window i 의 절대 위치 (CSV 기준):
      lookback  : [tr_count - seq_len + i,  tr_count + i)
      prediction: [tr_count + i,            tr_count + i + pred_len)
    """
    dataset_csv = os.path.join(DATASET_DIR, f'{name}.csv')
    if not os.path.exists(dataset_csv):
        print(f"  [SKIP] Dataset CSV not found: {name}.csv")
        return None, 0

    df_raw    = pd.read_csv(dataset_csv)
    label_col = next((c for c in df_raw.columns if c.lower() == 'label'), None)
    if label_col is None:
        print(f"  [SKIP] No label column: {name}")
        return None, 0

    labels   = df_raw[label_col].values.astype(int)
    m        = re.search(r'_tr_(\d+)_', name)
    tr_count = int(m.group(1)) if m else int(len(df_raw) * 0.6)

    npy_dir = find_npy_dir(name)
    if npy_dir is None:
        print(f"  [SKIP] No results dir for: {name}")
        return None, 0

    pred_path = os.path.join(npy_dir, 'pred.npy')
    true_path = os.path.join(npy_dir, 'true.npy')
    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print(f"  [SKIP] pred/true.npy missing: {npy_dir}")
        return None, 0

    pred      = np.load(pred_path)   # [n_windows, pred_len, n_features]
    true      = np.load(true_path)
    n_windows = pred.shape[0]

    valid_idx = []
    for i in range(n_windows):
        lb_start   = tr_count - SEQ_LEN + i
        lb_end     = tr_count + i
        pred_start = tr_count + i
        pred_end   = tr_count + i + PRED_LEN

        if lb_start < 0 or pred_end > len(labels):
            continue
        if np.any(labels[lb_start : lb_end]    == 1):
            continue
        if np.any(labels[pred_start : pred_end] == 1):
            continue
        valid_idx.append(i)

    n_valid = len(valid_idx)
    print(f"    normal-only windows: {n_valid} / {n_windows}")

    vp = pred[valid_idx]
    vt = true[valid_idx]

    # per-channel error: mean over (windows, pred_len) → [n_features]
    if metric == 'MAE':
        per_ch = np.mean(np.abs(vp - vt), axis=(0, 1))
    else:
        per_ch = np.mean((vp - vt) ** 2, axis=(0, 1))

    error_mean   = float(np.mean(per_ch))
    error_median = float(np.median(per_ch))
    return error_mean, error_median, n_valid


# ─────────────────────────── data collection ─────────────────────────────────

def collect_data(metric: str) -> list:
    if not os.path.exists(AUC_CSV):
        print(f"[ERROR] AUC CSV not found: {AUC_CSV}")
        return []

    auc_df = pd.read_csv(AUC_CSV)
    records = []

    for _, row in auc_df.iterrows():
        name    = row['dataset']
        auc_val = row['auc_roc']
        family  = dataset_family(name)

        if family in EXCLUDE_FAMILIES:
            print(f"  [SKIP] Excluded family ({family}): {name}")
            continue
        if pd.isna(auc_val):
            print(f"  [SKIP] AUC is nan: {name}")
            continue
        if float(auc_val) < MIN_AUC:
            print(f"  [SKIP] AUC={auc_val:.4f} < {MIN_AUC}: {name}")
            continue

        print(f"  {name}")
        result = compute_normal_only_error(name, metric)
        if result[0] is None:
            continue
        error_mean, error_median, n_valid = result

        records.append({
            'name'         : name,
            'family'       : dataset_family(name),
            'error_mean'   : error_mean,
            'error_median' : error_median,
            'n_normal_win' : n_valid,
            'auc_roc'      : float(auc_val),
            'n_anomaly'    : int(row['n_anomaly']),
        })
        print(f"    {metric}_mean={error_mean:.6f}  {metric}_median={error_median:.6f}  "
              f"AUC-ROC={auc_val:.4f}  n_anomaly={row['n_anomaly']}")

    return records


# ─────────────────────────────── plot ────────────────────────────────────────

def plot_scatter(records: list, save_dir: str, fmt: str, metric: str,
                 x_agg: str = 'mean', suffix: str = '') -> None:
    """x_agg: 'mean' or 'median' — 채널별 error의 집계 방식."""
    if not records:
        print("No data to plot.")
        return

    x_key = f'error_{x_agg}'
    records_sorted  = sorted(records, key=lambda r: r[x_key])
    xs_raw          = np.array([r[x_key] for r in records_sorted])
    ys_raw          = np.array([r['auc_roc']   for r in records_sorted])
    families_sorted = [r['family'] for r in records_sorted]

    n       = len(records_sorted)
    x_ranks = np.arange(1, n + 1)
    y_ranks = stats.rankdata(ys_raw)

    color_map = build_color_map(families_sorted)

    pearson_r,  pearson_p  = stats.pearsonr(xs_raw, ys_raw)
    spearman_r, spearman_p = stats.spearmanr(xs_raw, ys_raw)
    print(f"\n  Pearson  r={pearson_r:.4f}  p={pearson_p:.4f}")
    print(f"  Spearman ρ={spearman_r:.4f}  p={spearman_p:.4f}")

    x_label    = f'{metric} (Normal-only, Test set, ch-{x_agg})'
    y_label    = 'AUC-ROC\n(MC Dropout variance)'
    title_base = f'MC Dropout AUC-ROC vs. Normal-only {metric} [ch-{x_agg}]'

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.subplots_adjust(right=0.83, wspace=0.38)

    # ── Panel 1: Rank 균등배치 ────────────────────────────────────────────────
    ax = axes[0]
    slope1, intercept1, *_ = stats.linregress(x_ranks, y_ranks)
    x_fit1 = np.linspace(1, n, 300)
    y_fit1 = slope1 * x_fit1 + intercept1

    tick_count = min(11, n)
    tick_pos   = np.linspace(1, n, num=tick_count)
    tick_idx   = np.clip(np.rint(tick_pos - 1).astype(int), 0, n - 1)

    for xr, yr, fam in zip(x_ranks, y_ranks, families_sorted):
        ax.scatter(xr, yr, s=28, color=color_map[fam], zorder=3,
                   edgecolors='white', linewidths=0.4)
    ax.plot(x_fit1, y_fit1, color='red', linewidth=1.4, zorder=2, linestyle='-',
            label=(f'Linear fit (Rank)\n'
                   f'Pearson $r$={pearson_r:.3f} ($p$={pearson_p:.3f})\n'
                   f'Spearman $\\rho$={spearman_r:.3f} ($p$={spearman_p:.3f})'))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f'{v:.4f}' for v in xs_raw[tick_idx]],
                       rotation=45, ha='right', fontsize=6)
    ax.set_xlim(0, n + 1)
    ys_sorted = np.sort(ys_raw)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([f'{v:.4f}' for v in ys_sorted[tick_idx]], fontsize=6)
    ax.set_ylim(0, n + 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'(a) Rank\n{title_base}')
    ax.grid(True, axis='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, edgecolor='black',
              fancybox=False, handlelength=1.6)

    # ── Panel 2: 원본 raw 값 ──────────────────────────────────────────────────
    ax = axes[1]
    slope2, intercept2, *_ = stats.linregress(xs_raw, ys_raw)
    x_fit2 = np.linspace(xs_raw.min(), xs_raw.max(), 300)
    y_fit2 = slope2 * x_fit2 + intercept2

    for x, y, fam in zip(xs_raw, ys_raw, families_sorted):
        ax.scatter(x, y, s=28, color=color_map[fam], zorder=3,
                   edgecolors='white', linewidths=0.4)
    ax.plot(x_fit2, y_fit2, color='red', linewidth=1.4, zorder=2, linestyle='-',
            label=(f'Linear fit\n'
                   f'Pearson $r$={pearson_r:.3f} ($p$={pearson_p:.3f})\n'
                   f'Spearman $\\rho$={spearman_r:.3f} ($p$={spearman_p:.3f})'))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'(b) Raw\n{title_base}')
    ax.grid(True, axis='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, edgecolor='black',
              fancybox=False, handlelength=1.6)
    ax.tick_params(axis='x', rotation=45)

    # ── 데이터셋 legend (figure 오른쪽 밖) ───────────────────────────────────
    sorted_for_legend = sorted(records_sorted, key=lambda r: (r['family'], r['name']))
    dataset_handles   = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map[r['family']],
               markersize=5, label=short_label(r['name']))
        for r in sorted_for_legend
    ]
    fig.legend(
        handles=dataset_handles,
        title='Dataset',
        title_fontsize=7,
        fontsize=6,
        loc='center left',
        bbox_to_anchor=(0.84, 0.5),
        frameon=True,
        edgecolor='black',
        fancybox=False,
        ncol=2,
    )

    fname     = f'mc_dropout_scatter_v9_{AGGREGATION_MODE}_{metric}_ch{x_agg}{suffix}.{fmt}'
    save_path = os.path.join(save_dir, fname)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\n  → saved: {save_path}")


# ──────────────────────── per-family subplot figure ──────────────────────────

def plot_family_scatter(records: list, save_dir: str, fmt: str, metric: str,
                        x_agg: str = 'mean', suffix: str = '',
                        min_points: int = 5) -> None:
    """
    family별로 subplot을 하나씩 그려 각자의 회귀선을 표시.
    min_points 미만인 family는 제외.
    """
    if not records:
        print("No data to plot.")
        return

    x_key = f'error_{x_agg}'
    x_label = f'{metric} (Normal-only, Test set, ch-{x_agg})'
    y_label = 'AUC-ROC\n(MC Dropout variance)'

    # family별 grouping
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r['family']].append(r)

    valid_fams = sorted(fam for fam, rs in groups.items() if len(rs) >= min_points)
    n_fams     = len(valid_fams)

    if n_fams == 0:
        print("  [SKIP] No family with enough points.")
        return

    # 색상은 전체 데이터 기준으로 통일
    all_families  = [r['family'] for r in records]
    color_map     = build_color_map(all_families)

    n_cols = min(4, n_fams)
    n_rows = (n_fams + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.38, hspace=0.55)

    for idx, fam in enumerate(valid_fams):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        rs  = groups[fam]
        fx  = np.array([r[x_key]    for r in rs])
        fy  = np.array([r['auc_roc'] for r in rs])
        color = color_map[fam]

        ax.scatter(fx, fy, s=32, color=color, zorder=3,
                   edgecolors='white', linewidths=0.4)

        if len(fx) >= 2:
            slope, intercept, r_val, p_val, _ = stats.linregress(fx, fy)
            x_fit = np.linspace(fx.min(), fx.max(), 200)
            ax.plot(x_fit, slope * x_fit + intercept,
                    color='red', linewidth=1.2, zorder=2,
                    label=f'r={r_val:.3f}  p={p_val:.3f}')
            ax.legend(loc='upper left', frameon=True, edgecolor='black',
                      fancybox=False, fontsize=6, handlelength=1.4)

        ax.set_title(f'{fam}  (n={len(rs)})', fontsize=8)
        ax.set_xlabel(x_label, fontsize=7)
        ax.set_ylabel(y_label, fontsize=7)
        ax.grid(True, axis='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', rotation=45, labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

    # 빈 subplot 숨기기
    for idx in range(n_fams, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f'Per-family: MC Dropout AUC-ROC vs. Normal-only {metric} [ch-{x_agg}]  '
        f'(min_points={min_points})',
        fontsize=9, y=1.01,
    )

    fname     = (f'mc_dropout_scatter_v9_{AGGREGATION_MODE}_{metric}'
                 f'_ch{x_agg}_perfamily{suffix}.{fmt}')
    save_path = os.path.join(save_dir, fname)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  → saved: {save_path}")


# ──────────────────────────────── main ───────────────────────────────────────

def main():
    save_dir = SAVE_DIR or _SCATTER_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    print(f"X: Normal-only {ERROR_METRIC}  |  Y: AUC-ROC ({AGGREGATION_MODE})\n")
    print(f"AUC CSV: {AUC_CSV}\n")

    records = collect_data(ERROR_METRIC)
    print(f"\n{len(records)} dataset(s) included.")

    for x_agg in ('mean',):
        x_key = f'error_{x_agg}'
        plot_scatter(records, save_dir, FMT, ERROR_METRIC, x_agg=x_agg)
        plot_family_scatter(records, save_dir, FMT, ERROR_METRIC, x_agg=x_agg)

        if X_UPPER_CLIP is not None:
            clipped   = [r for r in records if r[x_key] <= X_UPPER_CLIP]
            n_removed = len(records) - len(clipped)
            print(f"\n[Clip/{x_agg}] X > {X_UPPER_CLIP} 제외: {n_removed}개 → {len(clipped)}개 남음")
            plot_scatter(clipped, save_dir, FMT, ERROR_METRIC,
                         x_agg=x_agg, suffix=f'_clip{X_UPPER_CLIP}')
            plot_family_scatter(clipped, save_dir, FMT, ERROR_METRIC,
                                x_agg=x_agg, suffix=f'_clip{X_UPPER_CLIP}')

    print("Done.")


if __name__ == '__main__':
    main()
