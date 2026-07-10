"""
Phase 2: Bivariate ELA–topography associations (cross-sectional)

Runs two parallel analyses:
  (A) tag='individual'  — 10 ELA factors × 15 networks (150 tests)
  (B) tag='composites'  — 3 a priori composites × 15 networks (45 tests)

For each predictor × network pair: MixedLM with family_id as random intercept
and site as fixed-effect dummies. Partial-r computed from t-statistic.
FDR (BH) applied within each tag.

Outputs per tag and timepoint:
  TAB_DIR/phase2_{tag}_r_matrix_{timepoint}.csv
  TAB_DIR/phase2_{tag}_p_matrix_{timepoint}.csv
  TAB_DIR/phase2_{tag}_q_matrix_{timepoint}.csv
  TAB_DIR/phase2_{tag}_significant_associations.csv
  FIG_DIR/fig_phase2_{tag}_heatmap_{timepoint}.png
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
from pathlib import Path

from adtopo.config import (
    FIG_DIR, TAB_DIR, DAT_DIR,
    NETWORKS, ELA_COLS, ELA_LABELS, ELA_LABELS_SHORT,
    COMPOSITE_COLS, COMPOSITE_LABELS, COMPOSITE_LABELS_SHORT,
    NET_GROUPS, NET_GROUP_COLOR,
    N_TESTS, N_TESTS_COMPOSITES, BONFERRONI_ALPHA, RANDOM_SEED,
)
from adtopo.re_models import fit_ols_cluster_table
from adtopo.logging_utils import get_logger
_log = get_logger('bivariate_associations')

np.random.seed(RANDOM_SEED)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

log_lines = []
def log(msg=''):
    _log.info(str(msg))
    log_lines.append(str(msg))

# ── Load data ─────────────────────────────────────────────────────────────────

df_base = pd.read_csv(DAT_DIR / 'df_base.csv')
df_y2   = pd.read_csv(DAT_DIR / 'df_y2.csv')
df_y4   = pd.read_csv(DAT_DIR / 'df_y4.csv')
df_y6   = pd.read_csv(DAT_DIR / 'df_y6.csv')
log(f'Loaded: df_base N={len(df_base)}, df_y2 N={len(df_y2)}, '
    f'df_y4 N={len(df_y4)}, df_y6 N={len(df_y6)}')


# ── Core association matrix ───────────────────────────────────────────────────

def assoc_matrix(df, pred_cols, networks, fd_col, site_col, family_col, label):
    """
    Partial-r (from the cluster-robust z-statistic) for each predictor × network
    pair, under the canonical reported specification: OLS with study site as
    fixed-effect dummies and family-cluster-robust standard errors
    (lib.re_models.fit_ols_cluster_table). One predictor is fitted at a time,
    matching the bivariate design. Returns r_mat, p_mat, ci_lo, ci_hi.
    """
    r_mat = pd.DataFrame(index=pred_cols, columns=networks, dtype=float)
    p_mat = pd.DataFrame(index=pred_cols, columns=networks, dtype=float)
    ci_lo = pd.DataFrame(index=pred_cols, columns=networks, dtype=float)
    ci_hi = pd.DataFrame(index=pred_cols, columns=networks, dtype=float)

    covars = ['interview_age', 'sex_num', fd_col]
    for pred in pred_cols:
        for net in networks:
            prop_col = f'prop_{net}'
            if prop_col not in df.columns or pred not in df.columns:
                continue
            tbl, meta = fit_ols_cluster_table(
                df, prop_col, [pred], covars,
                site_col=site_col, family_col=family_col)
            if not meta['converged'] or tbl.empty:
                continue
            row = tbl.iloc[0]
            r_val, p_val, n = float(row['partial_r']), float(row['p']), meta['n']
            # Fisher z-transform of the partial r, then a normal-approximation
            # 95% CI on the z scale that is back-transformed with tanh below.
            z = np.arctanh(np.clip(r_val, -0.9999, 0.9999))
            # SE of the Fisher z: 1/sqrt(n - 3). The "-3" is the bias/df
            # correction for a partial correlation (n - 3 - k with k=0 covariates
            # partialled here, since partial_r already conditions on covariates).
            se_z = 1.0 / np.sqrt(max(n - 3, 1))

            r_mat.loc[pred, net] = r_val
            p_mat.loc[pred, net] = p_val
            ci_lo.loc[pred, net] = np.tanh(z - 1.96 * se_z)
            ci_hi.loc[pred, net] = np.tanh(z + 1.96 * se_z)

    log(f'  [{label}] Association matrix computed ({len(pred_cols)} preds x {len(networks)} nets).')
    return r_mat, p_mat, ci_lo, ci_hi


# ── FDR correction ────────────────────────────────────────────────────────────

def apply_fdr(p_mat, n_tests, label):
    p_flat = p_mat.values.astype(float).flatten()
    valid  = ~np.isnan(p_flat)
    q_flat = np.full(len(p_flat), np.nan)
    if valid.sum():
        _, q_valid, _, _ = multipletests(p_flat[valid], alpha=0.05, method='fdr_bh')
        q_flat[valid] = q_valid
        bonf_alpha = 0.05 / n_tests
        log(f'  [{label}] FDR q<0.05: {(q_valid < 0.05).sum()}/{n_tests}  '
            f'Bonferroni (p<{bonf_alpha:.5f}): {(p_flat[valid] < bonf_alpha).sum()}/{n_tests}')
    q_mat = pd.DataFrame(q_flat.reshape(p_mat.shape),
                         index=p_mat.index, columns=p_mat.columns)
    return q_mat


# ── Heatmap ───────────────────────────────────────────────────────────────────

def draw_heatmap(r_mat, q_mat, p_mat, pred_labels, networks, title, fname, bonf_alpha):
    r_vals = r_mat.values.astype(float)
    q_vals = q_mat.values.astype(float)
    p_vals = p_mat.values.astype(float)

    n_preds = len(pred_labels)
    n_nets  = len(networks)
    fig_w = n_nets * 0.6 + 3
    fig_h = n_preds * 0.5 + 2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Annotation: only for FDR-sig cells
    annot = np.full(r_vals.shape, '', dtype=object)
    for i in range(r_vals.shape[0]):
        for j in range(r_vals.shape[1]):
            r, q, p = r_vals[i, j], q_vals[i, j], p_vals[i, j]
            if np.isnan(r) or np.isnan(q) or q >= 0.05:
                continue
            star = '*' if (not np.isnan(p) and p < bonf_alpha) else ''
            annot[i, j] = f'{r:.3f}{star}'

    sns.heatmap(
        r_vals, annot=annot, fmt='',
        cmap='RdBu_r', center=0, vmin=-0.15, vmax=0.15,
        xticklabels=networks, yticklabels=pred_labels,
        linewidths=0.3, linecolor='#cccccc', ax=ax,
        annot_kws={'size': 7},
        cbar_kws={'label': 'Partial r', 'shrink': 0.7},
    )

    # Bold border around FDR-sig cells
    for i in range(q_vals.shape[0]):
        for j in range(q_vals.shape[1]):
            if not np.isnan(q_vals[i, j]) and q_vals[i, j] < 0.05:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False, edgecolor='black', lw=1.5,
                ))

    ax.set_facecolor('white')
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel('Network', fontsize=10)
    ax.set_ylabel('Predictor', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    fig.text(
        0.5, -0.01,
        '* also Bonferroni significant   bold box = FDR q<0.05   annotated = FDR sig',
        ha='center', fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


# ── Significant-associations table ───────────────────────────────────────────

def collect_sig(r_mat, p_mat, q_mat, ci_lo, ci_hi, timepoint, bonf_alpha):
    rows = []
    for pred in r_mat.index:
        for net in r_mat.columns:
            r = float(r_mat.loc[pred, net]) if not pd.isnull(r_mat.loc[pred, net]) else np.nan
            p = float(p_mat.loc[pred, net]) if not pd.isnull(p_mat.loc[pred, net]) else np.nan
            q = float(q_mat.loc[pred, net]) if not pd.isnull(q_mat.loc[pred, net]) else np.nan
            if np.isnan(q) or q >= 0.05:
                continue
            rows.append({
                'timepoint':      timepoint,
                'predictor':      pred,
                'network':        net,
                'partial_r':      round(r, 4),
                'CI95_lower':     round(float(ci_lo.loc[pred, net]), 4),
                'CI95_upper':     round(float(ci_hi.loc[pred, net]), 4),
                'p_uncorrected':  round(p, 6),
                'q_FDR':          round(q, 6),
                'bonferroni_sig': (not np.isnan(p)) and (p < bonf_alpha),
            })
    return pd.DataFrame(rows)


# ── Network grouping summary ──────────────────────────────────────────────────

def net_group_summary(r_mat, label):
    log(f'\n  {label}:')
    for grp, nets in NET_GROUPS.items():
        avail = [n for n in nets if n in r_mat.columns]
        if avail:
            mean_abs_r = np.nanmean(np.abs(r_mat[avail].values.astype(float)))
            log(f'    {grp:20s}: mean |r| = {mean_abs_r:.4f}')


# ── Run one analysis tag ──────────────────────────────────────────────────────

TIMEPOINTS = [
    ('baseline', df_base, '00A'),
    ('year2',    df_y2,   '02A'),
    ('year4',    df_y4,   '04A'),
    ('year6',    df_y6,   '06A'),
]

ANALYSES = [
    {
        'tag':               'individual',
        'pred_cols':         ELA_COLS,
        'pred_labels':       ELA_LABELS,
        'pred_labels_short': [ELA_LABELS_SHORT[e] for e in ELA_COLS],
        'n_tests':           N_TESTS,
        'bonf_alpha':        BONFERRONI_ALPHA,
    },
    {
        'tag':               'composites',
        'pred_cols':         COMPOSITE_COLS,
        'pred_labels':       [COMPOSITE_LABELS[c] for c in COMPOSITE_COLS],
        'pred_labels_short': [COMPOSITE_LABELS_SHORT[c] for c in COMPOSITE_COLS],
        'n_tests':           N_TESTS_COMPOSITES,
        'bonf_alpha':        0.05 / N_TESTS_COMPOSITES,
    },
]

for analysis in ANALYSES:
    tag          = analysis['tag']
    pred_cols    = analysis['pred_cols']
    pred_labels  = analysis['pred_labels']
    pred_labels_s = analysis['pred_labels_short']
    n_tests      = analysis['n_tests']
    bonf_alpha   = analysis['bonf_alpha']

    log()
    log('=' * 70)
    log(f'ANALYSIS: {tag.upper()}  ({len(pred_cols)} predictors x {len(NETWORKS)} networks)')
    log('=' * 70)

    all_sig_dfs = []

    for tp_label, df_tp, tp_code in TIMEPOINTS:
        # Filter to predictors that exist in this df
        avail_preds = [p for p in pred_cols if p in df_tp.columns]
        if not avail_preds:
            log(f'  [{tp_label}] WARNING: none of pred_cols found in dataframe. Skipping.')
            continue

        log(f'\n  --- {tp_label} ---')

        # Phase 1 writes generic 'fd' and 'study_site'
        fd_col   = 'fd'   if 'fd'   in df_tp.columns else 'rest_mean_FD'
        site_col = 'study_site' if 'study_site' in df_tp.columns else 'study_site_baseline'

        r_mat, p_mat, ci_lo, ci_hi = assoc_matrix(
            df_tp, avail_preds, NETWORKS,
            fd_col=fd_col, site_col=site_col,
            family_col='family_id', label=tp_label,
        )

        q_mat = apply_fdr(p_mat, n_tests, tp_label)

        # Save matrices
        r_mat.round(4).to_csv(TAB_DIR / f'phase2_{tag}_r_matrix_{tp_label}.csv')
        p_mat.round(6).to_csv(TAB_DIR / f'phase2_{tag}_p_matrix_{tp_label}.csv')
        q_mat.round(6).to_csv(TAB_DIR / f'phase2_{tag}_q_matrix_{tp_label}.csv')

        # Heatmap
        tp_title_map = {
            'baseline': 'Baseline (~9-10y)',
            'year2':    'Year-2 (~11-12y)',
            'year4':    'Year-4 (~13-14y)',
            'year6':    'Year-6 (~15-16y)',
        }
        draw_heatmap(
            r_mat, q_mat, p_mat,
            pred_labels_s, NETWORKS,
            title=f'ELA [{tag}] × Network Proportion — Partial r\n({tp_title_map[tp_label]})',
            fname=FIG_DIR / f'fig_phase2_{tag}_heatmap_{tp_label}.png',
            bonf_alpha=bonf_alpha,
        )

        # Significant associations
        sig_df = collect_sig(r_mat, p_mat, q_mat, ci_lo, ci_hi, tp_label, bonf_alpha)
        all_sig_dfs.append(sig_df)
        log(f'  [{tp_label}] Significant (FDR q<0.05): {len(sig_df)}')
        if len(sig_df):
            log(sig_df.sort_values('partial_r', key=abs, ascending=False).to_string(index=False))

        # Network group summary
        net_group_summary(r_mat, tp_label)

    # Combine and save significant associations
    if all_sig_dfs:
        sig_all = pd.concat(all_sig_dfs, ignore_index=True)
    else:
        sig_all = pd.DataFrame()
    sig_all.to_csv(TAB_DIR / f'phase2_{tag}_significant_associations.csv', index=False)
    log(f'\n  Saved phase2_{tag}_significant_associations.csv '
        f'(total FDR-sig rows: {len(sig_all)})')

log()
log('=' * 70)

with open(DAT_DIR / 'progress_log.txt', 'a') as f:
    f.write('\n\nPHASE 2 COMPLETE\n')
    f.write('\n'.join(log_lines[-40:]))

log('Phase 2 complete.')
