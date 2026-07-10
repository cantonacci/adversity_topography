"""
Phase 3: Multivariate mixed-effects models — all four timepoints.

Runs two parallel analyses:
  (A) tag='individual' — 10 ELA factors simultaneously as predictors
  (B) tag='composites' — 3 a priori composites simultaneously as predictors

Model per network (crossed random intercepts; OLS fallback):
  prop_net ~ [predictors] + interview_age + sex_num + fd
             + (1|family_id) + (1|study_site)

Delta-R² via pseudo-R² (OLS; full - covariates-only).

Outputs per tag and timepoint:
  TAB_DIR/phase3_{tag}_results_{timepoint}.csv
  TAB_DIR/phase3_{tag}_delta_r2_{timepoint}.csv
  TAB_DIR/phase3_{tag}_brain_surface_inputs.csv
  FIG_DIR/fig_phase3_{tag}_beta_heatmap_{timepoint}.png
"""
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pathlib import Path

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import (
    FIG_DIR, TAB_DIR, DAT_DIR,
    NETWORKS, ELA_COLS, ELA_LABELS_SHORT,
    COMPOSITE_COLS, COMPOSITE_LABELS_SHORT,
    N_TESTS, N_TESTS_COMPOSITES, BONFERRONI_ALPHA, RANDOM_SEED,
)
from lib.re_models import fit_ols_cluster_table

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
    print(msg, flush=True)
    log_lines.append(str(msg))

# ── Load data ─────────────────────────────────────────────────────────────────

df_base = pd.read_csv(DAT_DIR / 'df_base.csv')
df_y2   = pd.read_csv(DAT_DIR / 'df_y2.csv')
df_y4   = pd.read_csv(DAT_DIR / 'df_y4.csv')
df_y6   = pd.read_csv(DAT_DIR / 'df_y6.csv')
log(f'Loaded: df_base N={len(df_base)}, df_y2 N={len(df_y2)}, '
    f'df_y4 N={len(df_y4)}, df_y6 N={len(df_y6)}')

TIMEPOINTS = [
    ('baseline', df_base),
    ('year2',    df_y2),
    ('year4',    df_y4),
    ('year6',    df_y6),
]


# ── Model fitting helper ──────────────────────────────────────────────────────

def fit_model(df, outcome_col, pred_cols, family_col='family_id'):
    """
    Canonical reported specification (lib.re_models.fit_ols_cluster_table):
        outcome ~ pred_cols + interview_age + sex_num + fd + C(study_site)
    fitted by OLS with study-site fixed effects and family-cluster-robust SEs.
    All predictors are entered simultaneously. Returns (results_df, method,
    r2, r2) with columns [predictor, beta, se, t, p]; the 't' column holds the
    cluster-robust z. R² is the model R² (no random effects; the reported
    variance-explained metric is the separate OLS pseudo-ΔR²).
    """
    fd_col   = 'fd'   if 'fd'   in df.columns else 'rest_mean_FD'
    site_col = 'study_site' if 'study_site' in df.columns else 'study_site_baseline'
    covars   = ['interview_age', 'sex_num', fd_col]

    tbl, meta = fit_ols_cluster_table(
        df, outcome_col, pred_cols, covars,
        site_col=site_col, family_col=family_col)
    if not meta['converged'] or tbl.empty:
        return None, 'skipped', np.nan, np.nan

    rows = [{'predictor': r['predictor'], 'beta': r['beta'], 'se': r['se'],
             't': r['z'], 'p': r['p']} for _, r in tbl.iterrows()]
    return pd.DataFrame(rows), meta['method'], np.nan, np.nan


# ── Delta-R² (OLS pseudo-R²) ─────────────────────────────────────────────────

def compute_delta_r2(df, pred_cols, label):
    fd_col   = 'fd'   if 'fd'   in df.columns else 'rest_mean_FD'
    site_col = 'study_site' if 'study_site' in df.columns else 'study_site_baseline'

    rows = []
    for net in NETWORKS:
        prop_col = f'prop_{net}'
        if prop_col not in df.columns:
            continue
        needed = [prop_col, 'interview_age', 'sex_num', fd_col, site_col] + pred_cols
        avail  = [c for c in needed if c in df.columns]
        tmp    = df[avail].dropna()
        if len(tmp) < 50:
            continue
        y      = tmp[prop_col].values
        site_d = pd.get_dummies(tmp[site_col], prefix='site', drop_first=True, dtype=float).values

        Xc = np.column_stack([
            np.ones(len(tmp)),
            tmp['interview_age'].values,
            tmp['sex_num'].values,
            tmp[fd_col].values,
            site_d,
        ])
        Xf = np.column_stack([
            np.ones(len(tmp)),
            tmp['interview_age'].values,
            tmp['sex_num'].values,
            tmp[fd_col].values,
            tmp[pred_cols].values,
            site_d,
        ])
        # Pseudo-R² = 1 - RSS_full / RSS_null
        rss_null = sm.OLS(y, Xc).fit().ssr
        rss_full = sm.OLS(y, Xf).fit().ssr
        r2_cov   = sm.OLS(y, Xc).fit().rsquared
        r2_full  = sm.OLS(y, Xf).fit().rsquared
        delta_r2 = 1.0 - (rss_full / rss_null) if rss_null > 0 else np.nan

        rows.append({
            'timepoint':     label,
            'network':       net,
            'R2_covariates': round(r2_cov, 6),
            'R2_full':       round(r2_full, 6),
            'delta_R2':      round(delta_r2, 6),
        })
        log(f'  [{label}] {net}: pseudo-ΔR²={delta_r2:.5f}')
    return pd.DataFrame(rows)


# ── Primary models per timepoint ─────────────────────────────────────────────

def run_primary_models(df, pred_cols, n_tests, label):
    all_rows, r2_rows = [], []
    method_used = {}

    for net in NETWORKS:
        prop_col = f'prop_{net}'
        if prop_col not in df.columns:
            continue
        avail_preds = [p for p in pred_cols if p in df.columns]
        res_df, method, r2m, r2c = fit_model(df, prop_col, avail_preds)
        if res_df is None:
            continue
        method_used[net] = method
        r2_rows.append({'network': net, 'method': method,
                        'R2_marginal': round(r2m, 6), 'R2_conditional': round(r2c, 6)})
        for _, row in res_df.iterrows():
            all_rows.append({'timepoint': label, 'network': net,
                             'method': method, **row.to_dict()})
        log(f'  [{label}] {net}: {method}, R²_marg={r2m:.5f}')

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    results = pd.DataFrame(all_rows)
    mask    = results['predictor'].isin(pred_cols)
    p_flat  = results.loc[mask, 'p'].values
    valid   = ~np.isnan(p_flat)
    q_out   = np.full(len(p_flat), np.nan)
    if valid.sum():
        _, q_v, _, _ = multipletests(p_flat[valid], alpha=0.05, method='fdr_bh')
        q_out[valid] = q_v
    results.loc[mask, 'q_FDR'] = q_out

    n_sig_fdr  = (q_out[valid] < 0.05).sum() if valid.sum() else 0
    bonf_alpha = 0.05 / n_tests
    n_sig_bonf = (p_flat[valid] < bonf_alpha).sum() if valid.sum() else 0
    log(f'\n  [{label}] FDR q<0.05: {n_sig_fdr}/{n_tests}  '
        f'Bonferroni: {n_sig_bonf}/{n_tests}')
    ols_nets = [n for n, m in method_used.items() if m == 'ols']
    if ols_nets:
        log(f'  [{label}] OLS fallback used for: {ols_nets}')

    return results, pd.DataFrame(r2_rows)


# ── Beta heatmap ─────────────────────────────────────────────────────────────

def beta_heatmap(results, pred_cols, pred_labels_short, networks, title, fname):
    beta_mat = pd.DataFrame(index=pred_cols, columns=networks, dtype=float)
    q_mat    = pd.DataFrame(index=pred_cols, columns=networks, dtype=float)

    for _, row in results.iterrows():
        net, pred = row['network'], row['predictor']
        if net in networks and pred in pred_cols:
            beta_mat.loc[pred, net] = row.get('beta', np.nan)
            q_mat.loc[pred, net]    = row.get('q_FDR', np.nan)

    bv = beta_mat.values.astype(float)
    qv = q_mat.values.astype(float)

    annot = np.full(bv.shape, '', dtype=object)
    for i in range(bv.shape[0]):
        for j in range(bv.shape[1]):
            b, q = bv[i, j], qv[i, j]
            if not np.isnan(b) and not np.isnan(q) and q < 0.05:
                annot[i, j] = f'{b:.4f}*'
            elif not np.isnan(b):
                annot[i, j] = f'{b:.4f}'

    n_preds = len(pred_labels_short)
    n_nets  = len(networks)
    fig_w = n_nets * 0.6 + 3
    fig_h = n_preds * 0.5 + 2
    clim = max(0.01, round(np.nanmax(np.abs(bv)) + 0.001, 3)) if not np.all(np.isnan(bv)) else 0.01

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        bv, annot=annot, fmt='',
        cmap='RdBu_r', center=0, vmin=-clim, vmax=clim,
        xticklabels=networks, yticklabels=pred_labels_short,
        linewidths=0.3, linecolor='#cccccc', ax=ax,
        annot_kws={'size': 7},
        cbar_kws={'label': 'Beta coefficient', 'shrink': 0.7},
    )
    for i in range(qv.shape[0]):
        for j in range(qv.shape[1]):
            if not np.isnan(qv[i, j]) and qv[i, j] < 0.05:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, fill=False, edgecolor='black', lw=1.5,
                ))
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel('Network', fontsize=10)
    ax.set_ylabel('Predictor', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    fig.text(0.5, -0.01, '* FDR q<0.05 (bold box)', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


# ── Run both analysis tags ────────────────────────────────────────────────────

ANALYSES = [
    {
        'tag':          'individual',
        'pred_cols':    ELA_COLS,
        'labels_short': [ELA_LABELS_SHORT[e] for e in ELA_COLS],
        'n_tests':      N_TESTS,
    },
    {
        'tag':          'composites',
        'pred_cols':    COMPOSITE_COLS,
        'labels_short': [COMPOSITE_LABELS_SHORT[c] for c in COMPOSITE_COLS],
        'n_tests':      N_TESTS_COMPOSITES,
    },
]

for analysis in ANALYSES:
    tag          = analysis['tag']
    pred_cols    = analysis['pred_cols']
    labels_short = analysis['labels_short']
    n_tests      = analysis['n_tests']

    log()
    log('=' * 70)
    log(f'ANALYSIS: {tag.upper()}  ({len(pred_cols)} predictors x {len(NETWORKS)} networks)')
    log('=' * 70)

    all_results = []
    all_dr2     = []
    surface_rows = []

    for tp_label, df_tp in TIMEPOINTS:
        avail_preds = [p for p in pred_cols if p in df_tp.columns]
        if not avail_preds:
            log(f'  [{tp_label}] WARNING: no predictor columns found, skipping.')
            continue

        log(f'\n  --- {tp_label} ---')
        log('  Primary models:')
        res, _ = run_primary_models(df_tp, avail_preds, n_tests, tp_label)

        log('  Delta-R²:')
        dr2 = compute_delta_r2(df_tp, avail_preds, tp_label)

        if not res.empty:
            res.to_csv(TAB_DIR / f'phase3_{tag}_results_{tp_label}.csv', index=False)
            all_results.append(res)

            # Report significant predictors
            sig = res[res.get('q_FDR', pd.Series(dtype=float)) < 0.05].copy()
            log(f'\n  [{tp_label}] FDR-sig predictors: {len(sig)}')
            if len(sig):
                log(sig[['network', 'predictor', 'beta', 'se', 't', 'p', 'q_FDR']].round(4).to_string(index=False))

            # Beta heatmap
            tp_title_map = {
                'baseline': 'Baseline (~9-10y)',
                'year2':    'Year-2 (~11-12y)',
                'year4':    'Year-4 (~13-14y)',
                'year6':    'Year-6 (~15-16y)',
            }
            beta_heatmap(
                res, avail_preds, labels_short, NETWORKS,
                title=f'ELA [{tag}] → Network Proportion: Beta ({tp_title_map[tp_label]})',
                fname=FIG_DIR / f'fig_phase3_{tag}_beta_heatmap_{tp_label}.png',
            )

        if not dr2.empty:
            dr2.to_csv(TAB_DIR / f'phase3_{tag}_delta_r2_{tp_label}.csv', index=False)
            all_dr2.append(dr2)

            # Build brain surface input rows
            for _, dr_row in dr2.iterrows():
                net = dr_row['network']
                surf_row = {
                    'timepoint': tp_label,
                    'network':   net,
                    'delta_R2':  dr_row['delta_R2'],
                }
                # Add predictor betas
                if not res.empty:
                    net_res = res[res['network'] == net]
                    for pred in avail_preds:
                        pred_row = net_res[net_res['predictor'] == pred]
                        beta_val = float(pred_row['beta'].iloc[0]) if len(pred_row) else np.nan
                        surf_row[f'{pred}_beta'] = beta_val
                surface_rows.append(surf_row)

    # Save brain surface inputs
    if surface_rows:
        pd.DataFrame(surface_rows).to_csv(
            TAB_DIR / f'phase3_{tag}_brain_surface_inputs.csv', index=False)
        log(f'\n  Saved phase3_{tag}_brain_surface_inputs.csv')

    # Delta-R² bar plot across timepoints
    if all_dr2:
        dr2_combined = pd.concat(all_dr2, ignore_index=True)
        tp_labels_plot = dr2_combined['timepoint'].unique()
        fig, axes = plt.subplots(1, len(tp_labels_plot),
                                 figsize=(5 * len(tp_labels_plot), 4))
        if len(tp_labels_plot) == 1:
            axes = [axes]
        for ax, tp_l in zip(axes, tp_labels_plot):
            dr = dr2_combined[dr2_combined['timepoint'] == tp_l]
            ax.bar(dr['network'], dr['delta_R2'] * 100, color='steelblue', edgecolor='black', lw=0.5)
            ax.set_title(tp_l.replace('year', 'Year-'), fontsize=11)
            ax.set_ylabel('Pseudo-ΔR² × 100 (%)', fontsize=9)
            ax.tick_params(axis='x', rotation=45)
        plt.suptitle(f'ELA [{tag}] — Unique Variance in Network Proportion', fontsize=11)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f'fig_phase3_{tag}_delta_r2.png')
        plt.close()
        log(f'  Saved fig_phase3_{tag}_delta_r2.png')

    log(f'\n  {tag.upper()} complete.')

log()
log('=' * 70)

with open(DAT_DIR / 'progress_log.txt', 'a') as f:
    f.write('\n\nPHASE 3 COMPLETE\n')
    f.write('\n'.join(log_lines[-60:]))

log('Phase 3 complete.')
