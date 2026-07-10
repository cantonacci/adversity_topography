"""
Phase 6: Mediation — ELA composites → network topography → outcomes

Predictors (X):
  threat_composite (primary; deprivation/unpredictability not included here
  because they are not tested in encroachment or FC analyses)

Mediators (M): all 15 network proportions (primary focus: SCAN)

Outcomes (Y):
  • CBCL subscales at year-6 follow-up (df_y6, ~ages 15-16)
  • NIH Toolbox fluid (year-6) and crystallized (year-6), already in df_base

All paths use OLS with fixed study-site dummies and FAMILY-cluster-robust SEs
(canonical cross-sectional spec; harmonized per Booil's model-consistency comment):
Path a:   OLS-cluster(M ~ X + age + sex + FD + C(site))
Path b+c': OLS-cluster(Y ~ X + M + age_y6 + sex + C(site_y6) + [matching baseline subscale])
Path c:   OLS-cluster(Y ~ X + age_y6 + sex + C(site_y6) + [matching baseline subscale])
Indirect = beta_a × beta_b
Bootstrap: 5000 cluster resamples (families), percentile CI
  (bootstrap uses precomputed numpy designs + OLS for both paths — fast)

For each CBCL outcome, the MATCHING baseline subscale is included as a covariate
(autoregressive / residualized-change model) to isolate prospective change in
that specific symptom domain. Decided 2026-06-17 after a covariate sensitivity
analysis (none / total / matching) showed the same FDR-surviving outcomes under
all schemes; matching gives the cleanest "prospective change" interpretation.
NIH Toolbox cognition outcomes do not receive a CBCL baseline covariate.

FDR correction: BH within threat_composite across 16 outcomes.

Outputs:
  TAB_DIR/phase6_mediation_SCAN.csv         — SCAN mediator, all predictors & outcomes
  TAB_DIR/phase6_mediation_allnetworks.csv  — all 15 networks
"""
import sys
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
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from pathlib import Path


def round_or_nan(x, ndigits):
    """Round x to ndigits, passing NaN through unchanged (for tidy result rows)."""
    return np.nan if x is None or np.isnan(x) else round(float(x), ndigits)


sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import (
    FIG_DIR, TAB_DIR, DAT_DIR,
    NETWORKS,
    COMPOSITE_COLS, COMPOSITE_LABELS_SHORT,
    ELA_COLS,
    CBCL_MEDIATION_OUTCOMES,
    NIH_MEDIATION_COLS, NIH_MEDIATION_LABELS,
    RANDOM_SEED,
)
from lib.re_models import fit_ols_cluster_table

np.random.seed(RANDOM_SEED)
N_BOOTSTRAP = 5000
N_JOBS = -1

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'axes.linewidth':     0.8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'legend.frameon':     False,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

log_lines = []
def log(msg=''):
    print(msg, flush=True)
    log_lines.append(str(msg))


# ── Load data ─────────────────────────────────────────────────────────────────

log('=' * 70)
log('PHASE 6: Mediation analysis (composites)')
log('=' * 70)

df_base = pd.read_csv(DAT_DIR / 'df_base.csv')
df_y6   = pd.read_csv(DAT_DIR / 'df_y6.csv')  # year-6 follow-up (~ages 15-16)
log(f'Loaded: df_base N={len(df_base)}, df_y6 N={len(df_y6)}')


# ── Predictor list ────────────────────────────────────────────────────────────

PREDICTORS  = ['threat_composite']
PRED_LABELS = COMPOSITE_LABELS_SHORT


# ── Column discovery helpers ──────────────────────────────────────────────────

def get_fd_site(df):
    fd   = 'fd'          if 'fd'          in df.columns else 'rest_mean_FD'
    site = 'study_site'  if 'study_site'  in df.columns else 'study_site_baseline'
    return fd, site


# ── Build merged dataset ──────────────────────────────────────────────────────

log()
log('Building merged baseline + year-6 dataset ...')

fd_base, site_base = get_fd_site(df_base)
fd_y6,   site_y6   = get_fd_site(df_y6)

# CBCL outcomes from year-6 dataframe
avail_cbcl = [c for c in CBCL_MEDIATION_OUTCOMES if c in df_y6.columns]
missing_cbcl = [c for c in CBCL_MEDIATION_OUTCOMES if c not in df_y6.columns]
if missing_cbcl:
    log(f'  WARNING: CBCL columns not in df_y6: {missing_cbcl}')

# NIH outcomes are already merged into df_base (nihtb_fluid_y6, nihtb_cryst_y6)
avail_nih = [c for c in NIH_MEDIATION_COLS if c in df_base.columns]
missing_nih = [c for c in NIH_MEDIATION_COLS if c not in df_base.columns]
if missing_nih:
    log(f'  WARNING: NIH columns not in df_base: {missing_nih}')

# Matching baseline subscale covariate: pull every baseline CBCL subscale from
# df_base and rename with a "_base" suffix so each year-6 outcome can be adjusted
# for its own baseline level (autoregressive / residualized change).
cbcl_base_avail = [c for c in CBCL_MEDIATION_OUTCOMES if c in df_base.columns]

base_keep = (
    ['sub_ID', 'family_id', 'interview_age', 'sex_num', fd_base, site_base] +
    PREDICTORS +
    [f'prop_{n}' for n in NETWORKS if f'prop_{n}' in df_base.columns] +
    avail_nih +
    cbcl_base_avail
)

y6_keep = ['sub_ID', 'interview_age', fd_y6, site_y6] + avail_cbcl

df_b = df_base[[c for c in base_keep if c in df_base.columns]].copy()
base_rename = {
    'interview_age': 'age_base',
    fd_base:         'fd_base',
    site_base:       'site_base',
}
# baseline subscale -> "<col>_base"  (avoid clashing with the year-6 outcome col)
for c in cbcl_base_avail:
    base_rename[c] = c + '_base'
df_b = df_b.rename(columns=base_rename)

df_6 = df_y6[[c for c in y6_keep if c in df_y6.columns]].copy()
df_6 = df_6.rename(columns={
    'interview_age': 'age_y6',
    fd_y6:           'fd_y6',
    site_y6:         'site_y6',
})

df_med = df_b.merge(df_6, on='sub_ID', how='inner')
log(f'  Merged N (inner join) = {len(df_med)}')

# Drop rows missing all composite predictors
df_med = df_med.dropna(subset=PREDICTORS)
log(f'  After dropping missing composite scores: N = {len(df_med)}')

outcome_cols  = avail_cbcl + avail_nih
outcome_labels = {**CBCL_MEDIATION_OUTCOMES, **NIH_MEDIATION_LABELS}


# ── Mediation helpers ─────────────────────────────────────────────────────────

def path_a_mixedlm(data, pred_col, med_col, age_col, fd_col, site_col, family_col):
    """Path a (X -> M) under the canonical reported spec
    (lib.re_models.fit_ols_cluster_table): M ~ X + age + sex + FD + C(site),
    OLS with family-cluster-robust SEs. This matches the family-clustered OLS
    used by the indirect-effect bootstrap. Returns (beta, se, z, p)."""
    tbl, meta = fit_ols_cluster_table(
        data, med_col, [pred_col], [age_col, 'sex_num', fd_col],
        site_col=site_col, family_col=family_col)
    if not meta['converged'] or tbl.empty:
        return np.nan, np.nan, np.nan, np.nan
    r = tbl.iloc[0]
    return float(r['beta']), float(r['se']), float(r['z']), float(r['p'])


def ols_cluster(data, y_col, x_cols, site_col=None, family_col='family_id'):
    """Path b / c' / c estimation: OLS with study-site fixed dummies and
    FAMILY-cluster-robust SEs, harmonized with path a and the rest of the paper's
    cross-sectional models (Booil's model-consistency comment). family_col is used
    only for clustering (not added to the design); falls back to HC3 if absent."""
    needed = [y_col] + x_cols + ([site_col] if site_col else []) + [family_col]
    tmp    = data[[c for c in dict.fromkeys(needed) if c in data.columns]].dropna().copy()
    if len(tmp) < 50:
        return None, tmp

    X_parts = []
    for c in x_cols:
        if c not in tmp.columns:
            continue
        if tmp[c].dtype == object or str(tmp[c].dtype) == 'category':
            X_parts.append(pd.get_dummies(tmp[c], prefix=c, drop_first=True, dtype=float).values)
        else:
            X_parts.append(tmp[c].values.astype(float))

    if site_col and site_col in tmp.columns:
        X_parts.append(pd.get_dummies(tmp[site_col], prefix='site', drop_first=True, dtype=float).values)

    if not X_parts:
        return None, tmp

    X = np.column_stack([np.ones(len(tmp))] + X_parts)
    try:
        if family_col in tmp.columns:
            res = sm.OLS(tmp[y_col].values.astype(float), X).fit(
                cov_type='cluster', cov_kwds={'groups': tmp[family_col].values})
        else:
            res = sm.OLS(tmp[y_col].values.astype(float), X).fit(cov_type='HC3')
        return res, tmp
    except Exception:
        return None, tmp


def build_covars(pred_col, med_col_or_none, outcome_col, is_cbcl_outcome):
    """
    Build covariate list for path b/c.
    CBCL outcomes include the MATCHING baseline subscale (outcome_col + '_base');
    NIH cognition outcomes do not.
    """
    base = [pred_col, 'age_y6', 'sex_num']
    if med_col_or_none:
        base = [pred_col, med_col_or_none] + [c for c in base if c != pred_col]
    if is_cbcl_outcome:
        match_base = outcome_col + '_base'
        if match_base in df_med.columns:
            base = base + [match_base]
    return [c for c in base if c in df_med.columns]


# ── Bootstrap helper ──────────────────────────────────────────────────────────

def _boot_fast(seed, fam_groups, Xa, ya, Xb, yb, med_idx):
    """Cluster (family) bootstrap of the indirect effect using precomputed
    numpy designs. Path a and path b are both OLS via lstsq — no pandas /
    get_dummies in the hot loop, so this is ~40x faster than rebuilding the
    design each iteration. Equivalent to the previous OLS bootstrap."""
    rng  = np.random.default_rng(seed)
    bi   = rng.integers(0, len(fam_groups), size=len(fam_groups))
    rows = np.concatenate([fam_groups[i] for i in bi])
    try:
        beta_a = np.linalg.lstsq(Xa[rows], ya[rows], rcond=None)[0][1]
        beta_b = np.linalg.lstsq(Xb[rows], yb[rows], rcond=None)[0][med_idx]
    except Exception:
        return np.nan
    if np.isnan(beta_a) or np.isnan(beta_b):
        return np.nan
    return beta_a * beta_b


def run_mediation(data, pred_col, med_col, outcome_col, n_boot=N_BOOTSTRAP):
    is_cbcl = outcome_col in CBCL_MEDIATION_OUTCOMES

    match_base = outcome_col + '_base'
    needed = ([pred_col, med_col, outcome_col,
               'age_base', 'fd_base', 'site_base', 'family_id', 'sex_num',
               'age_y6', 'site_y6'] +
              ([match_base] if is_cbcl and match_base in data.columns else []))
    tmp = data[[c for c in needed if c in data.columns]].dropna().copy().reset_index(drop=True)
    n   = len(tmp)
    if n < 80:
        return None

    unique_fams = tmp['family_id'].unique()
    fam_inv     = tmp['family_id'].map({f: i for i, f in enumerate(unique_fams)}).values
    fam_groups  = [np.where(fam_inv == i)[0] for i in range(len(unique_fams))]

    # Path a
    beta_a, se_a, t_a, p_a = path_a_mixedlm(
        tmp, pred_col, med_col,
        age_col='age_base', fd_col='fd_base', site_col='site_base',
        family_col='family_id',
    )

    # Path c (total)
    y_covars_c  = build_covars(pred_col, None, outcome_col, is_cbcl)
    res_c, _    = ols_cluster(tmp, outcome_col, y_covars_c, site_col='site_y6')
    beta_c      = res_c.params[1] if res_c is not None and len(res_c.params) > 1 else np.nan
    p_c         = res_c.pvalues[1] if res_c is not None and len(res_c.pvalues) > 1 else np.nan

    # Path b + c'
    y_covars_b   = build_covars(pred_col, med_col, outcome_col, is_cbcl)
    res_b, _     = ols_cluster(tmp, outcome_col, y_covars_b, site_col='site_y6')
    beta_c_prime  = res_b.params[1]  if res_b is not None and len(res_b.params)  > 1 else np.nan
    p_c_prime     = res_b.pvalues[1] if res_b is not None and len(res_b.pvalues) > 1 else np.nan
    beta_b        = res_b.params[2]  if res_b is not None and len(res_b.params)  > 2 else np.nan
    se_b          = res_b.bse[2]     if res_b is not None and len(res_b.bse)     > 2 else np.nan
    p_b           = res_b.pvalues[2] if res_b is not None and len(res_b.pvalues) > 2 else np.nan

    indirect = (beta_a * beta_b
                if not np.isnan(beta_a) and not np.isnan(beta_b) else np.nan)

    # ── Precompute numpy designs for the bootstrap (OLS both paths) ──────────
    # Path a design: [1, pred, age_base, sex_num, fd_base, site_base dummies]
    site_a = pd.get_dummies(tmp['site_base'], prefix='sa', drop_first=True, dtype=float)
    Xa = np.column_stack([
        np.ones(n), tmp[pred_col].values, tmp['age_base'].values,
        tmp['sex_num'].values, tmp['fd_base'].values, site_a.values,
    ]).astype(float)
    ya = tmp[med_col].values.astype(float)

    # Path b design: [1, <y_covars_b in order>, site_y6 dummies]
    # y_covars_b = [pred, med, age_y6, sex_num, (matching_base)] — all numeric.
    site_b = pd.get_dummies(tmp['site_y6'], prefix='s6', drop_first=True, dtype=float)
    Xb = np.column_stack(
        [np.ones(n)] + [tmp[c].values.astype(float) for c in y_covars_b] + [site_b.values]
    ).astype(float)
    yb = tmp[outcome_col].values.astype(float)
    med_idx = 1 + y_covars_b.index(med_col)   # design-column index of the mediator

    boot_samples = [
        _boot_fast(i, fam_groups, Xa, ya, Xb, yb, med_idx)
        for i in range(n_boot)
    ]
    boot_arr = np.array([b for b in boot_samples if b is not None and not np.isnan(b)])

    if len(boot_arr) < 100:
        ci_lo = ci_hi = boot_p = np.nan
    else:
        ci_lo = float(np.percentile(boot_arr, 2.5))
        ci_hi = float(np.percentile(boot_arr, 97.5))
        if not np.isnan(indirect):
            prop_opp = (np.mean(boot_arr < 0) if indirect > 0
                        else np.mean(boot_arr > 0))
            boot_p   = 2 * min(prop_opp, 1 - prop_opp)
        else:
            boot_p = np.nan

    return {
        'predictor':      pred_col,
        'mediator':       med_col,
        'outcome':        outcome_col,
        'n':              n,
        'beta_a':         round_or_nan(beta_a, 5),
        'se_a':           round_or_nan(se_a,   5),
        'p_a':            round_or_nan(p_a,    5),
        'beta_b':         round_or_nan(beta_b, 5),
        'se_b':           round_or_nan(se_b,   5),
        'p_b':            round_or_nan(p_b,    5),
        'beta_c':         round_or_nan(beta_c, 5),
        'p_c':            round_or_nan(p_c,    5),
        'beta_c_prime':   round_or_nan(beta_c_prime, 5),
        'p_c_prime':      round_or_nan(p_c_prime,    5),
        'indirect':       round_or_nan(indirect, 6),
        'boot_ci_lo':     round_or_nan(ci_lo,  6),
        'boot_ci_hi':     round_or_nan(ci_hi,  6),
        'boot_p':         round_or_nan(boot_p, 5),
        'boot_n_valid':   len(boot_arr),
        'cbcl_base_cov':  is_cbcl and match_base in tmp.columns,
        'cbcl_base_cov_col': match_base if (is_cbcl and match_base in tmp.columns) else '',
    }


# ── Run all triads ────────────────────────────────────────────────────────────

log()
log('=' * 70)
log(f'STEP 6.1 — Mediation: predictor → network (baseline) → outcomes (year-6)')
log(f'Predictors: {PREDICTORS}')
log(f'Bootstraps: {N_BOOTSTRAP}  |  Outcomes: {len(outcome_cols)}')
log('=' * 70)

triads = [
    (pred, net, f'prop_{net}', outcome)
    for pred    in PREDICTORS
    for net     in NETWORKS
    for outcome in outcome_cols
    if f'prop_{net}' in df_med.columns
    and pred         in df_med.columns
    and outcome      in df_med.columns
]
log(f'  Total triads: {len(triads)}  |  N_JOBS={N_JOBS}')

triad_results = Parallel(n_jobs=N_JOBS, backend='loky')(
    delayed(run_mediation)(df_med, pred, med_col, outcome)
    for pred, net, med_col, outcome in triads
)

# ── Post-process: per-network FDR ─────────────────────────────────────────────

all_rows_scan = []
all_rows_full = []

for net in NETWORKS:
    med_col = f'prop_{net}'
    net_rows = []
    for (pred, n, mc, outcome), result in zip(triads, triad_results):
        if n != net or result is None:
            continue
        ind  = result['indirect']
        bp   = result['boot_p']
        clo  = result['boot_ci_lo']
        chi  = result['boot_ci_hi']
        log(f'  {pred} → {net} → {outcome[:28]}'
            f'  a={result["beta_a"]:.4f}(p={result["p_a"]:.3f})'
            f'  indirect={ind:.4f}  CI=[{clo:.4f},{chi:.4f}]  boot_p={bp:.4f}')
        net_rows.append(result)

    if net_rows:
        df_net = pd.DataFrame(net_rows)
        p_boot = df_net['boot_p'].values
        valid  = ~np.isnan(p_boot)
        q_out  = np.full(len(p_boot), np.nan)
        if valid.sum():
            _, q_v, _, _ = multipletests(p_boot[valid], alpha=0.05, method='fdr_bh')
            q_out[valid] = q_v
        df_net['q_FDR_indirect'] = q_out

        n_sig = int((q_out[valid] < 0.05).sum()) if valid.sum() else 0
        log(f'  [{net}] FDR-sig indirect effects: {n_sig}/{len(net_rows)}')

        all_rows_full.extend(df_net.to_dict('records'))
        if net == 'SCAN':
            all_rows_scan.extend(df_net.to_dict('records'))


# ── Save results ──────────────────────────────────────────────────────────────

log()
log('=' * 70)
log('STEP 6.2 — Saving results')
log('=' * 70)

def _label_df(df_out):
    df_out['outcome_label']    = df_out['outcome'].map(lambda x: outcome_labels.get(x, x))
    df_out['predictor_label']  = df_out['predictor'].map(lambda x: PRED_LABELS.get(x, x))
    df_out['mediation_sig']    = (
        df_out['boot_ci_lo'].notna() &
        (df_out['boot_ci_lo'] * df_out['boot_ci_hi'] > 0)
    )
    return df_out

if all_rows_scan:
    df_scan = _label_df(pd.DataFrame(all_rows_scan))
    df_scan.to_csv(TAB_DIR / 'phase6_mediation_SCAN.csv', index=False)
    log(f'  Saved phase6_mediation_SCAN.csv  ({len(df_scan)} rows)')

if all_rows_full:
    df_full = _label_df(pd.DataFrame(all_rows_full))
    df_full.to_csv(TAB_DIR / 'phase6_mediation_allnetworks.csv', index=False)
    log(f'  Saved phase6_mediation_allnetworks.csv  ({len(df_full)} rows)')


# ── Print SCAN summary ────────────────────────────────────────────────────────

if all_rows_scan:
    log()
    log('=' * 70)
    log('STEP 6.3 — SCAN mediation summary')
    log('=' * 70)
    sig_mask = df_scan['mediation_sig']
    log(f'  Total rows: {len(df_scan)}')
    log(f'  CI excludes zero: {sig_mask.sum()}')
    log(f'  FDR q<0.05: {(df_scan["q_FDR_indirect"] < 0.05).sum()}')
    if sig_mask.sum():
        cols_show = ['predictor_label', 'outcome_label', 'beta_a', 'p_a',
                     'beta_b', 'p_b', 'indirect', 'boot_ci_lo', 'boot_ci_hi',
                     'boot_p', 'q_FDR_indirect', 'mediation_sig']
        cols_show = [c for c in cols_show if c in df_scan.columns]
        log('\n  *** Significant (CI excludes zero) ***')
        log(df_scan.loc[sig_mask, cols_show].to_string(index=False))


# ── Figure: SCAN indirect-effect heatmap ─────────────────────────────────────

if all_rows_scan:
    log()
    log('Generating SCAN mediation heatmap ...')
    df_s       = df_scan.copy()
    avail_preds = [p for p in PREDICTORS if p in df_s['predictor'].unique()]
    avail_out   = [o for o in outcome_cols if o in df_s['outcome'].unique()]

    ind_mat = pd.DataFrame(index=avail_preds, columns=avail_out, dtype=float)
    sig_mat = pd.DataFrame(index=avail_preds, columns=avail_out, dtype=bool)
    q_mat   = pd.DataFrame(index=avail_preds, columns=avail_out, dtype=float)

    for _, row in df_s.iterrows():
        p = row['predictor']; o = row['outcome']
        if p in avail_preds and o in avail_out:
            ind_mat.loc[p, o] = row['indirect']
            sig_mat.loc[p, o] = bool(row.get('mediation_sig', False))
            q_mat.loc[p, o]   = row.get('q_FDR_indirect', np.nan)

    iv = ind_mat.values.astype(float)
    qv = q_mat.values.astype(float)

    out_labels_list  = [outcome_labels.get(o, o) for o in avail_out]
    pred_labels_list = [PRED_LABELS.get(p, p)    for p in avail_preds]

    clim  = max(0.005, np.nanmax(np.abs(iv))) if not np.all(np.isnan(iv)) else 0.005
    n_p   = len(avail_preds)
    n_o   = len(avail_out)
    fig_w = n_o * 0.75 + 3
    fig_h = n_p * 0.7  + 2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    annot = np.full(iv.shape, '', dtype=object)
    for i in range(iv.shape[0]):
        for j in range(iv.shape[1]):
            if np.isnan(qv[i, j]) or qv[i, j] >= 0.05:
                continue
            sig = not (np.isnan(ind_mat.values[i, j]))
            ci_lo_val = float(df_s.loc[
                (df_s['predictor'] == avail_preds[i]) &
                (df_s['outcome']   == avail_out[j]), 'boot_ci_lo'
            ].values[0]) if len(df_s.loc[
                (df_s['predictor'] == avail_preds[i]) &
                (df_s['outcome']   == avail_out[j])]) else np.nan
            ci_hi_val = float(df_s.loc[
                (df_s['predictor'] == avail_preds[i]) &
                (df_s['outcome']   == avail_out[j]), 'boot_ci_hi'
            ].values[0]) if len(df_s.loc[
                (df_s['predictor'] == avail_preds[i]) &
                (df_s['outcome']   == avail_out[j])]) else np.nan
            if not (np.isnan(ci_lo_val) or np.isnan(ci_hi_val)):
                if ci_lo_val * ci_hi_val > 0:
                    annot[i, j] = f'{iv[i,j]:.3f}*'
                else:
                    annot[i, j] = f'{iv[i,j]:.3f}'

    sns.heatmap(
        iv, annot=annot, fmt='',
        cmap='RdBu_r', center=0, vmin=-clim, vmax=clim,
        xticklabels=out_labels_list, yticklabels=pred_labels_list,
        linewidths=0.3, linecolor='#cccccc', ax=ax,
        annot_kws={'size': 7},
        cbar_kws={'label': 'Indirect effect (a×b)', 'shrink': 0.7},
    )
    ax.set_title('SCAN Mediation — Indirect Effects\n(* FDR q<0.05; CI excludes 0)',
                 fontsize=11, pad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig_phase6_SCAN_mediation_heatmap.png')
    plt.close(fig)
    log('  Saved fig_phase6_SCAN_mediation_heatmap.png')

log()
log('Phase 6 complete.')
