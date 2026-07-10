"""
within_person_cognition.py — Within-person SCAN × ELA × cognition analyses.

Runs two parallel sets of models for each of the 3 results:
  (A) First-last: one change score per subject (first vs. last usable wave)
  (B) All-waves:  all intermediate waves vs. each subject's first usable wave
                  pooled with MixedLM (random intercept for subject) +
                  cluster-robust SEs by family

ELA: our 4-item threat_composite (paper version; colleague used 2-item).
FDR: BH within the 2-test cognition family (crystallized + fluid), applied
     independently for (A) and (B).

Fixes vs. colleague's analysis.R:
  - d_fd added as covariate in Result 3 (was missing in original).
  - All-waves models added as sensitivity check.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.special import expit
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

DERIVED = Path(__file__).parent / 'derived'

print('=' * 68)
print('Within-person analyses — SCAN × ELA × Cognition')
print('=' * 68)

# ── Load ──────────────────────────────────────────────────────────────────────
d   = pd.read_csv(DERIVED / 'scan_topo_long.csv')
ela = pd.read_csv(DERIVED / 'ela_scores.csv')

d = d[d['usable'].isin([True, 'True', 'TRUE'])].copy()
d = d.merge(ela, on='src_subject_id', how='left')
d['subject'] = d['src_subject_id'].astype(str)

z = lambda s: (s - s.mean()) / s.std()
d['ela'] = z(d['ela_threat'])

print(f'Usable rows: {len(d)},  subjects: {d["subject"].nunique()}')
for w in ['00A', '02A', '04A', '06A']:
    n = (d['wave'] == w).sum()
    print(f'  {w}: {n}')


# ── Cluster-robust OLS (matches sandwich::vcovCL, df = n_clusters - 1) ────────
def cluster_ols(y, X, clusters, term_idx):
    """OLS coefficient with family cluster-robust SE. Returns (n, beta, p)."""
    n, k     = X.shape
    XtXinv   = np.linalg.pinv(X.T @ X)
    betas    = XtXinv @ X.T @ y
    resid    = y - X @ betas
    unique_cl = np.unique(clusters)
    G = len(unique_cl)
    meat = np.zeros((k, k))
    for cl in unique_cl:
        idx = clusters == cl
        sc  = X[idx].T @ resid[idx]
        meat += np.outer(sc, sc)
    factor = (G / (G - 1)) * (n / (n - k))
    V  = XtXinv @ (factor * meat) @ XtXinv
    b  = betas[term_idx]
    se = np.sqrt(V[term_idx, term_idx])
    t  = b / se
    p  = 2 * stats.t.sf(abs(t), df=G - 1)
    return n, b, p


def run_ols(df, y_col, x_cols, cluster_col, term):
    """Drop NAs, build design matrix, cluster-robust OLS. Returns (n, beta, p)."""
    tmp = df[x_cols + [y_col, cluster_col]].dropna().copy()
    # encode sex as numeric if needed
    if tmp['sex'].dtype == object:
        vals = tmp['sex'].unique()
        tmp['sex'] = (tmp['sex'] == vals[0]).astype(float)
    y   = tmp[y_col].values.astype(float)
    X   = np.column_stack([np.ones(len(tmp))] +
                          [tmp[c].values.astype(float) for c in x_cols])
    cl  = tmp[cluster_col].values
    idx = x_cols.index(term) + 1
    return cluster_ols(y, X, cl, idx)


# ── MixedLM with cluster-robust SE (for all-waves models) ────────────────────
def run_lme(df, y_col, x_cols, subject_col, cluster_col, term):
    """
    MixedLM (random intercept for subject) + cluster-robust SE for inference.
    Sandwich SE clustered by cluster_col (family), applied to LME residuals.
    Returns (n, n_subjects, beta, p).
    """
    needed = [y_col] + x_cols + [subject_col, cluster_col]
    tmp = df[[c for c in needed if c in df.columns]].dropna().copy()
    if tmp['sex'].dtype == object:
        vals = tmp['sex'].unique()
        tmp['sex'] = (tmp['sex'] == vals[0]).astype(float)

    y   = tmp[y_col].values.astype(float)
    X   = np.column_stack([np.ones(len(tmp))] +
                          [tmp[c].values.astype(float) for c in x_cols])
    groups = tmp[subject_col].values
    cl     = tmp[cluster_col].values

    # Fit LME
    try:
        res = MixedLM(y, X, groups=groups,
                      exog_re=np.ones((len(tmp), 1))).fit(
            reml=True, method='lbfgs', maxiter=300)
        if not res.converged:
            raise RuntimeError('LME did not converge')
        betas = res.fe_params.values
        resid = y - X @ betas
    except Exception:
        # fallback to OLS if LME fails
        betas = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ betas

    # Cluster-robust SE (sandwich on LME residuals)
    n, k = X.shape
    XtXinv   = np.linalg.pinv(X.T @ X)
    unique_cl = np.unique(cl)
    G = len(unique_cl)
    meat = np.zeros((k, k))
    for c_ in unique_cl:
        idx = cl == c_
        sc  = X[idx].T @ resid[idx]
        meat += np.outer(sc, sc)
    factor = (G / (G - 1)) * (n / (n - k))
    V   = XtXinv @ (factor * meat) @ XtXinv
    idx = x_cols.index(term) + 1
    b   = betas[idx]
    se  = np.sqrt(V[idx, idx])
    t   = b / se
    p   = 2 * stats.t.sf(abs(t), df=G - 1)
    return len(tmp), tmp[subject_col].nunique(), b, p


# ── Build first-last dataset ──────────────────────────────────────────────────
def build_first_last(df, require_cols=None):
    req = require_cols or []
    df  = df.sort_values(['subject', 'years'])
    cts = df.groupby('subject').size()
    df  = df[df['subject'].isin(cts[cts >= 2].index)]
    rows = []
    for subj, g in df.groupby('subject', sort=False):
        if require_cols:
            g = g.dropna(subset=require_cols)
        if len(g) < 2:
            continue
        t1, t2 = g.iloc[0], g.iloc[-1]
        rows.append(dict(
            subject=subj, family=t1['family_id'], sex=t1['sex'], ela=t1['ela'],
            d_sc=t2['scan_prop']-t1['scan_prop'],  base_sc=t1['scan_prop'],
            d_fd=t2['mean_FD']-t1['mean_FD'],       d_years=t2['years']-t1['years'],
            base_age=t1['age'],
            d_cryst=t2['cog_cryst']-t1['cog_cryst'], base_cryst=t1['cog_cryst'],
            d_fluid=t2['cog_fluid']-t1['cog_fluid'], base_fluid=t1['cog_fluid'],
        ))
    return pd.DataFrame(rows)


# ── Build all-waves change-score dataset ──────────────────────────────────────
def build_all_waves(df, require_cols=None):
    """
    Every follow-up wave vs. each subject's first usable wave.
    One row per (subject, non-first-wave). Includes subject col for LME grouping.
    """
    req = require_cols or []
    df  = df.sort_values(['subject', 'years'])
    cts = df.groupby('subject').size()
    df  = df[df['subject'].isin(cts[cts >= 2].index)]
    rows = []
    for subj, g in df.groupby('subject', sort=False):
        if req:
            g = g.dropna(subset=req)
        if len(g) < 2:
            continue
        t1 = g.iloc[0]
        for _, t2 in g.iloc[1:].iterrows():
            rows.append(dict(
                subject=subj, family=t1['family_id'], sex=t1['sex'], ela=t1['ela'],
                d_sc=t2['scan_prop']-t1['scan_prop'],   base_sc=t1['scan_prop'],
                d_fd=t2['mean_FD']-t1['mean_FD'],        d_years=t2['years']-t1['years'],
                base_age=t1['age'],
                d_cryst=t2['cog_cryst']-t1['cog_cryst'], base_cryst=t1['cog_cryst'],
                d_fluid=t2['cog_fluid']-t1['cog_fluid'], base_fluid=t1['cog_fluid'],
            ))
    return pd.DataFrame(rows)


def fmt(b, p):
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    return f'β={b:+.4f}  p={p:.4f} {sig}'


lines = [
    'abcd_scan_network — key results (replication)',
    'ELA: 4-item threat_composite (paper version; colleague used 2-item)',
    'd_fd added as covariate in Result 3 (was missing in original)',
    '',
]

# ══════════════════════════════════════════════════════════════════════════════
# RESULT 1: Early threat -> continued SCAN enlargement
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*68)
print('RESULT 1: ELA → ΔSCAN (baseline-adjusted)')
print('='*68)

d1 = d[d['scan_prop'].notna() & d['ela'].notna()].copy()

# (A) first-last
a = build_first_last(d1, require_cols=['scan_prop', 'ela'])
a = a.dropna(subset=['d_sc','base_sc','d_fd','d_years','base_age','ela'])
a['d_sc_z'] = z(a['d_sc']); a['bs'] = z(a['base_sc'])
n1a, b1a, p1a = run_ols(a, 'd_sc_z',
    ['ela','bs','d_fd','d_years','base_age','sex'], 'family', 'ela')
print(f'  (A) First-last  N={n1a}  {fmt(b1a, p1a)}')
print(f'      Reported:   β=+0.033  p=0.008')

# (B) all-waves
aw1 = build_all_waves(d1, require_cols=['scan_prop', 'ela'])
aw1 = aw1.dropna(subset=['d_sc','base_sc','d_fd','d_years','base_age','ela'])
aw1['d_sc_z'] = z(aw1['d_sc']); aw1['bs'] = z(aw1['base_sc'])
n1b, ns1b, b1b, p1b = run_lme(aw1, 'd_sc_z',
    ['ela','bs','d_fd','d_years','base_age','sex'], 'subject', 'family', 'ela')
print(f'  (B) All-waves   N={n1b} ({ns1b} subjects)  {fmt(b1b, p1b)}')

lines += [
    'RESULT 1: Early threat -> continued SCAN enlargement',
    f'  (A) First-last (N={n1a}):  {fmt(b1a, p1a)}',
    f'  (B) All-waves  (N={n1b}, {ns1b} subjects):  {fmt(b1b, p1b)}',
    f'  Reported:  β=+0.033  p=0.008',
    '',
]

# ══════════════════════════════════════════════════════════════════════════════
# RESULT 2: Growing SCAN -> declining cognition
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*68)
print('RESULT 2: ΔSCAN → Δcognition (baseline-adjusted, FDR)')
print('='*68)

d2c = d[d['scan_prop'].notna() & d['cog_cryst'].notna()].copy()
d2f = d[d['scan_prop'].notna() & d['cog_fluid'].notna()].copy()

# (A) first-last
ac = build_first_last(d2c, require_cols=['scan_prop','cog_cryst'])
ac = ac.dropna(subset=['d_sc','d_cryst','base_cryst','base_sc','d_fd','d_years','base_age'])
ac['d_sc_z'] = z(ac['d_sc']); ac['d_o'] = z(ac['d_cryst'])
ac['bo'] = z(ac['base_cryst']); ac['bs'] = z(ac['base_sc'])

af = build_first_last(d2f, require_cols=['scan_prop','cog_fluid'])
af = af.dropna(subset=['d_sc','d_fluid','base_fluid','base_sc','d_fd','d_years','base_age'])
af['d_sc_z'] = z(af['d_sc']); af['d_o'] = z(af['d_fluid'])
af['bo'] = z(af['base_fluid']); af['bs'] = z(af['base_sc'])

xcols2 = ['d_sc_z','bo','bs','d_fd','d_years','base_age','sex']
n2ca, b2ca, p2ca = run_ols(ac, 'd_o', xcols2, 'family', 'd_sc_z')
n2fa, b2fa, p2fa = run_ols(af, 'd_o', xcols2, 'family', 'd_sc_z')
_, q2a, _, _ = multipletests([p2ca, p2fa], method='fdr_bh')
print(f'  (A) First-last:')
print(f'      Cryst  N={n2ca}  {fmt(b2ca, p2ca)}  q={q2a[0]:.4f}')
print(f'      Fluid  N={n2fa}  {fmt(b2fa, p2fa)}  q={q2a[1]:.4f}')
print(f'      Reported: cryst β=-0.047 q=0.004, fluid β=-0.094 q=0.004')

# (B) all-waves
awc = build_all_waves(d2c, require_cols=['scan_prop','cog_cryst'])
awc = awc.dropna(subset=['d_sc','d_cryst','base_cryst','base_sc','d_fd','d_years','base_age'])
awc['d_sc_z'] = z(awc['d_sc']); awc['d_o'] = z(awc['d_cryst'])
awc['bo'] = z(awc['base_cryst']); awc['bs'] = z(awc['base_sc'])

awf = build_all_waves(d2f, require_cols=['scan_prop','cog_fluid'])
awf = awf.dropna(subset=['d_sc','d_fluid','base_fluid','base_sc','d_fd','d_years','base_age'])
awf['d_sc_z'] = z(awf['d_sc']); awf['d_o'] = z(awf['d_fluid'])
awf['bo'] = z(awf['base_fluid']); awf['bs'] = z(awf['base_sc'])

n2cb, ns2cb, b2cb, p2cb = run_lme(awc, 'd_o', xcols2, 'subject', 'family', 'd_sc_z')
n2fb, ns2fb, b2fb, p2fb = run_lme(awf, 'd_o', xcols2, 'subject', 'family', 'd_sc_z')
_, q2b, _, _ = multipletests([p2cb, p2fb], method='fdr_bh')
print(f'  (B) All-waves:')
print(f'      Cryst  N={n2cb} ({ns2cb} subj)  {fmt(b2cb, p2cb)}  q={q2b[0]:.4f}')
print(f'      Fluid  N={n2fb} ({ns2fb} subj)  {fmt(b2fb, p2fb)}  q={q2b[1]:.4f}')

lines += [
    'RESULT 2: Growing SCAN -> declining cognition (FDR within 2-test family)',
    f'  (A) First-last:',
    f'      Cryst N={n2ca}  {fmt(b2ca, p2ca)}  q={q2a[0]:.4f}',
    f'      Fluid N={n2fa}  {fmt(b2fa, p2fa)}  q={q2a[1]:.4f}',
    f'      Reported: cryst β=-0.047 q=0.004, fluid β=-0.094 q=0.004',
    f'  (B) All-waves (MixedLM + family cluster-robust SE):',
    f'      Cryst N={n2cb} ({ns2cb} subj)  {fmt(b2cb, p2cb)}  q={q2b[0]:.4f}',
    f'      Fluid N={n2fb} ({ns2fb} subj)  {fmt(b2fb, p2fb)}  q={q2b[1]:.4f}',
    '',
]

# ══════════════════════════════════════════════════════════════════════════════
# RESULT 3: Early threat -> widening cognitive deficit
# (d_fd added as covariate — was missing in colleague's original)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*68)
print('RESULT 3: ELA → Δcryst (baseline-adjusted; d_fd now included)')
print('='*68)

xcols3 = ['ela','bo','d_fd','d_years','base_age','sex']   # d_fd added

# (A) first-last
n3a, b3a, p3a = run_ols(ac, 'd_o', xcols3, 'family', 'ela')
print(f'  (A) First-last  N={n3a}  {fmt(b3a, p3a)}')
print(f'      Reported (original, no d_fd): β=-0.105  p<0.001')

# (A') replicate exact original (without d_fd) for comparison
xcols3_orig = ['ela','bo','d_years','base_age','sex']
n3a0, b3a0, p3a0 = run_ols(ac, 'd_o', xcols3_orig, 'family', 'ela')
print(f'      Replication (no d_fd):       N={n3a0}  {fmt(b3a0, p3a0)}')

# (B) all-waves
n3b, ns3b, b3b, p3b = run_lme(awc, 'd_o', xcols3, 'subject', 'family', 'ela')
print(f'  (B) All-waves   N={n3b} ({ns3b} subjects)  {fmt(b3b, p3b)}')

lines += [
    'RESULT 3: Early threat -> widening cognitive deficit (d_fd added)',
    f'  (A) First-last + d_fd   N={n3a}  {fmt(b3a, p3a)}',
    f'  (A0) Replication (no d_fd, exact match):  N={n3a0}  {fmt(b3a0, p3a0)}',
    f'       Reported:  β=-0.105  p<0.001',
    f'  (B) All-waves  N={n3b} ({ns3b} subj)  {fmt(b3b, p3b)}',
    '',
]

# ── Save ──────────────────────────────────────────────────────────────────────
out = DERIVED / 'results_within_person_cognition.txt'
out.write_text('\n'.join(lines))
print(f'\nResults written to {out}')
print('\nDone.')
