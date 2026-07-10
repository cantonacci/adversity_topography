"""
within_person_cbcl.py — Within-person SCAN × ELA × CBCL analyses.

Mirrors within_person_cognition.py EXACTLY,
swapping the 3 FDR-significant CBCL subscales in as outcomes:
    Attention Problems  (cbcl_scr_syn_attention_r)
    DSM5 ADHD           (cbcl_scr_dsm5_adhd_r)
    Thought Problems    (cbcl_scr_syn_thought_r)

Same machinery as cognition:
  (A) First-last: one change score per subject (first vs last usable wave)
  (B) All-waves:  every follow-up wave vs first wave, MixedLM (subject RE)
                  with family cluster-robust SEs.
  ELA = 4-item threat_composite (z). FDR BH within the 3-subscale family.

This is the change-based test that parallels Result 2/3 for cognition:
  Result 2: ΔSCAN  → ΔCBCL  (does SCAN change track CBCL change?)
  Result 3: ELA    → ΔCBCL  (does early threat predict CBCL change?)

CBCL subscales (raw _r scores) are pulled from the four wave dataframes
(df_base/df_y2/df_y4/df_y6) and merged onto scan_topo_long by (subject, wave).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests

from adtopo.config import DAT_DIR, CBCL_MEDIATION_OUTCOMES

DERIVED = Path(__file__).parent / 'derived'

print('=' * 68)
print('Within-person analyses — SCAN × ELA × CBCL')
print('=' * 68)

# All 14 CBCL subscales (identity mapping: keep the raw cbcl_scr_*_r column names as
# the working keys). Previously restricted to the 3 subscales that were significant in
# the cross-sectional mediation; with no significant cross-sectional effects there is no
# basis for that filter, so we test all 14 with FDR across the full family.
CBCL  = {src: src for src in CBCL_MEDIATION_OUTCOMES}   # source col -> working key (identity)
LABEL = dict(CBCL_MEDIATION_OUTCOMES)                   # source col -> display label

# ── Load long SCAN/topography + ELA ───────────────────────────────────────────
d   = pd.read_csv(DERIVED / 'scan_topo_long.csv')
ela = pd.read_csv(DERIVED / 'ela_scores.csv')
d = d[d['usable'].isin([True, 'True', 'TRUE'])].copy()
d = d.merge(ela, on='src_subject_id', how='left')
d['subject'] = d['src_subject_id'].astype(str)
z = lambda s: (s - s.mean()) / s.std()
d['ela'] = z(d['ela_threat'])

# ── Build long CBCL from the four wave dataframes, merge on (subject, wave) ────
wave_files = {'00A': 'df_base.csv', '02A': 'df_y2.csv',
              '04A': 'df_y4.csv',  '06A': 'df_y6.csv'}
frames = []
for w, f in wave_files.items():
    dd = pd.read_csv(DAT_DIR / f)
    keep = ['sub_ID'] + [c for c in CBCL if c in dd.columns]
    t = dd[keep].rename(columns={'sub_ID': 'src_subject_id', **CBCL})
    t['wave'] = w
    frames.append(t)
cbcl_long = pd.concat(frames, ignore_index=True)
d = d.merge(cbcl_long, on=['src_subject_id', 'wave'], how='left')

print(f'Usable rows: {len(d)},  subjects: {d["subject"].nunique()}')
for w in ['00A', '02A', '04A', '06A']:
    sub = d[d['wave'] == w]
    cov = {nm: int(sub[nm].notna().sum()) for nm in LABEL}
    print(f'  {w}: n={len(sub)}  CBCL non-missing {cov}')


# ── Cluster-robust OLS (family), df = n_clusters - 1 ──────────────────────────
def cluster_ols(y, X, clusters, term_idx):
    n, k = X.shape
    XtXinv = np.linalg.pinv(X.T @ X)
    betas  = XtXinv @ X.T @ y
    resid  = y - X @ betas
    uc = np.unique(clusters); G = len(uc)
    meat = np.zeros((k, k))
    for cl in uc:
        idx = clusters == cl
        sc = X[idx].T @ resid[idx]
        meat += np.outer(sc, sc)
    factor = (G / (G - 1)) * (n / (n - k))
    V = XtXinv @ (factor * meat) @ XtXinv
    b = betas[term_idx]; se = np.sqrt(V[term_idx, term_idx])
    t = b / se; p = 2 * stats.t.sf(abs(t), df=G - 1)
    return n, b, p


def run_ols(df, y_col, x_cols, cluster_col, term):
    tmp = df[x_cols + [y_col, cluster_col]].dropna().copy()
    if tmp['sex'].dtype == object:
        vals = tmp['sex'].unique()
        tmp['sex'] = (tmp['sex'] == vals[0]).astype(float)
    y = tmp[y_col].values.astype(float)
    X = np.column_stack([np.ones(len(tmp))] + [tmp[c].values.astype(float) for c in x_cols])
    return cluster_ols(y, X, tmp[cluster_col].values, x_cols.index(term) + 1)


def run_lme(df, y_col, x_cols, subject_col, cluster_col, term):
    needed = [y_col] + x_cols + [subject_col, cluster_col]
    tmp = df[[c for c in needed if c in df.columns]].dropna().copy()
    if tmp['sex'].dtype == object:
        vals = tmp['sex'].unique()
        tmp['sex'] = (tmp['sex'] == vals[0]).astype(float)
    y = tmp[y_col].values.astype(float)
    X = np.column_stack([np.ones(len(tmp))] + [tmp[c].values.astype(float) for c in x_cols])
    groups = tmp[subject_col].values; cl = tmp[cluster_col].values
    try:
        res = MixedLM(y, X, groups=groups, exog_re=np.ones((len(tmp), 1))).fit(
            reml=True, method='lbfgs', maxiter=300)
        if not res.converged:
            raise RuntimeError('no converge')
        betas = res.fe_params.values; resid = y - X @ betas
    except Exception:
        betas = np.linalg.lstsq(X, y, rcond=None)[0]; resid = y - X @ betas
    n, k = X.shape
    XtXinv = np.linalg.pinv(X.T @ X)
    uc = np.unique(cl); G = len(uc)
    meat = np.zeros((k, k))
    for c_ in uc:
        idx = cl == c_
        sc = X[idx].T @ resid[idx]
        meat += np.outer(sc, sc)
    factor = (G / (G - 1)) * (n / (n - k))
    V = XtXinv @ (factor * meat) @ XtXinv
    idx = x_cols.index(term) + 1
    b = betas[idx]; se = np.sqrt(V[idx, idx])
    t = b / se; p = 2 * stats.t.sf(abs(t), df=G - 1)
    return len(tmp), tmp[subject_col].nunique(), b, p


# ── Change-score builders (carry one CBCL outcome at a time) ──────────────────
def _rows(df, outcome, all_waves):
    df = df.sort_values(['subject', 'years'])
    cts = df.groupby('subject').size()
    df = df[df['subject'].isin(cts[cts >= 2].index)]
    out = []
    for subj, g in df.groupby('subject', sort=False):
        g = g.dropna(subset=['scan_prop', outcome])
        if len(g) < 2:
            continue
        t1 = g.iloc[0]
        targets = [t2 for _, t2 in g.iloc[1:].iterrows()] if all_waves else [g.iloc[-1]]
        for t2 in targets:
            out.append(dict(
                subject=subj, family=t1['family_id'], sex=t1['sex'], ela=t1['ela'],
                d_sc=t2['scan_prop'] - t1['scan_prop'], base_sc=t1['scan_prop'],
                d_fd=t2['mean_FD'] - t1['mean_FD'], d_years=t2['years'] - t1['years'],
                base_age=t1['age'],
                d_out=t2[outcome] - t1[outcome], base_out=t1[outcome],
            ))
    return pd.DataFrame(out)


def build_first_last(df, outcome):
    return _rows(df, outcome, all_waves=False)


def build_all_waves(df, outcome):
    return _rows(df, outcome, all_waves=True)


def fmt(b, p):
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    return f'β={b:+.4f}  p={p:.4f} {sig}'


lines = ['Within-person SCAN × ELA × CBCL (all 14 subscales)',
         'ELA: 4-item threat_composite; FDR BH within 14-subscale family', '']

# ══════════════════════════════════════════════════════════════════════════════
# RESULT 2 (CBCL): ΔSCAN → ΔCBCL   (does SCAN change track CBCL change?)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 68)
print('RESULT 2 (CBCL): ΔSCAN → ΔCBCL  (baseline-adjusted, FDR within 3)')
print('=' * 68)

xcols2 = ['d_sc_z', 'bo', 'bs', 'd_fd', 'd_years', 'base_age', 'sex']
res2a, res2b = {}, {}
for nm in LABEL:
    src = [k for k, v in CBCL.items() if v == nm][0]
    dd = d[d['scan_prop'].notna() & d[nm].notna()].copy()

    a = build_first_last(dd, nm)
    a = a.dropna(subset=['d_sc', 'd_out', 'base_out', 'base_sc', 'd_fd', 'd_years', 'base_age'])
    a['d_sc_z'] = z(a['d_sc']); a['d_o'] = z(a['d_out'])
    a['bo'] = z(a['base_out']); a['bs'] = z(a['base_sc'])
    res2a[nm] = run_ols(a, 'd_o', xcols2, 'family', 'd_sc_z')

    aw = build_all_waves(dd, nm)
    aw = aw.dropna(subset=['d_sc', 'd_out', 'base_out', 'base_sc', 'd_fd', 'd_years', 'base_age'])
    aw['d_sc_z'] = z(aw['d_sc']); aw['d_o'] = z(aw['d_out'])
    aw['bo'] = z(aw['base_out']); aw['bs'] = z(aw['base_sc'])
    res2b[nm] = run_lme(aw, 'd_o', xcols2, 'subject', 'family', 'd_sc_z')

q2a = multipletests([res2a[nm][2] for nm in LABEL], method='fdr_bh')[1]
q2b = multipletests([res2b[nm][3] for nm in LABEL], method='fdr_bh')[1]
print('  (A) First-last:')
for i, nm in enumerate(LABEL):
    n, b, p = res2a[nm]
    print(f'      {LABEL[nm]:<20} N={n}  {fmt(b, p)}  q={q2a[i]:.4f}')
    lines.append(f'  R2 first-last {LABEL[nm]}: N={n} {fmt(b,p)} q={q2a[i]:.4f}')
print('  (B) All-waves:')
for i, nm in enumerate(LABEL):
    n, ns, b, p = res2b[nm]
    print(f'      {LABEL[nm]:<20} N={n} ({ns} subj)  {fmt(b, p)}  q={q2b[i]:.4f}')
    lines.append(f'  R2 all-waves {LABEL[nm]}: N={n} ({ns} subj) {fmt(b,p)} q={q2b[i]:.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# RESULT 3 (CBCL): ELA → ΔCBCL   (does early threat predict CBCL change?)
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 68)
print('RESULT 3 (CBCL): ELA → ΔCBCL  (baseline-adjusted, FDR within 3)')
print('=' * 68)

xcols3 = ['ela', 'bo', 'd_fd', 'd_years', 'base_age', 'sex']
res3a, res3b = {}, {}
for nm in LABEL:
    dd = d[d['scan_prop'].notna() & d[nm].notna()].copy()
    a = build_first_last(dd, nm)
    a = a.dropna(subset=['d_out', 'base_out', 'd_fd', 'd_years', 'base_age', 'ela'])
    a['d_o'] = z(a['d_out']); a['bo'] = z(a['base_out'])
    res3a[nm] = run_ols(a, 'd_o', xcols3, 'family', 'ela')

    aw = build_all_waves(dd, nm)
    aw = aw.dropna(subset=['d_out', 'base_out', 'd_fd', 'd_years', 'base_age', 'ela'])
    aw['d_o'] = z(aw['d_out']); aw['bo'] = z(aw['base_out'])
    res3b[nm] = run_lme(aw, 'd_o', xcols3, 'subject', 'family', 'ela')

q3a = multipletests([res3a[nm][2] for nm in LABEL], method='fdr_bh')[1]
q3b = multipletests([res3b[nm][3] for nm in LABEL], method='fdr_bh')[1]
print('  (A) First-last:')
for i, nm in enumerate(LABEL):
    n, b, p = res3a[nm]
    print(f'      {LABEL[nm]:<20} N={n}  {fmt(b, p)}  q={q3a[i]:.4f}')
    lines.append(f'  R3 first-last {LABEL[nm]}: N={n} {fmt(b,p)} q={q3a[i]:.4f}')
print('  (B) All-waves:')
for i, nm in enumerate(LABEL):
    n, ns, b, p = res3b[nm]
    print(f'      {LABEL[nm]:<20} N={n} ({ns} subj)  {fmt(b, p)}  q={q3b[i]:.4f}')
    lines.append(f'  R3 all-waves {LABEL[nm]}: N={n} ({ns} subj) {fmt(b,p)} q={q3b[i]:.4f}')

out = DERIVED / 'results_within_person_cbcl.txt'
out.write_text('\n'.join(lines))
print(f'\nResults written to {out}')
print('\nDone.')
