"""
Supplement: Bayes factors for the expansion associations (Poldrack #16/#21).
adversity_topography/code/supplement/bayes_factors_expansion.py

At the very large ABCD N, near-zero p-values are hard to interpret and null
results (deprivation / unpredictability) cannot be quantified by NHST. We report
BIC-approximation Bayes factors for the ELA-composite → network-area associations,
computed from the same fixed-site OLS models used throughout (BIC is a
likelihood criterion, unaffected by the cluster-robust SE correction).

For a coefficient, BF10 = exp((BIC_null - BIC_full) / 2), where the null model
drops that predictor and retains all covariates (age, sex, FD) and fixed site.
BF10 > 1 favors an association (H1); BF10 < 1 favors the null (report as
BF01 = 1/BF10). Kass & Raftery (1995) benchmarks: >3 positive, >20 strong,
>150 very strong.

Outputs:
  TAB_DIR/supp_bayes_factors_bivariate.csv     (each composite × each network)
  TAB_DIR/supp_bayes_factors_multivariate_SCAN.csv  (3 composites, joint model, SCAN)
  DAT_DIR/supp_bayes_factors.txt
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

_CODE = next(a for a in Path(__file__).resolve().parents if (a / 'config.py').exists())
sys.path.insert(0, str(_CODE))
from config import TAB_DIR, DAT_DIR, NETWORKS, COMPOSITE_COLS, COMPOSITE_LABELS

COVARS = ['interview_age', 'sex_num', 'fd']
SITE = 'study_site'

log_lines = []
def log(m=''):
    print(m, flush=True); log_lines.append(str(m))


def _bic(df, outcome, terms):
    """BIC of OLS: outcome ~ terms + C(site). terms is a list (may be empty)."""
    rhs = ' + '.join(terms + [f'C({SITE})']) if terms else f'C({SITE})'
    return smf.ols(f'{outcome} ~ {rhs}', data=df).fit().bic


def bf10_from_bic(bic_null, bic_full):
    return float(np.exp((bic_null - bic_full) / 2.0))


def label_bf(bf10):
    """Kass & Raftery verbal label for the evidence (toward H1 or H0)."""
    bf, direction = (bf10, 'H1') if bf10 >= 1 else (1.0 / bf10, 'H0')
    if bf < 3:    strength = 'not worth mentioning'
    elif bf < 20: strength = 'positive'
    elif bf < 150: strength = 'strong'
    else:         strength = 'very strong'
    return f'{strength} for {direction}'


df = pd.read_csv(DAT_DIR / 'df_base.csv')
log(f'Loaded df_base N={len(df)}\n')

# ── (1) Bivariate: each composite × each network ──────────────────────────────
log('=' * 74)
log('(1) BIVARIATE Bayes factors — composite → prop_NET (one predictor at a time)')
log('=' * 74)
rows = []
for comp in COMPOSITE_COLS:
    for net in NETWORKS:
        outcome = f'prop_{net}'
        need = [outcome, comp] + COVARS + [SITE]
        tmp = df[[c for c in need if c in df.columns]].dropna().copy()
        tmp[SITE] = tmp[SITE].astype(str)
        if len(tmp) < 50:
            continue
        bic_null = _bic(tmp, outcome, COVARS)
        bic_full = _bic(tmp, outcome, [comp] + COVARS)
        bf10 = bf10_from_bic(bic_null, bic_full)
        rows.append({'predictor': comp, 'network': net, 'n': len(tmp),
                     'BF10': bf10, 'BF01': 1.0 / bf10 if bf10 > 0 else np.inf,
                     'log10_BF10': np.log10(bf10) if bf10 > 0 else np.nan,
                     'evidence': label_bf(bf10)})
biv = pd.DataFrame(rows)
biv.to_csv(TAB_DIR / 'supp_bayes_factors_bivariate.csv', index=False)
log('\n  composite → SCAN (headline):')
for _, r in biv[biv.network == 'SCAN'].iterrows():
    log(f'    {COMPOSITE_LABELS[r.predictor]:16s} BF10={r.BF10:.3g}  ({r.evidence})')

# ── (2) Multivariate joint model (SCAN): BF for each composite ────────────────
log('\n' + '=' * 74)
log('(2) MULTIVARIATE Bayes factors — 3 composites jointly → prop_SCAN')
log('=' * 74)
outcome = 'prop_SCAN'
need = [outcome] + COMPOSITE_COLS + COVARS + [SITE]
tmp = df[[c for c in need if c in df.columns]].dropna().copy()
tmp[SITE] = tmp[SITE].astype(str)
bic_full = _bic(tmp, outcome, COMPOSITE_COLS + COVARS)
rows = []
for comp in COMPOSITE_COLS:
    others = [c for c in COMPOSITE_COLS if c != comp]
    bic_drop = _bic(tmp, outcome, others + COVARS)     # null drops just this composite
    bf10 = bf10_from_bic(bic_drop, bic_full)
    rows.append({'predictor': comp, 'network': 'SCAN', 'n': len(tmp),
                 'BF10': bf10, 'BF01': 1.0 / bf10 if bf10 > 0 else np.inf,
                 'log10_BF10': np.log10(bf10) if bf10 > 0 else np.nan,
                 'evidence': label_bf(bf10)})
    log(f'  {COMPOSITE_LABELS[comp]:16s} BF10={bf10:.3g}  BF01={1/bf10:.3g}  ({label_bf(bf10)})')
mv = pd.DataFrame(rows)
mv.to_csv(TAB_DIR / 'supp_bayes_factors_multivariate_SCAN.csv', index=False)

with open(DAT_DIR / 'supp_bayes_factors.txt', 'w') as f:
    f.write('\n'.join(log_lines))
log('\nSaved: supp_bayes_factors_bivariate.csv, supp_bayes_factors_multivariate_SCAN.csv')
