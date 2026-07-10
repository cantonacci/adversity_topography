"""
Supplement: Bayes factors for the INDIVIDUAL-factor bivariate associations
(revision request — Russ: report Bayes factors alongside the headline bivariate
r/q values, not only for the composites).

Same BIC-approximation Bayes factor used for the composite associations
(code/supplement/bayes_factors_expansion.py), applied here to each of the 10
data-driven ELA factors. For a given factor -> prop_NET, the full model is
    prop_NET ~ factor + age + sex + FD + C(site)
and the null drops just that factor (all covariates + fixed site retained), so
the BF quantifies exactly the partial association reported as the partial-r/q in
the main text. BF10 = exp((BIC_null - BIC_full) / 2); BF10 > 1 favors an
association (H1), BF10 < 1 favors the null (report BF01 = 1/BF10). Kass & Raftery
(1995): >3 positive, >20 strong, >150 very strong.

Outputs:
  TAB_DIR/supp_bayes_factors_individual.csv   (each factor x each network)
  DAT_DIR/supp_bayes_factors_individual.txt
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

_CODE = next(a for a in Path(__file__).resolve().parents if (a / 'config.py').exists())
sys.path.insert(0, str(_CODE))
from config import TAB_DIR, DAT_DIR, NETWORKS, ELA_COLS, ELA_LABELS_SHORT
from lib.bayes_factors import _bic, bf10_from_bic, label_bf

COVARS = ['interview_age', 'sex_num', 'fd']
SITE = 'study_site'

log_lines = []
def log(m=''):
    print(m, flush=True); log_lines.append(str(m))


df = pd.read_csv(DAT_DIR / 'df_base.csv')
log(f'Loaded df_base N={len(df)}\n')

log('=' * 74)
log('BIVARIATE Bayes factors — individual ELA factor -> prop_NET (one at a time)')
log('=' * 74)
rows = []
for fac in ELA_COLS:
    for net in NETWORKS:
        outcome = f'prop_{net}'
        need = [outcome, fac] + COVARS + [SITE]
        tmp = df[[c for c in need if c in df.columns]].dropna().copy()
        tmp[SITE] = tmp[SITE].astype(str)
        if len(tmp) < 50:
            continue
        bic_null = _bic(tmp, outcome, COVARS)
        bic_full = _bic(tmp, outcome, [fac] + COVARS)
        bf10 = bf10_from_bic(bic_null, bic_full)
        rows.append({'predictor': fac, 'network': net, 'n': len(tmp),
                     'BF10': bf10, 'BF01': 1.0 / bf10 if bf10 > 0 else np.inf,
                     'log10_BF10': np.log10(bf10) if bf10 > 0 else np.nan,
                     'evidence': label_bf(bf10)})
out = pd.DataFrame(rows)
out.to_csv(TAB_DIR / 'supp_bayes_factors_individual.csv', index=False)

log('\n  factor -> SCAN (headline bivariate effects):')
scan = out[out.network == 'SCAN'].copy()
scan['absBF'] = scan['BF10'].where(scan['BF10'] >= 1, 1.0 / scan['BF10'])
for _, r in scan.sort_values('BF10', ascending=False).iterrows():
    bf_str = (f'BF10={r.BF10:.3g}' if r.BF10 >= 1 else f'BF01={1/r.BF10:.3g}')
    log(f'    {ELA_LABELS_SHORT[r.predictor]:22s} {bf_str:16s} ({r.evidence})')

with open(DAT_DIR / 'supp_bayes_factors_individual.txt', 'w') as f:
    f.write('\n'.join(log_lines))
log('\nSaved: supp_bayes_factors_individual.csv')
