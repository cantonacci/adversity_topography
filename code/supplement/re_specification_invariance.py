"""
Supplement: random-effects specification invariance.
adversity_topography/code/supplement/re_specification_invariance.py

Reviewer questions (Andre): why is study site a fixed effect rather than a
random effect; should scanner be nested; should family be nested within site?
This refits the headline SCAN-expansion associations under a ladder of variance
structures and shows the fixed-effect estimate is invariant to the choice.

Runs at BASELINE (the reported timepoint for expansion effects):
  (1) Bivariate: each ELA composite -> prop_SCAN, one predictor at a time
      (mirrors code/02_expansion/bivariate_associations.py).
  (2) Multivariate: all three composites -> prop_SCAN simultaneously
      (mirrors code/02_expansion/multivariate_models.py).

For each, every specification in re_models.SPECS is fitted. The currently
reported value is embedded so the old-vs-new delta is explicit.

Outputs:
  TAB_DIR/supp_re_invariance_bivariate.csv
  TAB_DIR/supp_re_invariance_multivariate.csv
  DAT_DIR/supp_re_invariance.txt   (human-readable preview)
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

_CODE = next(a for a in Path(__file__).resolve().parents if (a / 'config.py').exists())
sys.path.insert(0, str(_CODE))
from config import TAB_DIR, DAT_DIR, COMPOSITE_COLS, COMPOSITE_LABELS
from lib.re_models import fit_spec, SPECS, SPEC_LABELS

COVARIATES = ['interview_age', 'sex_num', 'fd']
OUTCOME = 'prop_SCAN'

# Currently reported baseline values (for the explicit old-vs-new delta).
REPORTED_BIVARIATE_PARTIAL_R = {         # code/02_expansion bivariate, baseline
    'threat_composite':          0.1578,
    'unpredictability_composite': 0.0979,
    'deprivation_composite':      0.0900,
}
REPORTED_MULTIVARIATE_BETA = {           # code/02_expansion multivariate, baseline
    'threat_composite': 0.0023,          # deprivation/unpred not separately reported
}

# The specification currently used to produce each reported number, for context.
REPORTED_SPEC = {'bivariate': 're_family', 'multivariate': 're_crossed'}

log_lines = []
def log(msg=''):
    print(msg, flush=True)
    log_lines.append(str(msg))


df = pd.read_csv(DAT_DIR / 'df_base.csv')
log(f'Loaded df_base N={len(df)}')
log(f'Outcome={OUTCOME}; covariates={COVARIATES}; site=study_site fixed/random per spec\n')


def run_block(kind, predictors, targets):
    """kind in {'bivariate','multivariate'}. Returns tidy DataFrame."""
    rows = []
    for tgt in targets:
        preds = [tgt] if kind == 'bivariate' else predictors
        for spec in SPECS:
            rec = fit_spec(df, OUTCOME, tgt, preds, COVARIATES, spec)
            rec['kind'] = kind
            if kind == 'bivariate':
                rep = REPORTED_BIVARIATE_PARTIAL_R.get(tgt, np.nan)
                rec['reported_partial_r'] = rep
                rec['delta_partial_r'] = (rec['partial_r'] - rep
                                          if not np.isnan(rep) else np.nan)
            else:
                rep = REPORTED_MULTIVARIATE_BETA.get(tgt, np.nan)
                rec['reported_beta'] = rep
                rec['delta_beta'] = (rec['beta'] - rep if not np.isnan(rep) else np.nan)
            rows.append(rec)
    return pd.DataFrame(rows)


# ── (1) Bivariate ─────────────────────────────────────────────────────────────
log('=' * 70)
log('(1) BIVARIATE  —  each composite -> prop_SCAN (one predictor at a time)')
log('=' * 70)
biv = run_block('bivariate', COMPOSITE_COLS, COMPOSITE_COLS)
biv.to_csv(TAB_DIR / 'supp_re_invariance_bivariate.csv', index=False)

for tgt in COMPOSITE_COLS:
    sub = biv[biv['target'] == tgt]
    rep = REPORTED_BIVARIATE_PARTIAL_R.get(tgt, np.nan)
    log(f'\n  {COMPOSITE_LABELS[tgt]} -> SCAN   (reported partial r = {rep:.4f}, '
        f'reported spec = {REPORTED_SPEC["bivariate"]})')
    for _, r in sub.iterrows():
        flag = '  <-- reported spec' if r['spec'] == REPORTED_SPEC['bivariate'] else ''
        log(f'    {r["spec"]:24s} partial_r={r["partial_r"]:+.4f}  '
            f'beta={r["beta"]:+.5f}  p={r["p"]:.2e}  conv={r["converged"]}{flag}')

# ── (2) Multivariate ──────────────────────────────────────────────────────────
log('\n' + '=' * 70)
log('(2) MULTIVARIATE  —  all three composites -> prop_SCAN simultaneously')
log('=' * 70)
mv = run_block('multivariate', COMPOSITE_COLS, COMPOSITE_COLS)
mv.to_csv(TAB_DIR / 'supp_re_invariance_multivariate.csv', index=False)

for tgt in COMPOSITE_COLS:
    sub = mv[mv['target'] == tgt]
    rep = REPORTED_MULTIVARIATE_BETA.get(tgt, np.nan)
    rep_s = f'{rep:.4f}' if not np.isnan(rep) else 'n/a'
    log(f'\n  {COMPOSITE_LABELS[tgt]} (adjusted) -> SCAN   (reported beta = {rep_s}, '
        f'reported spec = {REPORTED_SPEC["multivariate"]})')
    for _, r in sub.iterrows():
        flag = '  <-- reported spec' if r['spec'] == REPORTED_SPEC['multivariate'] else ''
        log(f'    {r["spec"]:24s} beta={r["beta"]:+.5f}  se={r["se"]:.5f}  '
            f'p={r["p"]:.2e}  conv={r["converged"]}  [{r["var_components"]}]{flag}')

# ── Compact invariance summary for the target (threat) ────────────────────────
log('\n' + '=' * 70)
log('INVARIANCE SUMMARY (threat -> SCAN, the headline effect)')
log('=' * 70)
for kind, tbl, val, rep in [
    ('bivariate', biv, 'partial_r', REPORTED_BIVARIATE_PARTIAL_R['threat_composite']),
    ('multivariate', mv, 'beta', REPORTED_MULTIVARIATE_BETA['threat_composite']),
]:
    s = tbl[tbl['target'] == 'threat_composite'][val].astype(float)
    log(f'  {kind:13s}: {val} range [{s.min():+.5f}, {s.max():+.5f}], '
        f'max abs deviation from reported ({rep:+.5f}) = {np.nanmax(np.abs(s - rep)):.5f}')

with open(DAT_DIR / 'supp_re_invariance.txt', 'w') as f:
    f.write('\n'.join(log_lines))
log('\nSaved: supp_re_invariance_bivariate.csv, supp_re_invariance_multivariate.csv, '
    'supp_re_invariance.txt')
