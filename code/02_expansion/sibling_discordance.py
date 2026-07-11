#!/usr/bin/env python3
"""
Analysis B — sibling-discordance / within-family test of threat -> SCAN.

Goal: does the more threat-exposed sibling have the larger SCAN, holding constant
everything shared within a family (genetics partially, SES, neighborhood, parenting,
site)? If the WITHIN-family effect survives, the adversity->SCAN association is not
merely confounded by stable family characteristics.

FEASIBILITY first: family-size distribution and, crucially, within-family variance in
threat (twins/sibs who share near-identical exposure carry little within signal).

PRIMARY estimator — between/within (Mundlak/hybrid) decomposition:
  threat_fam = family mean of threat;  threat_dev = threat_i - threat_fam
  prop_SCAN ~ threat_fam + threat_dev + age + sex + fd + C(site) + (1|family_id)
  -> coefficient on threat_dev is the within-family (sibling-discordance) effect;
     coefficient on threat_fam is the between-family effect.

SENSITIVITY — discordant-sibling difference scores (families with exactly 2 sibs):
  dSCAN ~ dthreat + dage + dfd   (OLS; one independent obs per pair)

Outputs:
  outputs/tables/B_sibling_discordance_summary.txt
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy.linalg import LinAlgError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DFB  = ROOT / 'outputs/data_processed/df_base.csv'
OUT_TXT = ROOT / 'outputs/tables/B_sibling_discordance_summary.txt'

from adtopo.logging_utils import get_logger
_log = get_logger('sibling_discordance')

FD, SITE, FAM = 'fd', 'study_site', 'family_id'
L = []
def log(s=''):
    _log.info(str(s)); L.append(s)

def main():
    df = pd.read_csv(DFB)
    need = ['sub_ID', FAM, 'threat_composite', 'prop_SCAN', 'interview_age', 'sex_num', FD, SITE]
    d = df[need].dropna().copy()
    d['age'] = d['interview_age'] / 12.0
    log('Analysis B — sibling-discordance / within-family test of threat -> SCAN')
    log(f'N subjects (complete) = {len(d)}, N families = {d[FAM].nunique()}')
    log('')

    # ---- FEASIBILITY -----------------------------------------------------------
    sizes = d.groupby(FAM).size()
    multi = sizes[sizes >= 2]
    log('FEASIBILITY')
    log(f'  family-size distribution: ' +
        ', '.join(f'{k} sib x{v}' for k, v in sizes.value_counts().sort_index().items()))
    log(f'  families with >=2 in analytic sample: {len(multi)}  '
        f'(covering {int(multi.sum())} subjects)')

    # within-family threat SD (discordance) among multi-sib families
    g = d[d[FAM].isin(multi.index)].groupby(FAM)['threat_composite']
    within_sd = g.std(ddof=0)
    n_discordant = int((within_sd > 1e-9).sum())
    log(f'  multi-sib families with ANY within-family threat variance (discordant): {n_discordant}/{len(multi)}')
    log(f'  mean within-family threat SD (multi-sib): {within_sd.mean():.3f} '
        f'(threat is z-scored, so 1.0 = one full between-subject SD)')
    # fraction of total threat variance that is within-family
    d['threat_fam'] = d.groupby(FAM)['threat_composite'].transform('mean')
    d['threat_dev'] = d['threat_composite'] - d['threat_fam']
    var_within  = d['threat_dev'].var(ddof=0)
    var_total   = d['threat_composite'].var(ddof=0)
    log(f'  share of threat variance that is WITHIN-family (whole sample): {var_within/var_total:.3f}')
    log('')

    # ---- PRIMARY: Mundlak between/within ---------------------------------------
    log('PRIMARY — between/within (Mundlak) decomposition  (N = full sample)')
    d[SITE] = d[SITE].astype(str)
    MUNDLAK_FORMULA = 'prop_SCAN ~ threat_fam + threat_dev + age + sex_num + fd'
    res, meth = None, None
    # Only the expected singular-matrix failure (site variance component at its zero
    # boundary) is caught and logged; genuine convergence failure is treated
    # separately, and any other exception propagates so it gets investigated rather
    # than silently swallowed.
    try:
        res = smf.mixedlm(MUNDLAK_FORMULA, data=d, groups=d[FAM],
                          vc_formula={'site': f'0 + C({SITE})'}
                          ).fit(reml=True, method='lbfgs', maxiter=500)
        meth = 'mixedlm_crossed'
        if not res.converged:
            log('  NOTE: crossed site+family RE model did not converge; '
                'falling back to family-only RE.')
            res = None
    except LinAlgError as exc:
        log(f'  NOTE: crossed-RE fit hit an expected singular-matrix error '
            f'({type(exc).__name__}: {exc}); falling back to family-only RE.')
        res = None
    if res is None:
        res = smf.mixedlm(MUNDLAK_FORMULA, data=d, groups=d[FAM]
                          ).fit(reml=True, method='lbfgs', maxiter=500)
        meth = 'mixedlm_family_only'
    for term in ['threat_dev', 'threat_fam']:
        b, se, p = res.params[term], res.bse[term], res.pvalues[term]
        lab = 'WITHIN-family (sibling-discordance)' if term == 'threat_dev' else 'BETWEEN-family'
        log(f'  {lab:42s}: beta={b:+.6f}  se={se:.6f}  z={b/se:+.2f}  p={p:.4g}')
    log(f'  (method: {meth})')
    log(f'  reference: overall multivariate-adjusted threat->SCAN beta (full sample, phase3) = +0.00232')
    log('')

    # ---- SENSITIVITY: discordant-pair difference scores ------------------------
    log('SENSITIVITY — discordant-sibling difference scores (exactly-2-sib families)')
    two = sizes[sizes == 2].index
    pairs = []
    for fid, grp in d[d[FAM].isin(two)].groupby(FAM):
        a, b = grp.iloc[0], grp.iloc[1]
        pairs.append({'dSCAN': a['prop_SCAN'] - b['prop_SCAN'],
                      'dthreat': a['threat_composite'] - b['threat_composite'],
                      'dage': a['age'] - b['age'],
                      'dfd': a['fd'] - b['fd']})
    pp = pd.DataFrame(pairs)
    pp_disc = pp[pp['dthreat'].abs() > 1e-9]
    log(f'  N 2-sib families: {len(pp)};  threat-discordant pairs: {len(pp_disc)}')
    if len(pp_disc) >= 30:
        X = sm.add_constant(pp_disc[['dthreat', 'dage', 'dfd']].values)
        r = sm.OLS(pp_disc['dSCAN'].values, X).fit()
        log(f'  dSCAN ~ dthreat (+dage,+dfd): beta_dthreat={r.params[1]:+.6f}  '
            f'se={r.bse[1]:.6f}  t={r.tvalues[1]:+.2f}  p={r.pvalues[1]:.4g}  (N={len(pp_disc)})')
    else:
        log('  too few discordant pairs for a stable difference-score model.')

    # ---- CANONICAL: family fixed-effects (demeaned) OLS, cluster-robust SE ------
    # This is the gold-standard sibling-comparison estimator. For 2-sib families it
    # equals the difference-score; it also uses the 3-sib families. Cluster SE by family.
    log('')
    log('CANONICAL — family fixed-effects (within-demeaned) OLS, cluster-robust SE  (multi-sib only)')
    ms = d[d[FAM].isin(multi.index)].copy()
    fe_cols = ['prop_SCAN', 'threat_composite', 'age', 'sex_num', 'fd']
    for c in fe_cols:
        ms[c + '_dm'] = ms[c] - ms.groupby(FAM)[c].transform('mean')
    Xfe = ms[['threat_composite_dm', 'age_dm', 'fd_dm']].values  # no intercept (demeaned)
    yfe = ms['prop_SCAN_dm'].values
    rfe = sm.OLS(yfe, Xfe).fit(cov_type='cluster', cov_kwds={'groups': ms[FAM]})
    # SE note: inference uses the family-cluster-robust (sandwich) SE reported below
    # (rfe.bse), NOT the classical OLS SE. By the Frisch–Waugh–Lovell theorem the
    # within-demeaned slope equals the family-fixed-effects (LSDV) slope exactly; the
    # demeaning only distorts the *classical* residual-df (nobs - k), which the cluster
    # sandwich (whose small-sample correction scales with the number of families) does
    # not rely on. The demeaned-vs-LSDV equivalence is checked against these data in
    # code/tests/test_sibling_discordance.py.
    log(f'  N obs={len(ms)} from {ms[FAM].nunique()} multi-sib families')
    log(f'  threat (within) beta={rfe.params[0]:+.6f}  se={rfe.bse[0]:.6f}  '
        f'z={rfe.tvalues[0]:+.2f}  p={rfe.pvalues[0]:.4g}')
    log('')

    OUT_TXT.write_text('\n'.join(L))
    log('')
    log(f'wrote {OUT_TXT}')


if __name__ == '__main__':
    main()
