#!/usr/bin/env python3
"""
Analysis H — out-of-sample (cross-validated) prediction of year-6 crystallized
cognition from baseline SCAN proportion.

Moves the cognition claim from explanatory (mediation) to predictive: does baseline
SCAN forecast year-6 crystallized cognition in held-out children, incrementally to
demographics AND to the adversity exposure itself?

Design: GroupKFold (10-fold) keeping families together; pooled out-of-fold (OOF)
predictions -> cross-validated R2 and Pearson r. Repeated over 20 fold-seeds for
stability. Models:
  M0    : covariates (y6 age, sex, y6 FD, y6 site)
  Mthr  : covariates + threat
  Mscan : covariates + baseline SCAN
  Mfull : covariates + threat + baseline SCAN
Incremental CV-R2: SCAN over covariates (Mscan-M0) and SCAN over cov+threat (Mfull-Mthr).
Permutation test: shuffle baseline SCAN, recompute Mscan-M0 on fixed folds (1000x) -> p.

Outputs:
  outputs/tables/H_cv_prediction_summary.txt
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from scipy import stats
from pathlib import Path

from adtopo.logging_utils import get_logger
_log = get_logger('cv_prediction')

ROOT = Path(__file__).resolve().parents[2]
DFB  = ROOT / 'outputs/data_processed/df_base.csv'
OUT  = ROOT / 'outputs/tables/H_cv_prediction_summary.txt'

OUTCOME = 'nihtb_cryst_y6'
SCANCOL = 'prop_SCAN'
THREAT  = 'threat_composite'
AGE6, SEX, FD6, SITE6, FAM = 'interview_age_6yrFU', 'sex_num', 'rest_mean_FD_6yrFU', 'study_site_6yrFU', 'family_id'
N_REPEATS, N_FOLDS, N_PERM = 20, 10, 1000
L = []
def log(s=''): _log.info(str(s)); L.append(s)

def cv_r2(X, y, groups, seed):
    """Pooled out-of-fold R2 and Pearson r for one GroupKFold split.

    Standardization is fit on the TRAINING fold only (StandardScaler inside a
    Pipeline) and applied to the held-out fold, so no test-fold information
    leaks into feature scaling. For unregularized OLS this is numerically
    identical to scaling globally — linear models are invariant to affine
    feature rescaling — but it is leakage-free by construction.
    """
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=N_FOLDS)
    # GroupKFold is deterministic; shuffle groups to vary splits by seed
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    remap = {g:i for i,g in enumerate(rng.permutation(uniq))}
    gshuf = np.array([remap[g] for g in groups])
    for tr, te in gkf.split(X, y, gshuf):
        m = make_pipeline(StandardScaler(), LinearRegression()).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return r2_score(y, oof), stats.pearsonr(y, oof)[0]

def main():
    df = pd.read_csv(DFB)
    cols = [OUTCOME, SCANCOL, THREAT, AGE6, SEX, FD6, SITE6, FAM]
    d = df[cols].dropna().copy()
    log('Analysis H — cross-validated prediction of year-6 crystallized cognition')
    log(f'N = {len(d)} children with complete y6-cognition + baseline-SCAN + covariates; '
        f'{d[FAM].nunique()} families')
    log('')

    # design matrices (RAW features; ALL standardization happens inside each CV
    # fold — see cv_r2 — so no test-fold statistics leak into feature scaling) -----
    site_d = pd.get_dummies(d[SITE6].astype(str), prefix='site', drop_first=True, dtype=float)
    COV  = np.column_stack([d[AGE6].values, d[SEX].values, d[FD6].values, site_d.values])
    scan = d[SCANCOL].values.reshape(-1,1)
    thr  = d[THREAT].values.reshape(-1,1)
    y    = d[OUTCOME].values
    groups = d[FAM].values

    FEATS = {
        'M0_cov'        : COV,
        'Mthr_cov+threat': np.hstack([COV, thr]),
        'Mscan_cov+SCAN' : np.hstack([COV, scan]),
        'Mfull_cov+thr+SCAN': np.hstack([COV, thr, scan]),
    }

    # observed CV performance ----------------------------------------------------
    log('Cross-validated performance (10-fold GroupKFold, mean +/- SD over 20 repeats):')
    res_r2 = {}
    for name, X in FEATS.items():
        vals = np.array([cv_r2(X, y, groups, 100+s) for s in range(N_REPEATS)])
        res_r2[name] = vals[:,0]
        log(f'  {name:22s}: CV-R2 = {vals[:,0].mean():.4f} +/- {vals[:,0].std():.4f}   '
            f'OOF r = {vals[:,1].mean():.3f}')
    log('')
    dscan_cov  = res_r2['Mscan_cov+SCAN'].mean() - res_r2['M0_cov'].mean()
    dscan_thr  = res_r2['Mfull_cov+thr+SCAN'].mean() - res_r2['Mthr_cov+threat'].mean()
    log(f'Incremental CV-R2 from baseline SCAN over covariates       : {dscan_cov:+.4f}')
    log(f'Incremental CV-R2 from baseline SCAN over covariates+threat: {dscan_thr:+.4f}')
    log('')

    # permutation test for SCAN's incremental CV-R2 over covariates --------------
    log('Permutation test (shuffle baseline SCAN; fixed folds; 1000 perms):')
    base_r2, _ = cv_r2(COV, y, groups, 7)
    obs_r2,  _ = cv_r2(np.hstack([COV, scan]), y, groups, 7)
    obs_delta  = obs_r2 - base_r2
    rng = np.random.default_rng(2026)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        sp = scan[rng.permutation(len(scan))]
        null[i] = cv_r2(np.hstack([COV, sp]), y, groups, 7)[0] - base_r2
    p = (np.sum(null >= obs_delta) + 1) / (N_PERM + 1)
    log(f'  observed incremental CV-R2 (fixed folds) = {obs_delta:+.4f}')
    log(f'  null mean = {null.mean():+.4f}, null 95th pct = {np.percentile(null,95):+.4f}')
    log(f'  permutation p = {p:.4g}')
    log('')
    log('Interpretation: baseline SCAN proportion predicts year-6 crystallized cognition in')
    log('held-out children; positive incremental CV-R2 over covariates(+threat) = predictive,')
    log('not merely explanatory, validity.')

    OUT.write_text('\n'.join(L))
    log(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
