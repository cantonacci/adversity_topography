#!/usr/bin/env python3
"""
Analysis A — discovery/replication split-half of the SCAN selectivity headline.

Randomly split the baseline sample into two halves, keeping whole families together,
and independently re-estimate in each half:
  (1) per-network incremental ΔR2 for the 3 ELA composites  -> is SCAN the #1 network in both?
  (2) the multivariate threat->SCAN beta                     -> significant in both?
  (3) the full 15-network threat-beta profile and ΔR2 profile -> cross-half correlation

Modeling matches multivariate_models.py exactly (OLS with fixed site + family-cluster-
robust SE; incremental ΔR2 = R2_full - R2_covariates). Repeated over N_SPLITS random
seeds to show stability.

Outputs:
  outputs/tables/A_splithalf_per_split.csv
  outputs/tables/A_splithalf_summary.txt
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from adtopo.re_models import fit_ols_cluster_table
DFB  = ROOT / 'outputs/data_processed/df_base.csv'
OUT_CSV = ROOT / 'outputs/tables/A_splithalf_per_split.csv'
OUT_TXT = ROOT / 'outputs/tables/A_splithalf_summary.txt'

NETWORKS = ['DMN','VIS','FP','DAN','VAN','SAL','CO','SMD','SML','AUD','Tpole','MTL','PMN','PON','SCAN']
PREDS    = ['threat_composite','deprivation_composite','unpredictability_composite']
FD, SITE, FAM = 'fd', 'study_site', 'family_id'
N_SPLITS = 20
SEED0    = 20260620


def fit_scan_threat(df):
    """Multivariate threat beta/p for SCAN under the canonical reported spec
    (lib.re_models.fit_ols_cluster_table): all 3 composites + covariates + fixed
    site, OLS with family-cluster-robust SEs. Returns (beta, p, method, n)."""
    tbl, meta = fit_ols_cluster_table(
        df, 'prop_SCAN', PREDS, ['interview_age', 'sex_num', FD],
        site_col=SITE, family_col=FAM)
    if not meta['converged'] or tbl.empty:
        return np.nan, np.nan, 'skipped', meta['n']
    row = tbl[tbl['predictor'] == 'threat_composite'].iloc[0]
    return float(row['beta']), float(row['p']), meta['method'], meta['n']


def delta_r2(df, net):
    prop = f'prop_{net}'
    needed = [prop,'interview_age','sex_num',FD,SITE] + PREDS
    tmp = df[[c for c in needed if c in df.columns]].dropna()
    if len(tmp) < 50:
        return np.nan
    y = tmp[prop].values
    site_d = pd.get_dummies(tmp[SITE].astype(str), prefix='s', drop_first=True, dtype=float).values
    Xc = np.column_stack([np.ones(len(tmp)), tmp['interview_age'].values, tmp['sex_num'].values, tmp[FD].values, site_d])
    Xf = np.column_stack([np.ones(len(tmp)), tmp['interview_age'].values, tmp['sex_num'].values, tmp[FD].values,
                          tmp[PREDS].values, site_d])
    # Incremental (semipartial) ΔR² = R²_full − R²_covariates (share of TOTAL
    # variance uniquely attributable to the 3 ELA composites); matches
    # multivariate_models.compute_delta_r2.
    r2_cov  = sm.OLS(y, Xc).fit().rsquared
    r2_full = sm.OLS(y, Xf).fit().rsquared
    return r2_full - r2_cov


def threat_beta_profile(df):
    """threat beta per network (OLS, site dummies) -> 15-vector for profile correlation."""
    betas = {}
    for net in NETWORKS:
        prop = f'prop_{net}'
        needed = [prop,'interview_age','sex_num',FD,SITE] + PREDS
        tmp = df[[c for c in needed if c in df.columns]].dropna()
        site_d = pd.get_dummies(tmp[SITE].astype(str), prefix='s', drop_first=True, dtype=float)
        X = np.column_stack([np.ones(len(tmp)), tmp[PREDS].values, tmp['interview_age'].values,
                             tmp['sex_num'].values, tmp[FD].values, site_d.values])
        betas[net] = sm.OLS(tmp[prop].values, X).fit().params[1]  # threat is first pred
    return betas


df = pd.read_csv(DFB)
fams = df[FAM].dropna().unique()
print(f'N subjects = {len(df)}, N families = {len(fams)}')

rows = []
prof_corr_b, prof_corr_d = [], []
profile_rows = []          # per-network disc/rep ΔR² for every split (for Fig 1d)
scan_top_both = 0
scan_sig_both = 0

for s in range(N_SPLITS):
    rng = np.random.default_rng(SEED0 + s)
    perm = rng.permutation(fams)
    half1 = set(perm[:len(fams)//2])
    d1 = df[df[FAM].isin(half1)]
    d2 = df[~df[FAM].isin(half1)]

    res = {'split': s}
    dr = {}
    for tag, d in [('disc', d1), ('rep', d2)]:
        dr[tag] = {net: delta_r2(d, net) for net in NETWORKS}
        b, p, meth, n = fit_scan_threat(d)
        # SCAN rank by delta-R2 (1 = largest)
        ser = pd.Series(dr[tag]).dropna().sort_values(ascending=False)
        rank = list(ser.index).index('SCAN') + 1
        res[f'{tag}_scan_dr2']  = dr[tag]['SCAN']
        res[f'{tag}_scan_rank'] = rank
        res[f'{tag}_scan_beta'] = b
        res[f'{tag}_scan_p']    = p
        res[f'{tag}_n']         = n
    # profile correlations across halves
    bp1, bp2 = threat_beta_profile(d1), threat_beta_profile(d2)
    vb1 = np.array([bp1[n] for n in NETWORKS]); vb2 = np.array([bp2[n] for n in NETWORKS])
    vd1 = np.array([dr['disc'][n] for n in NETWORKS]); vd2 = np.array([dr['rep'][n] for n in NETWORKS])
    res['beta_profile_r'] = stats.pearsonr(vb1, vb2)[0]
    res['dr2_profile_r']  = stats.pearsonr(vd1, vd2)[0]
    prof_corr_b.append(res['beta_profile_r']); prof_corr_d.append(res['dr2_profile_r'])
    for net in NETWORKS:
        profile_rows.append({'split': s, 'network': net,
                             'disc_dr2': dr['disc'][net], 'rep_dr2': dr['rep'][net],
                             'dr2_profile_r': res['dr2_profile_r']})
    if res['disc_scan_rank'] == 1 and res['rep_scan_rank'] == 1:
        scan_top_both += 1
    if res['disc_scan_p'] < 0.05 and res['rep_scan_p'] < 0.05:
        scan_sig_both += 1
    rows.append(res)
    print(f"split {s}: SCAN rank disc={res['disc_scan_rank']} rep={res['rep_scan_rank']}; "
          f"beta disc={res['disc_scan_beta']:.5f}(p={res['disc_scan_p']:.1e}) "
          f"rep={res['rep_scan_beta']:.5f}(p={res['rep_scan_p']:.1e}); "
          f"profile r(beta)={res['beta_profile_r']:.2f}")

tab = pd.DataFrame(rows)
tab.to_csv(OUT_CSV, index=False)

# Per-network disc/rep ΔR² profiles: all splits + a representative split for Fig 1d.
prof = pd.DataFrame(profile_rows)
prof.to_csv(ROOT / 'outputs/tables/A_splithalf_profiles_allsplits.csv', index=False)
mean_r  = tab['dr2_profile_r'].mean()
rep_split = int((tab['dr2_profile_r'] - mean_r).abs().idxmin())   # split closest to mean r
prof[prof['split'] == rep_split].to_csv(
    ROOT / 'outputs/tables/A_splithalf_profile_repsplit.csv', index=False)
print(f'Representative split for Fig 1d = {rep_split} '
      f'(dr2_profile_r={tab.loc[rep_split, "dr2_profile_r"]:.3f}, mean={mean_r:.3f})')

L = []
L.append('Analysis A — discovery/replication split-half (baseline, family-respecting)')
L.append(f'N subjects = {len(df)}, N families = {len(fams)}, splits = {N_SPLITS}')
L.append('')
L.append(f'SCAN is the #1 delta-R2 network in BOTH halves: {scan_top_both}/{N_SPLITS} splits')
L.append(f'threat->SCAN beta significant (p<.05) in BOTH halves: {scan_sig_both}/{N_SPLITS} splits')
L.append('')
L.append(f'SCAN delta-R2  : disc {tab.disc_scan_dr2.mean():.4f} +/- {tab.disc_scan_dr2.std():.4f} | '
         f'rep {tab.rep_scan_dr2.mean():.4f} +/- {tab.rep_scan_dr2.std():.4f}')
L.append(f'threat->SCAN b : disc {tab.disc_scan_beta.mean():.5f} | rep {tab.rep_scan_beta.mean():.5f}')
L.append(f'  disc p range : {tab.disc_scan_p.min():.1e} .. {tab.disc_scan_p.max():.1e}')
L.append(f'  rep  p range : {tab.rep_scan_p.min():.1e} .. {tab.rep_scan_p.max():.1e}')
L.append('')
L.append(f'Cross-half threat-beta PROFILE correlation (15 nets): '
         f'mean r = {np.mean(prof_corr_b):.3f} (range {np.min(prof_corr_b):.3f}..{np.max(prof_corr_b):.3f})')
L.append(f'Cross-half delta-R2 PROFILE correlation (15 nets): '
         f'mean r = {np.mean(prof_corr_d):.3f} (range {np.min(prof_corr_d):.3f}..{np.max(prof_corr_d):.3f})')
txt = '\n'.join(L)
OUT_TXT.write_text(txt)
print('\n' + txt)
