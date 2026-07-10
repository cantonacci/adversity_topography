#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fc_analysis.py  —  Merge FC chunks and run LME: FC ~ threat_composite + covariates.

Steps:
  1. Merge fc_chunks/fc_chunk_*.csv -> fc_ses-00A.csv
  2. Join with df_base.csv on sub_ID
  3. LME for each of 105 between-network FC pairs (off-diagonal upper triangle)
     model: FC ~ threat_composite + interview_age + sex_num + mean_FD +
                  n_usable_frames + C(study_site) + (1|family_id)
     with fallbacks: family-only RE -> OLS+site dummies (mirrors phase3)
  4. Dual FDR correction:
       q_fdr_all  : BH across all 105 pairs
       q_fdr_scan : BH across the 14 SCAN-specific pairs (threat only)
  5. Save outputs/tables/fc_lme_threat_baseline.csv

Usage:
  python fc_analysis.py [--skip-merge]
"""

import sys, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import DAT_DIR, TAB_DIR, NETWORKS
from lib.re_models import fit_ols_cluster_table

CHUNK_DIR = DAT_DIR / 'fc_chunks'
FC_PATH   = DAT_DIR / 'fc_ses-00A.csv'
OUT_DIR   = TAB_DIR                       # repo outputs/tables (was code/outputs/tables)
OUT_PATH  = OUT_DIR / 'fc_lme_threat_baseline.csv'

NET_NAMES = list(NETWORKS)
N         = len(NET_NAMES)

# 105 off-diagonal (between-network) pairs
PAIRS = [(NET_NAMES[i], NET_NAMES[j]) for i in range(N) for j in range(i + 1, N)]
FC_COLS = [f'fc_{n1}_{n2}' for n1, n2 in PAIRS]

COVARIATES = ['interview_age', 'sex_num', 'mean_FD', 'n_usable_frames']
PRED       = 'threat_composite'


def log(msg): print(msg, flush=True)


# ── Step 1: merge chunks ───────────────────────────────────────────────────────

def merge_chunks():
    files = sorted(CHUNK_DIR.glob('fc_chunk_*.csv'))
    if not files:
        raise FileNotFoundError(f'No fc_chunk_*.csv files in {CHUNK_DIR}')
    log(f'Merging {len(files)} chunks ...')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset='sub_ID')
    df.to_csv(FC_PATH, index=False)
    log(f'  Merged N={len(df)}  →  {FC_PATH}')
    return df


# ── Step 2: join with covariates ───────────────────────────────────────────────

def load_merged():
    df_fc   = pd.read_csv(FC_PATH)
    df_base = pd.read_csv(DAT_DIR / 'df_base.csv')

    keep_base = ['sub_ID', 'family_id', 'study_site', PRED,
                 'interview_age', 'sex_num']
    df_base = df_base[[c for c in keep_base if c in df_base.columns]].copy()

    if 'study_site' not in df_base.columns and 'study_site_baseline' in df_base.columns:
        df_base['study_site'] = df_base['study_site_baseline']

    df = df_fc.merge(df_base, on='sub_ID', how='inner')
    log(f'Merged FC + covariates: N={len(df)}')
    return df


# ── Step 3: per-pair association ──────────────────────────────────────────────

def fit_one(df, fc_col, site_col='study_site', family_col='family_id'):
    """Canonical reported spec (lib.re_models.fit_ols_cluster_table):
    FC ~ threat_composite + covariates + C(site), OLS with family-cluster-robust
    SEs. The 't' field holds the cluster-robust z."""
    tbl, meta = fit_ols_cluster_table(
        df, fc_col, [PRED], COVARIATES, site_col=site_col, family_col=family_col)
    if not meta['converged'] or tbl.empty:
        return dict(n=meta['n'], method='insufficient_data',
                    beta=np.nan, se=np.nan, t=np.nan, p=np.nan)
    r = tbl.iloc[0]
    return dict(n=meta['n'], method=meta['method'],
                beta=float(r['beta']), se=float(r['se']),
                t=float(r['z']), p=float(r['p']))


def run_lme(df):
    site_col = 'study_site'
    results  = []

    for k, (n1, n2) in enumerate(PAIRS):
        fc_col = f'fc_{n1}_{n2}'
        if k % 10 == 0:
            log(f'  [{k:3d}/{len(PAIRS)}] {fc_col}')

        r = fit_one(df, fc_col, site_col)
        r.update({'net1': n1, 'net2': n2, 'fc_col': fc_col})
        results.append(r)

    return pd.DataFrame(results)


# ── Step 4: dual FDR ──────────────────────────────────────────────────────────

def apply_fdr(df_res):
    p = df_res['p'].values
    valid = ~np.isnan(p)

    # All 105 pairs
    q_all = np.full(len(p), np.nan)
    if valid.sum() > 0:
        _, q_corr, _, _ = multipletests(p[valid], method='fdr_bh')
        q_all[valid] = q_corr
    df_res['q_fdr_all'] = q_all

    # SCAN-specific 14 pairs
    scan_mask = (df_res['net1'] == 'SCAN') | (df_res['net2'] == 'SCAN')
    q_scan = np.full(len(p), np.nan)
    sv   = scan_mask.values & valid
    if sv.sum() > 0:
        _, q_corr_s, _, _ = multipletests(p[sv], method='fdr_bh')
        q_scan[sv] = q_corr_s
    df_res['q_fdr_scan'] = q_scan

    df_res['sig_all']  = df_res['q_fdr_all']  < 0.05
    df_res['sig_scan'] = df_res['q_fdr_scan'] < 0.05

    return df_res


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-merge', action='store_true',
                        help='Skip chunk merging (fc_ses-00A.csv already exists)')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(DAT_DIR).mkdir(parents=True, exist_ok=True)

    log('=' * 60)
    log('FC ANALYSIS  (threat_composite → FC, LME)')
    log('=' * 60)

    if not args.skip_merge or not FC_PATH.exists():
        merge_chunks()

    df = load_merged()

    log(f'\nRunning LME for {len(PAIRS)} FC pairs ...')
    df_res = run_lme(df)
    df_res = apply_fdr(df_res)

    # Order columns
    col_order = ['net1', 'net2', 'fc_col', 'n', 'method',
                 'beta', 'se', 't', 'p', 'q_fdr_all', 'q_fdr_scan',
                 'sig_all', 'sig_scan']
    df_res = df_res[[c for c in col_order if c in df_res.columns]]
    df_res.to_csv(OUT_PATH, index=False)

    # Summary
    n_sig_all  = df_res['sig_all'].sum()
    n_sig_scan = df_res['sig_scan'].sum()
    log(f'\nFDR significant (all 105 pairs):          {n_sig_all}')
    log(f'FDR significant (SCAN-specific 14 pairs): {n_sig_scan}')
    log(f'\nSaved → {OUT_PATH}')

    log('\nTop results (SCAN-specific, sorted by |beta|):')
    scan_res = df_res[df_res['net1'].eq('SCAN') | df_res['net2'].eq('SCAN')].copy()
    scan_res = scan_res.sort_values('p')
    for _, row in scan_res.head(10).iterrows():
        sig = '*' if row['sig_scan'] else ' '
        log(f"  {sig} {row['net1']:6s}-{row['net2']:6s}  β={row['beta']:+.4f}  "
            f"t={row['t']:+.2f}  p={row['p']:.4f}  q_scan={row['q_fdr_scan']:.4f}")

    log('\nDone.')


if __name__ == '__main__':
    main()
