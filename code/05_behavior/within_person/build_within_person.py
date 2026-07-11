#!/usr/bin/env python3
"""
Adapted version of build_topo.py and build_ela.py.

Key difference from colleague's scripts:
  - ELA composite uses our paper's 4-item threat_composite (pre-computed in df_base)
    rather than the 2-item (family_conflict + physical_trauma) the colleague used.
    This ensures consistency with all other analyses in the paper.
  - SCAN/FD/age loaded from our processed per-wave CSVs rather than raw files.
  - NIH Toolbox loaded from nc_y_nihtb.tsv (same source, wave-specific scores).

Outputs:
  derived/scan_topo_long.csv   -- long-format per-subject x wave dataset
  derived/ela_scores.csv       -- subject-level threat composite
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE   = Path(__file__).resolve().parents[3]
DAT    = BASE / 'outputs' / 'data_processed'
DATA   = BASE / 'data'
OUT    = Path(__file__).parent / 'derived'

WAVES = ['00A', '02A', '04A', '06A']
YEARS = {'00A': 0, '02A': 2, '04A': 4, '06A': 6}
FD_THRESH = 0.20

WAVE_FILES = {
    '00A': DAT / 'df_base.csv',
    '02A': DAT / 'df_y2.csv',
    '04A': DAT / 'df_y4.csv',
    '06A': DAT / 'df_y6.csv',
}

def main():
    print('=' * 68)
    print('Building within-person dataset')
    print('=' * 68)

    # ── 1. SCAN proportion + FD + age per wave ────────────────────────────────────
    rows = []
    for w, fpath in WAVE_FILES.items():
        df = pd.read_csv(fpath, usecols=lambda c: c in
                         ['sub_ID', 'prop_SCAN', 'fd', 'interview_age', 'sex_num'])
        df = df.rename(columns={
            'sub_ID': 'src_subject_id',
            'prop_SCAN': 'scan_prop',
            'fd': 'mean_FD',
            'interview_age': 'age',
            'sex_num': 'sex',
        })
        df['wave']  = w
        df['years'] = YEARS[w]
        rows.append(df)

    long = pd.concat(rows, ignore_index=True)
    print(f'Long (SCAN+FD+age): {len(long)} rows, {long["src_subject_id"].nunique()} subjects')

    # ── 2. family_id from df_base ─────────────────────────────────────────────────
    base = pd.read_csv(DAT / 'df_base.csv', usecols=['sub_ID', 'family_id'])
    long = long.merge(
        base.rename(columns={'sub_ID': 'src_subject_id'}),
        on='src_subject_id', how='left')

    # ── 3. NIH Toolbox cognition (wave-specific) ──────────────────────────────────
    nih = pd.read_csv(
        DATA / 'ABCD_NIH_ToolBox' / 'nc_y_nihtb.tsv', sep='\t',
        usecols=['participant_id', 'session_id',
                 'nc_y_nihtb__comp__fluid__agecorr_score',
                 'nc_y_nihtb__comp__cryst__agecorr_score'],
        dtype=str)
    nih['wave']      = nih['session_id'].str.replace('ses-', '', regex=False)
    nih              = nih[nih['wave'].isin(WAVES)].copy()
    nih['cog_fluid'] = pd.to_numeric(nih['nc_y_nihtb__comp__fluid__agecorr_score'], errors='coerce')
    nih['cog_cryst'] = pd.to_numeric(nih['nc_y_nihtb__comp__cryst__agecorr_score'], errors='coerce')
    long = long.merge(
        nih.rename(columns={'participant_id': 'src_subject_id'})[
            ['src_subject_id', 'wave', 'cog_fluid', 'cog_cryst']],
        on=['src_subject_id', 'wave'], how='left')

    # ── 4. QC ─────────────────────────────────────────────────────────────────────
    long['fd_ok']        = long['mean_FD'] < FD_THRESH
    long['topo_present'] = long['scan_prop'].notna()
    long['usable']       = long['fd_ok'] & long['topo_present']
    long = long.sort_values(['src_subject_id', 'years']).reset_index(drop=True)

    print(f'\nQC: mean_FD < {FD_THRESH} AND scan_prop present')
    print(f'Total rows: {len(long)},  usable: {long["usable"].sum()} ({100*long["usable"].mean():.1f}%)')
    for w in WAVES:
        s = long[long.wave == w]
        print(f'  ses-{w}: {int(s["usable"].sum())}/{len(s)} usable')

    uw = long[long.usable].groupby('src_subject_id').size()
    print('\nUsable-wave-count (post-QC):')
    for k in [1, 2, 3, 4]:
        print(f'  {k} wave(s): {int((uw == k).sum())}')
    print(f'  >=2 usable: {int((uw >= 2).sum())}  (needed for first-last analysis)')

    print('\nSCAN proportion by wave (usable): mean (SD)')
    for w in WAVES:
        v = long[(long.wave == w) & long.usable]['scan_prop']
        print(f'  ses-{w}: {v.mean():.4f} ({v.std():.4f})  n={v.notna().sum()}')

    long.to_csv(OUT / 'scan_topo_long.csv', index=False)
    print(f'\nSaved → {OUT}/scan_topo_long.csv')

    # ── 5. ELA threat composite (4-item, from df_base — consistent with paper) ────
    base_ela = pd.read_csv(DAT / 'df_base.csv',
                           usecols=['sub_ID', 'threat_composite'])
    ela_out = base_ela.rename(columns={
        'sub_ID': 'src_subject_id',
        'threat_composite': 'ela_threat',
    }).dropna()
    ela_out.to_csv(OUT / 'ela_scores.csv', index=False)
    print(f'Saved → {OUT}/ela_scores.csv  (n={len(ela_out)})')
    print('\nNOTE: ela_threat uses our paper\'s 4-item threat_composite.')
    print('      Colleague used 2-item (family_conflict + physical_trauma).')
    print('      Results may differ slightly.')
    print('\nDone.')


if __name__ == '__main__':
    main()
