"""
Phase 1b: Add composite scores, NIH Toolbox, and CBCL outcomes to processed dataframes.

Reads:
  TAB_DIR/ela_composites.csv   — threat/deprivation/unpredictability composites (baseline)
  DATA_DIR/ABCD_NIH_ToolBox/nc_y_nihtb.tsv
  DATA_DIR/cbcl/6_CBCL_allSubscales.csv

Updates (overwrites) in DAT_DIR:
  df_base.csv, df_y2.csv, df_y4.csv, df_y6.csv
"""
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import (
    TAB_DIR, DAT_DIR, DATA_DIR,
    NIH_TOOLBOX_FILE, NIH_FLUID_COL, NIH_CRYST_COL, NIH_COLS,
    CBCL_FILE, CBCL_MEDIATION_OUTCOMES, CBCL_SRC_COLUMNS,
    COMPOSITE_COLS,
    NIH_FLUID_Y6_COL, NIH_CRYST_Y6_COL,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg=''):
    print(msg, flush=True)

def count_nonnan(df, cols, label):
    for col in cols:
        if col in df.columns:
            n = df[col].notna().sum()
            log(f'  [{label}] {col}: N non-NaN = {n}')
        else:
            log(f'  [{label}] {col}: COLUMN MISSING')

# ── Load processed dataframes ─────────────────────────────────────────────────

log('=' * 70)
log('Loading processed dataframes ...')
log('=' * 70)

df_base = pd.read_csv(DAT_DIR / 'df_base.csv')
df_y2   = pd.read_csv(DAT_DIR / 'df_y2.csv')
df_y4   = pd.read_csv(DAT_DIR / 'df_y4.csv')
df_y6   = pd.read_csv(DAT_DIR / 'df_y6.csv')

log(f'  df_base N={len(df_base)}, df_y2 N={len(df_y2)}, '
    f'df_y4 N={len(df_y4)}, df_y6 N={len(df_y6)}')

# ── Step 1: Merge composite scores ───────────────────────────────────────────

log()
log('=' * 70)
log('STEP 1 — Merging ELA composite scores (baseline, applied to all timepoints)')
log('=' * 70)

comp_scores = pd.read_csv(TAB_DIR / 'ela_composites.csv')
log(f'  Composites loaded: N={len(comp_scores)}, columns={list(comp_scores.columns)}')

# Drop existing composite columns if present (avoid _x/_y conflicts)
for df in [df_base, df_y2, df_y4, df_y6]:
    for col in COMPOSITE_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

comp_merge = comp_scores[['sub_ID'] + COMPOSITE_COLS].copy()

df_base = df_base.merge(comp_merge, on='sub_ID', how='left', validate='many_to_one')
df_y2   = df_y2.merge(comp_merge,  on='sub_ID', how='left', validate='many_to_one')
df_y4   = df_y4.merge(comp_merge,  on='sub_ID', how='left', validate='many_to_one')
df_y6   = df_y6.merge(comp_merge,  on='sub_ID', how='left', validate='many_to_one')

log('  Composite scores merged. Non-NaN counts:')
for label, df in [('baseline', df_base), ('year-2', df_y2),
                  ('year-4',   df_y4),   ('year-6', df_y6)]:
    count_nonnan(df, COMPOSITE_COLS, label)

# ── Step 2: NIH Toolbox ───────────────────────────────────────────────────────

log()
log('=' * 70)
log('STEP 2 — NIH Toolbox fluid/crystallized scores')
log('=' * 70)

nih_raw = pd.read_csv(NIH_TOOLBOX_FILE, sep='\t', low_memory=False)
log(f'  NIH Toolbox loaded: N={len(nih_raw)} rows, columns include '
    f'participant_id, session_id')

# participant_id is already sub-XXXX format; session_id → timepoint mapping
session_map = {
    'ses-00A': '00A',
    'ses-02A': '02A',
    'ses-04A': '04A',
    'ses-06A': '06A',
}

nih_cols_needed = ['participant_id', 'session_id'] + [
    c for c in NIH_COLS if c in nih_raw.columns
]
missing_nih = [c for c in NIH_COLS if c not in nih_raw.columns]
if missing_nih:
    log(f'  WARNING: NIH columns not found in file: {missing_nih}')

nih = nih_raw[nih_cols_needed].copy()
nih = nih.rename(columns={'participant_id': 'sub_ID'})
nih['timepoint'] = nih['session_id'].map(session_map)
nih = nih.dropna(subset=['timepoint'])

# Merge per timepoint
tp_df_map = {
    '00A': df_base,
    '02A': df_y2,
    '04A': df_y4,
    '06A': df_y6,
}
tp_label = {
    '00A': 'baseline',
    '02A': 'year-2',
    '04A': 'year-4',
    '06A': 'year-6',
}

nih_avail_cols = [c for c in NIH_COLS if c in nih.columns]

for tp, df_ref in tp_df_map.items():
    nih_tp = nih[nih['timepoint'] == tp][['sub_ID'] + nih_avail_cols].copy()
    # Drop any pre-existing NIH columns
    for col in nih_avail_cols:
        if col in df_ref.columns:
            df_ref.drop(columns=[col], inplace=True)
    merged = df_ref.merge(nih_tp, on='sub_ID', how='left', validate='many_to_one')
    tp_df_map[tp] = merged

df_base = tp_df_map['00A']
df_y2   = tp_df_map['02A']
df_y4   = tp_df_map['04A']
df_y6   = tp_df_map['06A']

log('  NIH Toolbox merged. Non-NaN counts:')
for label_key, df in [('baseline', df_base), ('year-2', df_y2),
                       ('year-4',   df_y4),   ('year-6', df_y6)]:
    count_nonnan(df, nih_avail_cols, label_key)

# ── Step 2b: Add year-6 NIH fluid and crystallized to df_base for mediation ───
# NIH fluid is only available at baseline and year-6.
# Crystallized is available at all timepoints; we use year-6 for consistency.
# Both are merged into df_base for the ELA(baseline) → SCAN(baseline) → outcome(year-6) path.
log()
log('  Adding year-6 NIH fluid and year-6 crystallized to df_base for mediation...')


def merge_year6_outcome(df_base, nih, src_col, dst_col):
    """Left-merge one year-6 (ses-06A) NIH outcome into df_base as ``dst_col``.

    Selects the 06A rows, renames ``src_col`` -> ``dst_col``, drops NaNs, and
    drops any pre-existing ``dst_col`` first so the step is safely re-runnable.
    Returns the new df_base (merge produces a new frame).
    """
    y6 = nih[nih['timepoint'] == '06A'][['sub_ID', src_col]].copy()
    y6 = y6.rename(columns={src_col: dst_col}).dropna(subset=[dst_col])
    if dst_col in df_base.columns:
        df_base = df_base.drop(columns=[dst_col])
    return df_base.merge(y6, on='sub_ID', how='left', validate='many_to_one')


for _src, _dst in [(NIH_FLUID_COL, NIH_FLUID_Y6_COL), (NIH_CRYST_COL, NIH_CRYST_Y6_COL)]:
    df_base = merge_year6_outcome(df_base, nih, _src, _dst)
    log(f'    {_dst} in df_base: N={df_base[_dst].notna().sum()}')
tp_df_map['00A'] = df_base  # sync after step-2b reassignment

# ── Step 3: CBCL outcomes ─────────────────────────────────────────────────────

log()
log('=' * 70)
log('STEP 3 — CBCL outcome scores')
log('=' * 70)

cbcl_raw = pd.read_csv(CBCL_FILE, sep='\t', low_memory=False)
log(f'  CBCL loaded: N={len(cbcl_raw)} rows (ABCD mh_p_cbcl release)')
log(f'  CBCL sessions: {sorted(cbcl_raw["session_id"].dropna().unique().tolist())}')

# ABCD 6.0 mh_p_cbcl uses BIDS session codes (ses-0XA) and mh_p_cbcl__* summary
# columns. Rename the raw-sum columns to the canonical cbcl_scr_*_r names, then map
# session_id DIRECTLY to the imaging-wave code: ses-02A/04A/06A ARE the 2/4/6-year
# visits (verified mh_p_cbcl_age matches imaging age per wave — 9.96/12.07/14.18/
# 16.06). Off-year CBCL waves (01A/03A/05A/07A) have no paired imaging and are
# dropped. This is the correct alignment; the previous timepoint-string map was off
# by one wave (1_year→02A etc.).
cbcl_raw = cbcl_raw.rename(columns={'participant_id': 'sub_ID', **CBCL_SRC_COLUMNS})

cbcl_session_map = {'ses-00A': '00A', 'ses-02A': '02A', 'ses-04A': '04A', 'ses-06A': '06A'}

cbcl_avail_cols = [c for c in CBCL_MEDIATION_OUTCOMES if c in cbcl_raw.columns]
missing_cbcl = [c for c in CBCL_MEDIATION_OUTCOMES if c not in cbcl_raw.columns]
if missing_cbcl:
    log(f'  WARNING: CBCL columns not found after rename: {missing_cbcl}')

cbcl = cbcl_raw[['sub_ID', 'session_id'] + cbcl_avail_cols].copy()
cbcl['tp_code'] = cbcl['session_id'].map(cbcl_session_map)
cbcl = cbcl.dropna(subset=['tp_code'])

for tp, df_ref in tp_df_map.items():
    cbcl_tp = cbcl[cbcl['tp_code'] == tp][['sub_ID'] + cbcl_avail_cols].copy()
    # Drop any pre-existing CBCL columns
    for col in cbcl_avail_cols:
        if col in df_ref.columns:
            df_ref.drop(columns=[col], inplace=True)
    merged = df_ref.merge(cbcl_tp, on='sub_ID', how='left', validate='many_to_one')
    tp_df_map[tp] = merged

df_base = tp_df_map['00A']
df_y2   = tp_df_map['02A']
df_y4   = tp_df_map['04A']
df_y6   = tp_df_map['06A']

log('  CBCL merged. Non-NaN counts:')
for label_key, df in [('baseline', df_base), ('year-2', df_y2),
                       ('year-4',   df_y4),   ('year-6', df_y6)]:
    count_nonnan(df, cbcl_avail_cols, label_key)

# ── Step 4: Save updated dataframes ──────────────────────────────────────────

log()
log('=' * 70)
log('STEP 4 — Saving updated dataframes')
log('=' * 70)

df_base.to_csv(DAT_DIR / 'df_base.csv', index=False)
df_y2.to_csv(DAT_DIR   / 'df_y2.csv',   index=False)
df_y4.to_csv(DAT_DIR   / 'df_y4.csv',   index=False)
df_y6.to_csv(DAT_DIR   / 'df_y6.csv',   index=False)

log(f'  Saved df_base.csv  ({len(df_base)} rows, {len(df_base.columns)} cols)')
log(f'  Saved df_y2.csv    ({len(df_y2)} rows, {len(df_y2.columns)} cols)')
log(f'  Saved df_y4.csv    ({len(df_y4)} rows, {len(df_y4.columns)} cols)')
log(f'  Saved df_y6.csv    ({len(df_y6)} rows, {len(df_y6.columns)} cols)')

log()
log('Phase 1b complete.')
