"""
SCAN vertex-level density maps: high vs. low adversity groups.

For each subject, the boldmap.dlabel.nii assigns every cortical vertex to a
network. This script computes, for each vertex, the fraction of subjects in a
given adversity group who have that vertex assigned to SCAN — giving a spatial
probability map of SCAN territory.

Splitting variable options (set SPLIT_VAR below):
  'threat_composite'         — strongest SCAN predictor, most interpretable
  'deprivation_composite'
  'mean_composite'           — mean of all 3 composites (global adversity)
  'SCAN'                     — split directly on SCAN topography as sanity check

Outputs (outputs/cifti_for_workbench/):
  SCAN_density_high_{group}_baseline.dscalar.nii   — P(SCAN) for high adversity
  SCAN_density_low_{group}_baseline.dscalar.nii    — P(SCAN) for low adversity
  SCAN_density_diff_{group}_baseline.dscalar.nii   — high minus low
  SCAN_density_all3_{group}_baseline.dscalar.nii   — 3-map file (high, low, diff)

Load in Connectome Workbench:
  The diff map will show regions where SCAN is expanded in high-adversity youth.
  Palette: ROY-BIG-BL (diverging), range ±0.15 or auto; Display zero: OFF
  The high/low maps: palette "hot" or videen_style, range 0–0.20
"""

import sys, os, glob, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import ATLAS_DIR, REPRO_DIR, DAT_DIR

# ── Configuration ──────────────────────────────────────────────────────────────
TIMEPOINT       = 'baseline'          # 'baseline', 'year2', 'year4', 'year6'
SESSION         = '00A'               # BIDS session label matching TIMEPOINT

# Splitting variable: what defines "high adversity"
SPLIT_VAR       = 'threat_composite'  # see docstring for options

# Group thresholds (in SD units; z-scores were computed within the analysis sample)
HIGH_THRESH     =  1.0   # subjects with SPLIT_VAR >= HIGH_THRESH are "high adversity"
LOW_THRESH      = -1.0   # subjects with SPLIT_VAR <= LOW_THRESH  are "low adversity"

SCAN_LABEL      = 18
N_JOBS          = min(16, cpu_count())

PROC_DIR  = DAT_DIR
OUT_DIR   = DAT_DIR.parent / 'cifti_for_workbench'
ATLAS_PATH = ATLAS_DIR / 'abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii'

SESSION_MAP = {'baseline': '00A', 'year2': '02A', 'year4': '04A', 'year6': '06A'}
DF_MAP      = {'baseline': 'df_base.csv', 'year2': 'df_year2.csv',
               'year4': 'df_year4.csv',   'year6': 'df_year6.csv'}

BOLDMAP_GLOB = (
    '*_task-rest_space-fsLR_den-91k_desc-denoised-spatially-interpolated-'
    'smoothed-2.25mm-censor-ReproTM_template-ABCC-a3-9to16_refine-SCAN_'
    'minsize-30_boldmap.dlabel.nii'
)

def log(msg=''):
    print(msg, flush=True)


# ── Atlas: get BrainModelAxis and cortical structure ───────────────────────────

def load_atlas_bm():
    img = nib.load(str(ATLAS_PATH))
    bm  = img.header.get_axis(1)
    # Identify cortical grayordinate ranges
    structs = {name: slc for name, slc, _ in bm.iter_structures()}
    l_slc = structs.get('CIFTI_STRUCTURE_CORTEX_LEFT')
    r_slc = structs.get('CIFTI_STRUCTURE_CORTEX_RIGHT')
    return bm, l_slc, r_slc


# ── Find boldmap files ─────────────────────────────────────────────────────────

def find_boldmap(sub_id, session):
    pattern = str(REPRO_DIR / sub_id / f'ses-{session}' / 'func' / BOLDMAP_GLOB)
    hits = glob.glob(pattern)
    return hits[0] if hits else None


# ── Per-subject worker ─────────────────────────────────────────────────────────

def load_scan_mask(args):
    """Return per-vertex binary SCAN mask (91282,) or None on failure."""
    sub_id, session = args
    fpath = find_boldmap(sub_id, session)
    if fpath is None:
        return None
    try:
        data = nib.load(fpath).get_fdata()[0]   # (91282,)
        return (data == SCAN_LABEL).astype(np.float32)
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log('=' * 70)
    log('SCAN DENSITY MAPS — high vs. low adversity')
    log(f'  Timepoint  : {TIMEPOINT} (ses-{SESSION_MAP[TIMEPOINT]})')
    log(f'  Split var  : {SPLIT_VAR}')
    log(f'  Thresholds : high ≥ {HIGH_THRESH} SD  |  low ≤ {LOW_THRESH} SD')
    log(f'  N workers  : {N_JOBS}')
    log('=' * 70)

    # ── Load processed dataframe ──────────────────────────────────────────────
    df_file = PROC_DIR / DF_MAP[TIMEPOINT]
    if not df_file.exists():
        log(f'ERROR: {df_file} not found'); sys.exit(1)
    df = pd.read_csv(df_file)
    log(f'\nDataframe loaded: N={len(df)}')

    # Build mean_composite if needed
    if SPLIT_VAR == 'mean_composite':
        comp_cols = ['threat_composite', 'deprivation_composite', 'unpredictability_composite']
        missing = [c for c in comp_cols if c not in df.columns]
        if missing:
            log(f'ERROR: missing columns for mean_composite: {missing}'); sys.exit(1)
        df['mean_composite'] = df[comp_cols].mean(axis=1)

    if SPLIT_VAR not in df.columns:
        log(f'ERROR: {SPLIT_VAR} not in dataframe. Available: {df.columns.tolist()}')
        sys.exit(1)

    # ── Split groups ──────────────────────────────────────────────────────────
    high_df = df[df[SPLIT_VAR] >= HIGH_THRESH].copy()
    low_df  = df[df[SPLIT_VAR] <= LOW_THRESH].copy()

    log(f'\nGroup sizes:')
    log(f'  High adversity (≥{HIGH_THRESH} SD): N={len(high_df)}'
        f'  mean {SPLIT_VAR}={high_df[SPLIT_VAR].mean():.2f}'
        f'  mean SCAN={high_df["SCAN"].mean():.0f} mm²' if 'SCAN' in high_df.columns else '')
    log(f'  Low  adversity (≤{LOW_THRESH} SD): N={len(low_df)}'
        f'  mean {SPLIT_VAR}={low_df[SPLIT_VAR].mean():.2f}'
        f'  mean SCAN={low_df["SCAN"].mean():.0f} mm²' if 'SCAN' in low_df.columns else '')

    # ── Load atlas BrainModelAxis ─────────────────────────────────────────────
    bm, l_slc, r_slc = load_atlas_bm()
    n_grayord = bm.size
    log(f'\nAtlas: {n_grayord:,} grayordinates')
    log(f'  Cortex L: {l_slc}  R: {r_slc}')

    # ── Load per-subject SCAN masks in parallel ───────────────────────────────
    ses = SESSION_MAP[TIMEPOINT]

    def load_group_masks(sub_ids, label):
        args = [(sid, ses) for sid in sub_ids]
        log(f'\nLoading {label} group ({len(args)} subjects, {N_JOBS} workers)...')
        with Pool(N_JOBS) as pool:
            results = pool.map(load_scan_mask, args)
        valid = [r for r in results if r is not None]
        n_fail = len(results) - len(valid)
        if n_fail:
            log(f'  WARNING: {n_fail} subjects failed to load (missing files)')
        log(f'  Loaded: {len(valid)} / {len(args)}')
        return valid

    high_masks = load_group_masks(high_df['sub_ID'].tolist(), 'high adversity')
    low_masks  = load_group_masks(low_df['sub_ID'].tolist(), 'low adversity')

    if not high_masks or not low_masks:
        log('ERROR: one or both groups have no loadable subjects'); sys.exit(1)

    # ── Compute density maps ──────────────────────────────────────────────────
    log('\nComputing density maps...')
    high_density = np.mean(np.stack(high_masks, axis=0), axis=0)  # (91282,)
    low_density  = np.mean(np.stack(low_masks,  axis=0), axis=0)
    diff_density = high_density - low_density

    # NaN out non-cortical grayordinates (subcortex) for clean visualization
    cortical_mask = np.zeros(n_grayord, dtype=bool)
    cortical_mask[l_slc] = True
    cortical_mask[r_slc] = True
    for arr in (high_density, low_density, diff_density):
        arr[~cortical_mask] = np.nan

    # Stats
    log(f'\nDensity stats (cortical vertices only):')
    for name, arr in [('High adversity', high_density),
                      ('Low adversity',  low_density),
                      ('Difference',     diff_density)]:
        cort = arr[cortical_mask]
        log(f'  {name}: mean={np.nanmean(cort):.4f}, '
            f'max={np.nanmax(cort):.4f}, min={np.nanmin(cort):.4f}')

    # ── Save CIFTI files ──────────────────────────────────────────────────────
    log('\nSaving CIFTI dscalar files...')
    var_label = SPLIT_VAR.replace('_composite','').replace('_','')
    prefix = f'SCAN_density_{var_label}_{TIMEPOINT}'

    def save_dscalar(data_2d, map_names, out_path):
        scalar_ax = nib.cifti2.ScalarAxis(map_names)
        header    = nib.Cifti2Header.from_axes((scalar_ax, bm))
        img       = nib.Cifti2Image(data_2d.astype(np.float32), header=header)
        nib.save(img, str(out_path))
        log(f'  → {out_path.name}')

    n_high = len(high_masks)
    n_low  = len(low_masks)

    # Individual maps
    save_dscalar(high_density[np.newaxis, :],
                 [f'SCAN_density_high_{var_label}_n{n_high}'],
                 OUT_DIR / f'{prefix}_high.dscalar.nii')
    save_dscalar(low_density[np.newaxis, :],
                 [f'SCAN_density_low_{var_label}_n{n_low}'],
                 OUT_DIR / f'{prefix}_low.dscalar.nii')
    save_dscalar(diff_density[np.newaxis, :],
                 [f'SCAN_density_diff_high-low_{var_label}'],
                 OUT_DIR / f'{prefix}_diff.dscalar.nii')

    # Combined 3-map file (most convenient for CW comparison)
    data_3map = np.stack([high_density, low_density, diff_density], axis=0)
    save_dscalar(data_3map,
                 [f'high_{var_label}_n{n_high}',
                  f'low_{var_label}_n{n_low}',
                  f'diff_high-minus-low'],
                 OUT_DIR / f'{prefix}_all3.dscalar.nii')

    log(f"""
Connectome Workbench tips for the density maps:
  High/Low maps  : palette "hot" or videen_style; range 0.00 → 0.20; Display zero: OFF
  Difference map : palette ROY-BIG-BL (diverging); symmetric range ±0.10; Display zero: OFF
  Vertices where diff > 0 → more subjects in high-adversity group have SCAN there (expansion).
  Vertices where diff < 0 → more subjects in low-adversity group have SCAN there.
""")

    log('Done.')


if __name__ == '__main__':
    main()
