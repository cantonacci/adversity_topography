# -*- coding: utf-8 -*-
"""
Generate SCAN territory dlabels and border files for Connectome Workbench.

Creates solid-color dlabels (purple SCAN territory) and crisp boundary border
files by thresholding SCAN probability maps. Five reference conditions:

  atlas            — template parcellation SCAN vertices (exact, binary)
  all_subjects     — whole-sample mean P(SCAN), threshold >= THRESHOLD
  low_adversity    — threat ≤ -1 SD group (from existing density dscalar)
  high_adversity_1sd   — threat ≥ +1 SD (from existing full-prob dscalar)
  high_adversity_p10p90 — top-10% threat (from existing full-prob dscalar)

Outputs:
  outputs/cifti_for_workbench/
    SCAN_dlabel_atlas.dlabel.nii
    SCAN_density_all_subjects_baseline.dscalar.nii   ← new, whole-sample
    SCAN_dlabel_all_subjects_tXX_baseline.dlabel.nii
    SCAN_dlabel_low_adversity_tXX_baseline.dlabel.nii
    SCAN_dlabel_high_adversity_1sd_tXX_baseline.dlabel.nii
    SCAN_dlabel_high_adversity_p10p90_tXX_baseline.dlabel.nii

  outputs/cifti_for_workbench/borders/
    SCAN_border_{name}_{L,R}.border

Usage:
  python generate_scan_borders.py [--threshold 0.25]

Requires: wb_command in PATH (module load biology workbench/1.3.1)
"""

import glob, warnings, subprocess, argparse
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from multiprocessing import Pool, cpu_count

from adtopo.config import cfg
from adtopo.logging_utils import get_logger
_log = get_logger('generate_scan_borders')

SCAN_LABEL = 18
N_CORT     = 59412
N_FULL     = 91282
N_JOBS     = min(16, cpu_count())

ATLAS_PATH = cfg.ATLAS_DIR / 'abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii'
OUT_DIR    = Path(__file__).parent.parent / 'outputs' / 'cifti_for_workbench'
BORDER_DIR = OUT_DIR / 'borders'

# Standard fsLR-32k midthickness (templateflow stubs at ELS_BIDS are empty;
# all ABCD subjects are registered to the same space so any subject surface is valid)
SURF_L = cfg.XCP_DIR / 'sub-DHCPMWJD/ses-04A/anat/sub-DHCPMWJD_ses-04A_rec-norm_hemi-L_space-fsLR_den-32k_desc-hcp_midthickness.surf.gii'
SURF_R = cfg.XCP_DIR / 'sub-DHCPMWJD/ses-04A/anat/sub-DHCPMWJD_ses-04A_rec-norm_hemi-R_space-fsLR_den-32k_desc-hcp_midthickness.surf.gii'

BOLDMAP_GLOB = (
    '*_task-rest_space-fsLR_den-91k_desc-denoised-spatially-interpolated-'
    'smoothed-2.25mm-censor-ReproTM_template-ABCC-a3-9to16_refine-SCAN_'
    'minsize-30_boldmap.dlabel.nii'
)

# SCAN purple: #8E0067 = (142, 0, 103)
SCAN_RGBA = (142 / 255, 0 / 255, 103 / 255, 1.0)


def log(msg=''):
    _log.info(str(msg))


# ── SCAN mask loading ─────────────────────────────────────────────────────────

def find_boldmap(sub_id):
    hits = glob.glob(str(cfg.REPRO_DIR / sub_id / 'ses-00A' / 'func' / BOLDMAP_GLOB))
    return hits[0] if hits else None


def load_scan_mask(sub_id):
    """Return (N_FULL,) float32 binary SCAN mask, or None."""
    fpath = find_boldmap(sub_id)
    if fpath is None:
        return None
    try:
        data = nib.load(fpath).get_fdata()[0].astype(np.int16)
        mask = np.zeros(N_FULL, dtype=np.float32)
        mask[:N_CORT] = (data[:N_CORT] == SCAN_LABEL).astype(np.float32)
        return mask
    except Exception:
        return None


def compute_density(sub_ids, label):
    """Compute mean P(SCAN) across sub_ids using multiprocessing."""
    log(f'  Computing SCAN density for {label} (N={len(sub_ids)}, {N_JOBS} workers)...')
    with Pool(N_JOBS) as pool:
        results = pool.map(load_scan_mask, sub_ids)
    valid = [r for r in results if r is not None]
    n_fail = len(results) - len(valid)
    if n_fail:
        log(f'    WARNING: {n_fail} subjects failed (missing files)')
    log(f'    Loaded: {len(valid)} / {len(sub_ids)}')
    density = np.mean(np.stack(valid), axis=0)
    return density, len(valid)


# ── CIFTI construction ────────────────────────────────────────────────────────

def make_scan_dlabel(label_data_full, bm_ax, map_name):
    """Wrap an (N_FULL,) int32 array into a SCAN-only dlabel image."""
    label_table = {
        0:          ('Background', (0.5, 0.5, 0.5, 0.0)),
        SCAN_LABEL: ('SCAN',       SCAN_RGBA),
    }
    label_ax = nib.cifti2.LabelAxis([map_name], [label_table])
    header   = nib.Cifti2Header.from_axes((label_ax, bm_ax))
    return nib.Cifti2Image(label_data_full[np.newaxis, :].astype(np.float32), header=header)


def atlas_to_scan_dlabel(bm_ax):
    """Extract SCAN vertices from the atlas parcellation (binary, no threshold)."""
    template = nib.load(str(ATLAS_PATH)).get_fdata()[0].astype(np.int16)
    label_data = np.zeros(N_FULL, dtype=np.int32)
    label_data[:N_CORT] = np.where(template[:N_CORT] == SCAN_LABEL, SCAN_LABEL, 0)
    return make_scan_dlabel(label_data, bm_ax, 'SCAN_atlas')


def prob_to_scan_dlabel(prob_full, cort_mask, bm_ax, map_name, threshold):
    """Threshold a (N_FULL,) probability array → SCAN-only dlabel."""
    label_data = np.zeros(N_FULL, dtype=np.int32)
    label_data[cort_mask & (prob_full >= threshold)] = SCAN_LABEL
    n_verts = int((label_data == SCAN_LABEL).sum())
    log(f'    SCAN vertices at threshold {threshold:.2f}: {n_verts:,}')
    return make_scan_dlabel(label_data, bm_ax, map_name)


def dscalar_to_scan_dlabel(dscalar_path, cort_mask, bm_ax, map_name, threshold):
    """Load an existing dscalar and threshold → SCAN dlabel."""
    img = nib.load(str(dscalar_path))
    prob = img.get_fdata()[0]  # (N_FULL,)
    return prob_to_scan_dlabel(prob, cort_mask, bm_ax, map_name, threshold)


def save_dlabel(img, out_path):
    nib.save(img, str(out_path))
    log(f'  → {out_path.name}')


def save_dscalar(data, map_name, bm_ax, out_path):
    scalar_ax = nib.cifti2.ScalarAxis([map_name])
    header    = nib.Cifti2Header.from_axes((scalar_ax, bm_ax))
    img       = nib.Cifti2Image(data[np.newaxis, :].astype(np.float32), header=header)
    nib.save(img, str(out_path))
    log(f'  → {out_path.name}')


# ── wb_command border generation ──────────────────────────────────────────────

def generate_borders(dlabel_path, base_name):
    """Run wb_command -cifti-label-to-border for L and R hemispheres."""
    out_L = BORDER_DIR / f'{base_name}_L.border'
    out_R = BORDER_DIR / f'{base_name}_R.border'
    cmd = [
        'wb_command', '-cifti-label-to-border',
        str(dlabel_path),
        '-border', str(SURF_L), str(out_L),
        '-border', str(SURF_R), str(out_R),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f'    ERROR (wb_command): {result.stderr.strip()}')
    else:
        log(f'  → {out_L.name}')
        log(f'  → {out_R.name}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate SCAN dlabels and border files')
    parser.add_argument('--threshold', type=float, default=0.25,
                        help='P(SCAN) threshold for dlabel creation (default: 0.25)')
    parser.add_argument('--skip-borders', action='store_true',
                        help='Skip wb_command border generation (run separately with workbench loaded)')
    args = parser.parse_args()
    THRESH    = args.threshold
    thr_tag   = f't{int(THRESH * 100):02d}'

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BORDER_DIR.mkdir(parents=True, exist_ok=True)

    log('=' * 65)
    log('SCAN DLABELS AND BORDER FILES')
    log(f'  Probability threshold : {THRESH} ({thr_tag})')
    log(f'  Output dir            : {OUT_DIR}')
    log(f'  Borders dir           : {BORDER_DIR}')
    log('=' * 65)

    # Load atlas BrainModelAxis and build cortical mask
    atlas_img = nib.load(str(ATLAS_PATH))
    bm_ax     = atlas_img.header.get_axis(1)
    cort_mask = np.zeros(N_FULL, dtype=bool)
    for name, slc, _ in bm_ax.iter_structures():
        if 'CORTEX' in name:
            cort_mask[slc] = True
    log(f'\nAtlas loaded | cortical vertices: {cort_mask.sum():,}')

    # Load baseline dataframe
    df = pd.read_csv(cfg.DAT_DIR / 'df_base.csv')
    log(f'Baseline dataframe N={len(df)}')

    # ── [1] Atlas template SCAN ───────────────────────────────────────────────
    log('\n[1] Atlas template SCAN (binary, no threshold)')
    atlas_out = OUT_DIR / 'SCAN_dlabel_atlas.dlabel.nii'
    save_dlabel(atlas_to_scan_dlabel(bm_ax), atlas_out)

    # ── [2] Whole-sample mean P(SCAN) ─────────────────────────────────────────
    log('\n[2] Whole-sample SCAN density (all baseline subjects)')
    all_density_path = OUT_DIR / 'SCAN_density_all_subjects_baseline.dscalar.nii'
    if all_density_path.exists():
        log('  Loading cached density dscalar...')
        prob_all = nib.load(str(all_density_path)).get_fdata()[0]
        n_all_str = 'cached'
    else:
        all_ids = df['sub_ID'].tolist()
        prob_all, n_all = compute_density(all_ids, 'all subjects')
        prob_all_nan = prob_all.copy()
        prob_all_nan[~cort_mask] = np.nan
        save_dscalar(prob_all_nan, f'SCAN_density_all_n{n_all}', bm_ax, all_density_path)
        n_all_str = str(n_all)

    log(f'  N={n_all_str} | mean P(SCAN) over cortex: {np.nanmean(prob_all[cort_mask]):.4f}')
    all_dlabel_out = OUT_DIR / f'SCAN_dlabel_all_subjects_{thr_tag}_baseline.dlabel.nii'
    save_dlabel(
        prob_to_scan_dlabel(prob_all, cort_mask, bm_ax, 'SCAN_all_subjects', THRESH),
        all_dlabel_out,
    )

    # ── [3] Low-adversity group (threat ≤ -1 SD) ──────────────────────────────
    log('\n[3] Low-adversity SCAN (threat ≤ -1 SD)')
    low_src = OUT_DIR / 'SCAN_density_threat_baseline_low.dscalar.nii'
    if not low_src.exists():
        log('  Existing density dscalar not found — computing from scratch...')
        low_ids = df[df['threat_composite'] <= -1.0]['sub_ID'].tolist()
        log(f'  N low-adversity: {len(low_ids)}')
        prob_low, n_low = compute_density(low_ids, 'low adversity')
        prob_low_nan = prob_low.copy()
        prob_low_nan[~cort_mask] = np.nan
        save_dscalar(prob_low_nan, f'SCAN_density_low_n{n_low}', bm_ax, low_src)
    else:
        log(f'  Using existing: {low_src.name}')
    low_dlabel_out = OUT_DIR / f'SCAN_dlabel_low_adversity_{thr_tag}_baseline.dlabel.nii'
    save_dlabel(
        dscalar_to_scan_dlabel(low_src, cort_mask, bm_ax, 'SCAN_low_adversity', THRESH),
        low_dlabel_out,
    )

    # ── [4] High-adversity groups (convert existing full-prob dscalars) ───────
    log('\n[4] High-adversity SCAN (from existing probability dscalars)')
    for split in ('1sd', 'p10p90'):
        src = OUT_DIR / f'SCAN_full_prob_high_{split}_baseline.dscalar.nii'
        if not src.exists():
            log(f'  SKIP: {src.name} not found')
            continue
        log(f'  Processing {src.name}...')
        out = OUT_DIR / f'SCAN_dlabel_high_adversity_{split}_{thr_tag}_baseline.dlabel.nii'
        save_dlabel(
            dscalar_to_scan_dlabel(src, cort_mask, bm_ax, f'SCAN_high_{split}', THRESH),
            out,
        )

    # ── [5] Border files via wb_command ───────────────────────────────────────
    if args.skip_borders:
        log('\n[5] Skipping border generation (--skip-borders set)')
        log('    Run wb_command step separately (see submit_scan_borders.sh)')
    else:
        log('\n[5] Generating border files (wb_command)')
        border_specs = [
            (OUT_DIR / 'SCAN_dlabel_atlas.dlabel.nii',
             'SCAN_border_atlas'),
            (OUT_DIR / f'SCAN_dlabel_all_subjects_{thr_tag}_baseline.dlabel.nii',
             f'SCAN_border_all_subjects_{thr_tag}'),
            (OUT_DIR / f'SCAN_dlabel_low_adversity_{thr_tag}_baseline.dlabel.nii',
             f'SCAN_border_low_adversity_{thr_tag}'),
            (OUT_DIR / f'SCAN_dlabel_high_adversity_1sd_{thr_tag}_baseline.dlabel.nii',
             f'SCAN_border_high_adversity_1sd_{thr_tag}'),
            (OUT_DIR / f'SCAN_dlabel_high_adversity_p10p90_{thr_tag}_baseline.dlabel.nii',
             f'SCAN_border_high_adversity_p10p90_{thr_tag}'),
        ]
        for dlabel_path, base_name in border_specs:
            if not dlabel_path.exists():
                log(f'  SKIP: {dlabel_path.name} not found')
                continue
            log(f'\n  {dlabel_path.name}')
            generate_borders(dlabel_path, base_name)

    log('\n' + '=' * 65)
    log('Done.')
    log(f"""
Workbench visualization notes:
  dlabels : SCAN territory shown in purple (#8E0067); background transparent.
            Load directly as an overlay on the surface.
  borders : Crisp SCAN outlines at threshold={THRESH} (P >= {THRESH*100:.0f}% of group).
            Load via File -> Open -> *.border; set color/width in the border dialog.

Three-way comparison:
  atlas            = template expectation (exact, derives from group)
  all_subjects     = where SCAN sits in a typical child in this sample
  low_adversity    = "unexposed" normative SCAN boundary
  high_adversity_* = expanded SCAN territory in high-threat youth
""")


if __name__ == '__main__':
    main()
