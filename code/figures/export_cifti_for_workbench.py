"""
Export statistical maps as CIFTI dscalar.nii files for Connectome Workbench.

Outputs (all in outputs/cifti_for_workbench/):
  ELA_delta_r2_baseline.dscalar.nii          — overall ΔR² (baseline)
  ELA_delta_r2_year2.dscalar.nii
  ELA_composite_betas_baseline.dscalar.nii   — 3 maps: threat / deprivation / unpredictability β
  ELA_composite_betas_year2.dscalar.nii
  ELA_composite_betas_year4.dscalar.nii
  ELA_composite_betas_year6.dscalar.nii
  ELA_atlas_parcellation.dlabel.nii          — network parcellation (re-export for CW use)

Load in Connectome Workbench:
  File → Open File → select .dscalar.nii
  In the overlay toolbar, click the color palette icon to set colormap + range.
  For the parcellation, use the .dlabel.nii which carries label colors automatically.
"""

import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

from adtopo.config import cfg
from adtopo.logging_utils import get_logger
_log = get_logger('export_cifti_for_workbench')

OUT_DIR = Path(__file__).parent.parent / 'outputs' / 'cifti_for_workbench'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATLAS_PATH = cfg.ATLAS_DIR / 'abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii'

NET_MAP = {
    'DMN': 1,  'VIS': 2,  'FP': 3,   'DAN': 5,   'VAN': 7,
    'SAL': 8,  'CO': 9,   'SMD': 10, 'SML': 11,  'AUD': 12,
    'Tpole': 13, 'MTL': 14, 'PMN': 15, 'PON': 16, 'SCAN': 18,
}

# Suggested CW palette settings (print at end for reference)
PALETTE_NOTES = """
Suggested Connectome Workbench palette settings
================================================
ΔR² maps (ELA_delta_r2_*.dscalar.nii):
  Palette: videen_style (warm sequential), or "ROY-BIG-BL" with pos only
  Range: 0 to ~0.01 (check actual max; use Display Range → Fixed → 0.000 to 0.010)
  Interpolate colors: ON; Display zero: OFF (show only positive values)

Composite beta maps (ELA_composite_betas_*.dscalar.nii):
  Palette: "ROY-BIG-BL" (diverging blue→red) or "PSYCH-NO-NONE"
  Range: symmetric, e.g. -0.004 to +0.004
  Interpolate colors: ON; Display zero: OFF

Network parcellation (ELA_atlas_parcellation.dlabel.nii):
  Label colors come from the embedded label table automatically.
  No palette adjustments needed.

Lynch et al. 2024 (Nature) used: inflated/very_inflated fsLR-32k surfaces
  with the Human Connectome Project S1200 group-average surfaces, which
  are available from: https://www.humanconnectome.org/study/hcp-young-adult/
  Surface: S1200.{L,R}.very_inflated_MSMAll.32k_fs_LR.surf.gii
  If you want to match exactly, load those surfaces in CW (File → Open File)
  and then overlay your dscalar on them.
"""


def log(msg=''):
    _log.info(str(msg))


def load_atlas_and_bm():
    img = nib.load(str(ATLAS_PATH))
    atlas_data = img.get_fdata()[0]          # shape: (91282,)
    bm_ax = img.header.get_axis(1)            # BrainModelAxis
    return atlas_data, bm_ax, img


def make_grayordinate_map(atlas_data, net_scores):
    """Map network-level scores onto grayordinates. NaN for unlabeled."""
    out = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    for net, score in net_scores.items():
        if net in NET_MAP and score is not None:
            try:
                v = float(score)
                if not np.isnan(v):
                    out[atlas_data == NET_MAP[net]] = v
            except (TypeError, ValueError):
                pass
    return out


def save_dscalar(data_2d, map_names, bm_ax, out_path):
    """
    Save a CIFTI2 dscalar.nii.

    data_2d: shape (n_maps, n_grayordinates), float32
    map_names: list of str, length n_maps
    """
    scalar_ax = nib.cifti2.ScalarAxis(map_names)
    header = nib.Cifti2Header.from_axes((scalar_ax, bm_ax))
    img = nib.Cifti2Image(data_2d.astype(np.float32), header=header)
    nib.save(img, str(out_path))
    log(f'  → {out_path.name}')


def export_delta_r2(atlas_data, bm_ax):
    """Export per-timepoint ΔR² as separate single-map dscalar files."""
    log('\n[1] ΔR² maps...')

    # Primary source: phase3_composites_brain_surface_inputs (has all timepoints)
    bsi_path = cfg.TAB_DIR / 'phase3_composites_brain_surface_inputs.csv'
    if not bsi_path.exists():
        log(f'  WARNING: {bsi_path} not found — skipping ΔR² export')
        return

    bsi = pd.read_csv(bsi_path)

    for tp in bsi['timepoint'].unique():
        sub = bsi[bsi['timepoint'] == tp].set_index('network')
        if 'delta_R2' not in sub.columns:
            continue
        scores = sub['delta_R2'].to_dict()
        gmap = make_grayordinate_map(atlas_data, scores)

        # Also try to load the individual-ELA delta_r2 table for richer info
        indiv_path = cfg.TAB_DIR / f'phase3_individual_delta_r2_{tp}.csv'
        extra_maps = []
        extra_names = []
        if indiv_path.exists():
            indiv = pd.read_csv(indiv_path)
            if 'ELA_factor' in indiv.columns:
                for ela in indiv['ELA_factor'].unique():
                    esub = indiv[indiv['ELA_factor'] == ela].set_index('network')
                    if 'delta_R2' in esub.columns:
                        extra_maps.append(make_grayordinate_map(atlas_data, esub['delta_R2'].to_dict()))
                        extra_names.append(f'delta_R2_{ela}')

        # Save composite ΔR² (single map)
        out_path = OUT_DIR / f'ELA_delta_r2_{tp}.dscalar.nii'
        save_dscalar(gmap[np.newaxis, :], [f'ELA_delta_R2_{tp}'], bm_ax, out_path)

        # Save per-factor ΔR² if available (multi-map)
        if extra_maps:
            out_path2 = OUT_DIR / f'ELA_individual_delta_r2_{tp}.dscalar.nii'
            data_2d = np.stack(extra_maps, axis=0)
            save_dscalar(data_2d, extra_names, bm_ax, out_path2)
            log(f'    ({len(extra_names)} individual ELA factors)')


def export_composite_betas(atlas_data, bm_ax):
    """Export threat/deprivation/unpredictability β as multi-map dscalar files."""
    log('\n[2] Composite beta maps...')

    bsi_path = cfg.TAB_DIR / 'phase3_composites_brain_surface_inputs.csv'
    if not bsi_path.exists():
        log(f'  WARNING: {bsi_path} not found — skipping beta export')
        return

    bsi = pd.read_csv(bsi_path)
    beta_cols = {
        'threat':           'threat_composite_beta',
        'deprivation':      'deprivation_composite_beta',
        'unpredictability': 'unpredictability_composite_beta',
    }

    for tp in bsi['timepoint'].unique():
        sub = bsi[bsi['timepoint'] == tp].set_index('network')
        maps = []
        names = []
        for label, col in beta_cols.items():
            if col not in sub.columns:
                continue
            gmap = make_grayordinate_map(atlas_data, sub[col].to_dict())
            maps.append(gmap)
            names.append(f'{label}_beta_{tp}')

        if not maps:
            continue

        out_path = OUT_DIR / f'ELA_composite_betas_{tp}.dscalar.nii'
        save_dscalar(np.stack(maps, axis=0), names, bm_ax, out_path)
        log(f'    ({len(names)} composites: {", ".join(names)})')


def export_individual_betas(atlas_data, bm_ax):
    """Export individual ELA factor β for each timepoint."""
    log('\n[3] Individual ELA factor beta maps...')

    timepoints = ['baseline', 'year2', 'year4', 'year6']
    for tp in timepoints:
        res_path = cfg.TAB_DIR / f'phase3_individual_results_{tp}.csv'
        if not res_path.exists():
            continue
        res = pd.read_csv(res_path)
        # predictor column may be named 'predictor' or 'ELA_factor'
        pred_col = 'predictor' if 'predictor' in res.columns else 'ELA_factor'
        if pred_col not in res.columns or 'beta' not in res.columns:
            continue
        if 'network' not in res.columns:
            continue
        # Filter to ELA predictors only (exclude age, sex, FD, etc.)
        ela_rows = res[res[pred_col].str.startswith('ELA_', na=False)]
        if ela_rows.empty:
            continue

        maps = []
        names = []
        for ela in sorted(ela_rows[pred_col].unique()):
            esub = ela_rows[ela_rows[pred_col] == ela].set_index('network')
            gmap = make_grayordinate_map(atlas_data, esub['beta'].to_dict())
            maps.append(gmap)
            names.append(f'{ela}_beta_{tp}')

        if not maps:
            continue

        out_path = OUT_DIR / f'ELA_individual_betas_{tp}.dscalar.nii'
        save_dscalar(np.stack(maps, axis=0), names, bm_ax, out_path)
        log(f'    {tp}: {len(names)} ELA factors')


def export_atlas_dlabel(bm_ax, template_img):
    """Re-export the atlas parcellation as a clean dlabel for CW use."""
    log('\n[4] Network parcellation dlabel...')
    # Just copy the existing atlas with a clean filename
    out_path = OUT_DIR / 'ELA_atlas_parcellation.dlabel.nii'
    nib.save(template_img, str(out_path))
    log(f'  → {out_path.name}')


def export_phase2_r_matrices(atlas_data, bm_ax):
    """Export bivariate partial r values as dscalar for each timepoint."""
    log('\n[5] Phase 2 bivariate partial r maps...')

    timepoints = ['baseline', 'year2', 'year4', 'year6']
    composites = ['threat', 'deprivation', 'unpredictability']

    for tp in timepoints:
        r_path = cfg.TAB_DIR / f'phase2_composites_r_matrix_{tp}.csv'
        q_path = cfg.TAB_DIR / f'phase2_composites_q_matrix_{tp}.csv'
        if not r_path.exists():
            continue

        r_df = pd.read_csv(r_path, index_col=0)
        q_df = pd.read_csv(q_path, index_col=0) if q_path.exists() else None

        maps = []
        names = []
        # Also make FDR-masked versions (only significant associations)
        maps_sig = []

        # r_df: rows = ELA factors (e.g. "threat_composite"), cols = networks
        # Transpose so we can look up by composite name
        r_df_T = r_df.T          # now rows = networks, cols = ELA factors
        q_df_T = q_df.T if q_df is not None else None

        for comp in composites:
            # Match columns that contain the composite name (e.g. "threat_composite")
            matching = [c for c in r_df_T.columns if comp in c.lower()]
            if not matching:
                continue
            col = matching[0]
            scores = r_df_T[col].to_dict()
            gmap = make_grayordinate_map(atlas_data, scores)
            maps.append(gmap)
            names.append(f'{comp}_r_{tp}')

            if q_df_T is not None and col in q_df_T.columns:
                q_scores = q_df_T[col].to_dict()
                sig_scores = {
                    net: (rv if q_scores.get(net, 1.0) < 0.05 else np.nan)
                    for net, rv in scores.items()
                }
                gmap_sig = make_grayordinate_map(atlas_data, sig_scores)
                maps_sig.append(gmap_sig)

        if not maps:
            continue

        out_path = OUT_DIR / f'ELA_bivariate_r_{tp}.dscalar.nii'
        save_dscalar(np.stack(maps, axis=0), names, bm_ax, out_path)

        if maps_sig:
            out_path_sig = OUT_DIR / f'ELA_bivariate_r_FDRsig_{tp}.dscalar.nii'
            sig_names = [n.replace('_r_', '_r_FDRsig_') for n in names[:len(maps_sig)]]
            save_dscalar(np.stack(maps_sig, axis=0), sig_names, bm_ax, out_path_sig)
            log(f'    {tp}: r maps + FDR-masked versions')


def main():
    log('=' * 70)
    log('CIFTI EXPORT FOR CONNECTOME WORKBENCH')
    log(f'Output: {OUT_DIR}')
    log('=' * 70)

    if not ATLAS_PATH.exists():
        log(f'ERROR: atlas not found at {ATLAS_PATH}')
        sys.exit(1)

    atlas_data, bm_ax, template_img = load_atlas_and_bm()
    log(f'Atlas loaded: {atlas_data.shape[0]:,} grayordinates')

    export_delta_r2(atlas_data, bm_ax)
    export_composite_betas(atlas_data, bm_ax)
    export_individual_betas(atlas_data, bm_ax)
    export_atlas_dlabel(bm_ax, template_img)
    export_phase2_r_matrices(atlas_data, bm_ax)

    log('\n' + '=' * 70)
    log('Done. Files saved to:')
    for f in sorted(OUT_DIR.glob('*.nii')):
        log(f'  {f.name}')

    log('\n' + PALETTE_NOTES)


if __name__ == '__main__':
    main()
