"""
Compute per-subject SCAN encroachment fractions for all 4 timepoints.

Algorithm (Python port of Hermosillo example_network_encroachment_area.m):
  For each subject, find cortical vertices where:
    - Subject's boldmap assigns SCAN (label 18)
    - The group template assigns a DIFFERENT labeled network (not SCAN, not background)
  These are "encroachment vertices" — SCAN territory in the subject that is
  typical territory of another network in the group average.
  For each displaced network N:
    encroach_frac_N = encroachment_vertices_where_template==N / template_vertices_where_label==N

Zone split (medial vs lateral, Lynch-style):
  Medial: |x| < MEDIAL_THRESH mm (cingulate/ACC/PCC/mPFC axis)
  Lateral: |x| >= MEDIAL_THRESH mm (insula/opercular/lateral PFC)
  Zone fractions use zone-specific template counts as denominators.

Uses vertex counts (not area-weighted), consistent with encrouched_percent_cort in the
MATLAB reference. Valid on fsLR-32k where vertex spacing is approximately uniform.

Outputs:
  outputs/encroachment/encroachment_{tp}.csv   (one row per subject, all 4 timepoints)
"""

import glob, warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from multiprocessing import Pool, cpu_count

from adtopo.config import cfg
from adtopo.logging_utils import get_logger
_log = get_logger('compute_encroachment')

# ── Constants ─────────────────────────────────────────────────────────────────
SCAN_LABEL    = 18
N_CORT        = 59412   # cortical grayordinates in 91k CIFTI (L: 0-29696, R: 29696-59412)
N_JOBS        = min(16, cpu_count())
MEDIAL_THRESH = 20.0    # mm; |x| < threshold → medial zone (cingulate axis)

ATLAS_PATH  = cfg.ATLAS_DIR / 'abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii'
OUT_DIR     = Path(__file__).parent.parent / 'outputs' / 'encroachment'

# fsLR-32k midthickness surfaces for vertex x-coordinates (zone split).
# Any subject's fsLR-32k surface works; all are registered to the same template.
SURF_L = cfg.XCP_DIR / 'sub-UM9EFLC3/ses-00A/anat/sub-UM9EFLC3_ses-00A_run-01_hemi-L_space-fsLR_den-32k_desc-hcp_midthickness.surf.gii'
SURF_R = cfg.XCP_DIR / 'sub-UM9EFLC3/ses-00A/anat/sub-UM9EFLC3_ses-00A_run-01_hemi-R_space-fsLR_den-32k_desc-hcp_midthickness.surf.gii'

BOLDMAP_GLOB = (
    '*_task-rest_space-fsLR_den-91k_desc-denoised-spatially-interpolated-'
    'smoothed-2.25mm-censor-ReproTM_template-ABCC-a3-9to16_refine-SCAN_'
    'minsize-30_boldmap.dlabel.nii'
)

NET_MAP = {
    'DMN': 1, 'VIS': 2, 'FP': 3,  'DAN': 5,  'VAN': 7,
    'SAL': 8, 'CO':  9, 'SMD': 10, 'SML': 11, 'AUD': 12,
    'Tpole': 13, 'MTL': 14, 'PMN': 15, 'PON': 16, 'SCAN': 18,
}
TARGET_NETS = [n for n in cfg.NETWORKS if n != 'SCAN']   # 14 networks

TP_MAP = {
    '00A': ('df_base.csv', 'baseline'),
    '02A': ('df_y2.csv',   'year2'),
    '04A': ('df_y4.csv',   'year4'),
    '06A': ('df_y6.csv',   'year6'),
}


def log(msg=''):
    _log.info(str(msg))


# ── Shared template (set once per worker process via initializer) ──────────────
_TEMPLATE_CORT        = None
_TEMPLATE_COUNTS      = None   # dict: net_name → total template vertices
_TEMPLATE_COUNTS_MED  = None   # dict: net_name → medial-zone template vertices
_TEMPLATE_COUNTS_LAT  = None   # dict: net_name → lateral-zone template vertices
_MEDIAL_MASK          = None   # bool array (N_CORT,): True = medial zone

def _init_worker(template_cort, template_counts, template_counts_med,
                 template_counts_lat, medial_mask):
    global _TEMPLATE_CORT, _TEMPLATE_COUNTS, _TEMPLATE_COUNTS_MED
    global _TEMPLATE_COUNTS_LAT, _MEDIAL_MASK
    _TEMPLATE_CORT       = template_cort
    _TEMPLATE_COUNTS     = template_counts
    _TEMPLATE_COUNTS_MED = template_counts_med
    _TEMPLATE_COUNTS_LAT = template_counts_lat
    _MEDIAL_MASK         = medial_mask


def find_boldmap(sub_id, session):
    hits = glob.glob(str(cfg.REPRO_DIR / sub_id / f'ses-{session}' / 'func' / BOLDMAP_GLOB))
    return hits[0] if hits else None


def _worker(args):
    """Compute encroachment stats for one subject. Returns dict or None."""
    sub_id, session = args
    fpath = find_boldmap(sub_id, session)
    if fpath is None:
        return None
    try:
        sub_cort = nib.load(fpath).get_fdata()[0][:N_CORT].astype(np.int16)
    except Exception:
        return None

    # Encroachment mask: subject has SCAN, template has labeled non-SCAN network
    sub_scan    = (sub_cort == SCAN_LABEL)
    tmpl_other  = (_TEMPLATE_CORT != SCAN_LABEL) & (_TEMPLATE_CORT != 0)
    encroach    = sub_scan & tmpl_other

    row = {'sub_ID': sub_id,
           'scan_size_subject':    int(sub_scan.sum()),
           'scan_size_template':   int((_TEMPLATE_CORT == SCAN_LABEL).sum()),
           'scan_expansion':       int(sub_scan.sum()) - int((_TEMPLATE_CORT == SCAN_LABEL).sum()),
           'total_encroach_count': int(encroach.sum()),
           'total_encroach_count_medial':  int((encroach & _MEDIAL_MASK).sum()),
           'total_encroach_count_lateral': int((encroach & ~_MEDIAL_MASK).sum())}

    for net in TARGET_NETS:
        num        = NET_MAP[net]
        tmpl_net   = (_TEMPLATE_CORT == num)

        # Overall
        count = int(np.sum(encroach & tmpl_net))
        row[f'encroach_count_{net}'] = count
        row[f'encroach_frac_{net}']  = count / _TEMPLATE_COUNTS[net] if _TEMPLATE_COUNTS[net] > 0 else np.nan

        # Medial zone
        count_med = int(np.sum(encroach & tmpl_net & _MEDIAL_MASK))
        row[f'encroach_count_{net}_medial'] = count_med
        n_med = _TEMPLATE_COUNTS_MED[net]
        row[f'encroach_frac_{net}_medial']  = count_med / n_med if n_med > 0 else np.nan

        # Lateral zone
        count_lat = int(np.sum(encroach & tmpl_net & ~_MEDIAL_MASK))
        row[f'encroach_count_{net}_lateral'] = count_lat
        n_lat = _TEMPLATE_COUNTS_LAT[net]
        row[f'encroach_frac_{net}_lateral']  = count_lat / n_lat if n_lat > 0 else np.nan

    return row


def run_timepoint(session, df_file, tp_label):
    log(f'\n{"="*60}')
    log(f'Timepoint: {tp_label} (ses-{session})')
    log(f'{"="*60}')

    df_path = cfg.DAT_DIR / df_file
    if not df_path.exists():
        log(f'  WARNING: {df_file} not found — skipping'); return

    df = pd.read_csv(df_path)
    sub_ids = df['sub_ID'].tolist()
    log(f'  Subjects in dataframe: {len(sub_ids)}')

    # Load template
    atlas_img     = nib.load(str(ATLAS_PATH))
    template_full = atlas_img.get_fdata()[0][:N_CORT].astype(np.int16)
    template_counts = {net: int(np.sum(template_full == NET_MAP[net]))
                       for net in TARGET_NETS}
    log(f'  Template SCAN vertices: {int(np.sum(template_full == SCAN_LABEL))}')
    log(f'  Template per-network counts: ' +
        ', '.join(f'{n}={template_counts[n]}' for n in TARGET_NETS[:5]) + '...')

    # Load surface coordinates for zone split.
    # fsLR-32k surfaces have 32,492 vertices/hemisphere (incl. medial wall);
    # CIFTI has only 59,412 cortical grayordinates. Use BrainModelAxis to map
    # the correct surface vertices to their CIFTI positions.
    coords_L     = nib.load(str(SURF_L)).darrays[0].data   # (32492, 3)
    coords_R     = nib.load(str(SURF_R)).darrays[0].data   # (32492, 3)
    coords_cort  = np.zeros((N_CORT, 3), dtype=np.float32)
    for struct_name, slc, bm in atlas_img.header.get_axis(1).iter_structures():
        if 'CORTEX_LEFT'  in struct_name:
            coords_cort[slc] = coords_L[bm.vertex]
        elif 'CORTEX_RIGHT' in struct_name:
            coords_cort[slc] = coords_R[bm.vertex]
    medial_mask  = np.abs(coords_cort[:, 0]) < MEDIAL_THRESH   # (N_CORT,)
    log(f'  Zone split (|x|<{MEDIAL_THRESH}mm): '
        f'medial={medial_mask.sum():,}, lateral={(~medial_mask).sum():,} vertices')

    template_counts_med = {net: int(np.sum((template_full == NET_MAP[net]) & medial_mask))
                           for net in TARGET_NETS}
    template_counts_lat = {net: int(np.sum((template_full == NET_MAP[net]) & ~medial_mask))
                           for net in TARGET_NETS}

    # Parallel computation
    args = [(sid, session) for sid in sub_ids]
    log(f'  Running {len(args)} subjects with {N_JOBS} workers...')
    with Pool(N_JOBS,
              initializer=_init_worker,
              initargs=(template_full, template_counts,
                        template_counts_med, template_counts_lat,
                        medial_mask)) as pool:
        results = pool.map(_worker, args)

    valid   = [r for r in results if r is not None]
    n_fail  = len(results) - len(valid)
    log(f'  Loaded: {len(valid)} / {len(args)}  ({n_fail} missing files)')

    out_df = pd.DataFrame(valid)
    out_df['timepoint'] = tp_label

    # Sanity checks
    frac_cols = [f'encroach_frac_{n}' for n in TARGET_NETS]
    out_of_range = (out_df[frac_cols] < 0).any().any() or (out_df[frac_cols] > 1).any().any()
    if out_of_range:
        log('  WARNING: some encroachment fractions outside [0,1]')

    # Summary stats
    log(f'\n  Mean encroachment fractions (top 5 by magnitude):')
    means = out_df[frac_cols].mean().sort_values(ascending=False)
    for col, val in means.head(5).items():
        net = col.replace('encroach_frac_', '')
        log(f'    {net:8s}: {val:.4f} ({val*100:.2f}%)')

    # Save
    out_path = OUT_DIR / f'encroachment_{tp_label}.csv'
    out_df.to_csv(out_path, index=False)
    log(f'\n  Saved: {out_path.name}  (N={len(out_df)})')
    return out_df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 60)
    log('SCAN ENCROACHMENT ANALYSIS — all timepoints')
    log(f'Output: {OUT_DIR}')
    log('=' * 60)

    for session, (df_file, tp_label) in TP_MAP.items():
        run_timepoint(session, df_file, tp_label)

    log('\nAll timepoints complete.')


if __name__ == '__main__':
    main()
