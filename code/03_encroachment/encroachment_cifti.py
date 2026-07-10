"""
CIFTI spatial encroachment maps for Connectome Workbench.

For each cortical vertex, computes:
  - Encroachment probability in high-adversity group (dscalar) — Lynch Fig 2C top
  - Displaced-network map colored by encroached network (dlabel) — Lynch Fig 2C bottom
  - Full SCAN territory probability for high-ELA group (dscalar) — Lynch Fig 2B equivalent
  - Representative subject (sub-BMFRB748) full SCAN territory (dscalar) — single-subject 2B

Outputs (outputs/cifti_for_workbench/):
  SCAN_encroachment_prob_high_{split}_baseline.dscalar.nii     ← encroachment only
  SCAN_encroachment_displaced_network_high_{split}_baseline.dlabel.nii  ← Lynch-style
  SCAN_full_prob_high_{split}_baseline.dscalar.nii             ← full SCAN territory
  SCAN_representative_subject_BMFRB748.dscalar.nii             ← single subject
  (encroachment set repeated for p10p90 split)

Load in Connectome Workbench:
  Encroachment probability:    palette "hot", range 0 → 0.15, Display zero: OFF
  Full SCAN probability:       palette "hot", range 0 → 0.80, Display zero: OFF
  Displaced-network dlabel:    colors from label table (each network = its atlas color)
  Representative subject:      palette "hot", range 0 → 1, Display zero: OFF

Lynch-style visualization workflow in Workbench:
  1. Load template atlas dlabel → isolate SCAN label → use as boundary outline (Panel A)
  2. Load SCAN_full_prob_high or representative subject dscalar (Panel B)
  3. Overlay encroachment_prob_high on template SCAN outline → overhang = encroachment
  4. Load displaced_network dlabel → shows WHICH network is displaced (Panel C)
"""

import sys, glob, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import ATLAS_DIR, DAT_DIR, NETWORKS, REPRO_DIR

SCAN_LABEL  = 18
N_CORT      = 59412
N_FULL      = 91282
N_JOBS      = min(16, cpu_count())

ATLAS_PATH  = ATLAS_DIR / 'abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii'
ENC_DIR     = Path(__file__).parent.parent / 'outputs' / 'encroachment'
OUT_DIR     = Path(__file__).parent.parent / 'outputs' / 'cifti_for_workbench'

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
TARGET_NETS = [n for n in NETWORKS if n != 'SCAN']

# Network colors (RGB 0-255) for the dlabel color table
NET_COLORS_RGB = {
    'DMN':   (255,   0,   0), 'VIS':   (  0,   0, 153), 'FP':    (204, 204,   0),
    'DAN':   (  0, 170,   0), 'VAN':   ( 13, 133, 160), 'SAL':   ( 50,  50,  50),
    'CO':    (102,   0, 204), 'SMD':   ( 68, 204, 204), 'SML':   (255, 128,   0),
    'AUD':   (178, 102, 153), 'Tpole': (  0, 102, 153), 'MTL':   ( 85, 204,  85),
    'PMN':   ( 60,  60, 251), 'PON':   (239, 239, 239), 'SCAN':  (142,   0, 103),
}

DF_FILES = {
    'baseline': 'df_base.csv',
}


def log(msg=''):
    print(msg, flush=True)


# ── Shared state ──────────────────────────────────────────────────────────────
_TEMPLATE_CORT = None

def _init_worker(template_cort):
    global _TEMPLATE_CORT
    _TEMPLATE_CORT = template_cort


def find_boldmap(sub_id, session='00A'):
    hits = glob.glob(str(REPRO_DIR / sub_id / f'ses-{session}' / 'func' / BOLDMAP_GLOB))
    return hits[0] if hits else None


def _load_encroach_mask(sub_id):
    """Return binary (N_FULL,) encroachment mask, or None."""
    fpath = find_boldmap(sub_id, '00A')
    if fpath is None:
        return None
    try:
        data = nib.load(fpath).get_fdata()[0].astype(np.int16)
    except Exception:
        return None
    sub_scan   = (data[:N_CORT] == SCAN_LABEL)
    tmpl_other = (_TEMPLATE_CORT != SCAN_LABEL) & (_TEMPLATE_CORT != 0)
    mask_full  = np.zeros(N_FULL, dtype=np.float32)
    mask_full[:N_CORT] = (sub_scan & tmpl_other).astype(np.float32)
    return mask_full


REPRO_SUBJECT = 'sub-BMFRB748'   # highest-SCAN subject in high-threat group

def get_high_ids(df, split='1sd'):
    tc = df['threat_composite']
    if split == '1sd':
        return df[tc >= 1.0]['sub_ID'].tolist()
    p90 = float(np.percentile(tc, 90))
    return df[tc >= p90]['sub_ID'].tolist()


def load_group_masks(sub_ids, label, template_cort):
    log(f'  Loading {label} group ({len(sub_ids)} subjects)...')
    with Pool(N_JOBS, initializer=_init_worker, initargs=(template_cort,)) as pool:
        results = pool.map(_load_encroach_mask, sub_ids)
    valid = [r for r in results if r is not None]
    log(f'    Loaded: {len(valid)} / {len(sub_ids)}')
    return valid


def _load_full_scan_mask(sub_id):
    """Return (N_FULL,) float32 map: 1 where subject has SCAN, 0 elsewhere."""
    fpath = find_boldmap(sub_id, '00A')
    if fpath is None:
        return None
    try:
        data = nib.load(fpath).get_fdata()[0].astype(np.int16)
    except Exception:
        return None
    mask = np.zeros(N_FULL, dtype=np.float32)
    mask[:N_CORT] = (data[:N_CORT] == SCAN_LABEL).astype(np.float32)
    return mask


def save_full_scan_probability(high_ids, bm_ax, cort_mask, split_tag, n_high):
    """Full SCAN territory probability for high-ELA group (Lynch Fig 2B group equivalent)."""
    log(f'  Loading full SCAN masks for {len(high_ids)} high-ELA subjects...')
    with Pool(N_JOBS) as pool:
        results = pool.map(_load_full_scan_mask, high_ids)
    valid = [r for r in results if r is not None]
    log(f'    Loaded: {len(valid)} / {len(high_ids)}')
    if not valid:
        return

    prob = np.mean(np.stack(valid), axis=0)
    prob[~cort_mask] = np.nan

    log(f'    Full SCAN prob (cortical): mean={np.nanmean(prob[cort_mask]):.4f}, '
        f'max={np.nanmax(prob[cort_mask]):.4f}')

    out_path = OUT_DIR / f'SCAN_full_prob_high_{split_tag}_baseline.dscalar.nii'
    save_dscalar(prob[None],
                 [f'SCAN_full_prob_high_{split_tag}_n{n_high}'],
                 bm_ax, out_path)


def save_representative_subject(bm_ax, cort_mask):
    """Save sub-BMFRB748 full SCAN territory as dscalar (Lynch Fig 2B single-subject)."""
    log(f'\n  Representative subject: {REPRO_SUBJECT}')
    fpath = find_boldmap(REPRO_SUBJECT, '00A')
    if fpath is None:
        log(f'    ERROR: boldmap not found for {REPRO_SUBJECT}'); return
    try:
        data = nib.load(fpath).get_fdata()[0].astype(np.int16)
    except Exception as e:
        log(f'    ERROR loading boldmap: {e}'); return

    scan_map = np.full(N_FULL, np.nan, dtype=np.float32)
    scan_map[:N_CORT] = np.where(data[:N_CORT] == SCAN_LABEL, 1.0, 0.0)
    scan_map[~cort_mask] = np.nan

    n_verts = int((data[:N_CORT] == SCAN_LABEL).sum())
    log(f'    SCAN vertices: {n_verts:,} (template = 1,336)')

    out_path = OUT_DIR / f'SCAN_representative_subject_{REPRO_SUBJECT.replace("sub-","")}.dscalar.nii'
    save_dscalar(scan_map[None],
                 [f'{REPRO_SUBJECT}_SCAN_n{n_verts}verts'],
                 bm_ax, out_path)


def save_dscalar(data_2d, map_names, bm_ax, out_path):
    scalar_ax = nib.cifti2.ScalarAxis(map_names)
    header    = nib.Cifti2Header.from_axes((scalar_ax, bm_ax))
    img       = nib.Cifti2Image(data_2d.astype(np.float32), header=header)
    nib.save(img, str(out_path))
    log(f'  → {out_path.name}')


def save_displaced_network_dlabel(prob_map, template_cort, bm_ax, out_path,
                                  threshold=0.05):
    """
    Create a dlabel where each vertex is colored by which network SCAN displaces.
    Only vertices where prob_map > threshold are included.
    """
    # Build label data: template network at vertices that exceed threshold
    label_data = np.zeros(N_FULL, dtype=np.int32)
    for v in range(N_CORT):
        if prob_map[v] > threshold:
            t = int(template_cort[v])
            if t != SCAN_LABEL and t != 0:
                label_data[v] = t

    # Build LabelAxis
    label_table = {0: ('Background', (0.5, 0.5, 0.5, 0.0))}  # transparent background
    for net, num in NET_MAP.items():
        if net == 'SCAN':
            continue
        r, g, b = NET_COLORS_RGB[net]
        label_table[num] = (net, (r/255, g/255, b/255, 1.0))

    label_ax = nib.cifti2.LabelAxis(
        ['displaced_network'],
        [label_table]
    )
    header = nib.Cifti2Header.from_axes((label_ax, bm_ax))
    img    = nib.Cifti2Image(label_data[np.newaxis, :].astype(np.float32), header=header)
    nib.save(img, str(out_path))
    log(f'  → {out_path.name}')
    n_labeled = int((label_data > 0).sum())
    log(f'    Labeled vertices (>{threshold*100:.0f}% threshold): {n_labeled:,}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 60)
    log('SCAN ENCROACHMENT CIFTI MAPS')
    log(f'Output: {OUT_DIR}')
    log('=' * 60)

    # Load atlas
    atlas_img    = nib.load(str(ATLAS_PATH))
    template_full = atlas_img.get_fdata()[0].astype(np.int16)
    template_cort = template_full[:N_CORT]
    bm_ax         = atlas_img.header.get_axis(1)

    # Cortical mask (NaN subcortex in outputs)
    cort_mask = np.zeros(N_FULL, dtype=bool)
    for name, slc, _ in bm_ax.iter_structures():
        if 'CORTEX' in name:
            cort_mask[slc] = True

    # Load baseline df
    df = pd.read_csv(DAT_DIR / 'df_base.csv')
    log(f'\nBaseline N={len(df)}')

    for split in ('1sd', 'p10p90'):
        log(f'\n{"─"*50}')
        log(f'Split: {split}')
        high_ids = get_high_ids(df, split)
        log(f'  High-ELA N={len(high_ids)}')

        # Encroachment masks (SCAN where template = other network)
        high_encroach_masks = load_group_masks(high_ids, 'high adversity', template_cort)
        if not high_encroach_masks:
            log('  ERROR: could not load encroachment masks'); continue

        high_prob = np.mean(np.stack(high_encroach_masks), axis=0)
        high_prob[~cort_mask] = np.nan

        vals = high_prob[cort_mask]
        log(f'  Encroachment probability (cortical): mean={np.nanmean(vals):.4f}, '
            f'max={np.nanmax(vals):.4f}')

        tag  = f'{split}_baseline'
        n_h  = len(high_encroach_masks)

        # Encroachment probability dscalar (Lynch Fig 2C top — the "overhang")
        save_dscalar(high_prob[None],
                     [f'encroach_prob_high_{split}_n{n_h}'],
                     bm_ax, OUT_DIR / f'SCAN_encroachment_prob_high_{tag}.dscalar.nii')

        # Displaced-network dlabel (Lynch Fig 2C bottom — colored by displaced network)
        save_displaced_network_dlabel(
            high_prob[:N_CORT], template_cort, bm_ax,
            OUT_DIR / f'SCAN_encroachment_displaced_network_high_{tag}.dlabel.nii',
            threshold=0.05)

        # Full SCAN territory probability (Lynch Fig 2B group equivalent)
        save_full_scan_probability(high_ids, bm_ax, cort_mask, split, n_h)

    # Representative subject — single high-ELA subject with largest SCAN (Lynch Fig 2B)
    save_representative_subject(bm_ax, cort_mask)

    log(f"""
Connectome Workbench palette settings:
  Probability maps (high/low): palette "hot", range 0.00 → 0.15, Display zero: OFF
  Difference map:              palette ROY-BIG-BL, symmetric ±0.08, Display zero: OFF
  Displaced-network dlabel:    colors from label table (each network has its own color)
    → This is the Lynch-style map: shows WHERE SCAN encroaches and WHICH network is displaced.
""")
    log('Done.')


if __name__ == '__main__':
    main()
