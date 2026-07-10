"""
Population modal (maximum-probability) network atlas — full-surface fill.

The staged consensus atlas (abcd_template_matching_v2_combined_clusters_thresh0.50)
leaves low-consensus vertices grey. This builds a winner-take-all / maximum-
probability map (MPM): for every cortical vertex, assign the network that is the
MODAL (most frequent) assignment across the whole baseline sample. Every vertex
gets a colour → a clean "normative population atlas" for the Fig 1 inset.

Same data source as scan_density_maps.py: each subject's boldmap.dlabel.nii gives
a per-vertex network label; we count all 15 networks per vertex (not just SCAN)
and take the argmax.

Outputs (outputs/cifti_for_workbench/):
  population_modal_atlas_baseline.dlabel.nii          — the MPM, canonical colours
  population_modal_atlas_winfrac_baseline.dscalar.nii — winning fraction per vertex
      (consensus/confidence; free byproduct — lets you optionally make a
       confidence-weighted version later by modulating opacity by this map)

Colours match figsrc.ATLAS_NET_COLORS so the surface is consistent with the
violin/bar/scatter network colours. Integer label codes are kept identical to the
source atlas (SCAN=18, etc.) so the MPM is directly comparable to it.
"""

import sys, glob, warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.cifti2 import cifti2_axes
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import ATLAS_DIR, REPRO_DIR, DAT_DIR

# ── Configuration ──────────────────────────────────────────────────────────────
TIMEPOINT = 'baseline'
N_JOBS    = min(16, cpu_count())

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

# network -> integer label (same codes as the source atlas / compute_network_gradient)
NET_LABEL = {
    'DMN': 1, 'VIS': 2, 'FP': 3, 'DAN': 5, 'VAN': 7, 'SAL': 8, 'CO': 9,
    'SMD': 10, 'SML': 11, 'AUD': 12, 'Tpole': 13, 'MTL': 14, 'PMN': 15,
    'PON': 16, 'SCAN': 18,
}
# canonical atlas colours (hex) — mirror figsrc.ATLAS_NET_COLORS
NET_HEX = {
    'DMN': '#FF0000', 'VIS': '#000099', 'FP': '#FFFF00', 'DAN': '#00FF00',
    'VAN': '#0D85A0', 'SAL': '#000000', 'CO': '#6600CC', 'SMD': '#66FFFF',
    'SML': '#FF8000', 'AUD': '#B266FF', 'Tpole': '#006699', 'MTL': '#66FF66',
    'PMN': '#3C3CFB', 'PON': '#EFEFEF', 'SCAN': '#8E0067',
}
NETWORKS  = list(NET_LABEL.keys())
LABEL_INTS = np.array([NET_LABEL[n] for n in NETWORKS], dtype=np.int16)


def log(msg=''):
    print(msg, flush=True)


def hex2rgba(h):
    h = h.lstrip('#')
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255,
            int(h[4:6], 16) / 255, 1.0)


def load_atlas_bm():
    img = nib.load(str(ATLAS_PATH))
    bm  = img.header.get_axis(1)
    structs = {name: slc for name, slc, _ in bm.iter_structures()}
    return bm, structs.get('CIFTI_STRUCTURE_CORTEX_LEFT'), \
        structs.get('CIFTI_STRUCTURE_CORTEX_RIGHT')


def find_boldmap(sub_id, session):
    hits = glob.glob(str(REPRO_DIR / sub_id / f'ses-{session}' / 'func' / BOLDMAP_GLOB))
    return hits[0] if hits else None


def load_label_vec(args):
    """Return per-vertex int16 network-label vector (91282,) or None on failure."""
    sub_id, session = args
    fpath = find_boldmap(sub_id, session)
    if fpath is None:
        return None
    try:
        return nib.load(fpath).get_fdata()[0].astype(np.int16)
    except Exception:
        return None


def main():
    log('=' * 70)
    log('POPULATION MODAL (maximum-probability) NETWORK ATLAS')
    log(f'  Timepoint : {TIMEPOINT} (ses-{SESSION_MAP[TIMEPOINT]})')
    log(f'  Networks  : {len(NETWORKS)}  | N workers: {N_JOBS}')
    log('=' * 70)

    df = pd.read_csv(PROC_DIR / DF_MAP[TIMEPOINT])
    sub_ids = df['sub_ID'].tolist()
    log(f'\nSubjects in sample: {len(sub_ids)}')

    bm, l_slc, r_slc = load_atlas_bm()
    n_grayord = bm.size
    log(f'Atlas grayordinates: {n_grayord:,}')

    # ── stream per-subject label vectors, accumulate per-network vertex counts ──
    ses  = SESSION_MAP[TIMEPOINT]
    args = [(sid, ses) for sid in sub_ids]
    counts = np.zeros((len(NETWORKS), n_grayord), dtype=np.int32)
    n_ok = n_fail = 0
    log(f'\nAccumulating modal counts ({N_JOBS} workers)...')
    with Pool(N_JOBS) as pool:
        for v in pool.imap_unordered(load_label_vec, args, chunksize=8):
            if v is None:
                n_fail += 1
                continue
            for ki, L in enumerate(LABEL_INTS):
                counts[ki] += (v == L)
            n_ok += 1
            if n_ok % 500 == 0:
                log(f'  ...{n_ok} subjects')
    log(f'  Loaded: {n_ok} / {len(args)}  ({n_fail} failed/missing)')
    if n_ok == 0:
        log('ERROR: no subjects loaded'); sys.exit(1)

    # ── modal assignment ──────────────────────────────────────────────────────
    total = counts.sum(axis=0)                       # vertices assigned to ANY net
    valid = total > 0
    argmax = counts.argmax(axis=0)                   # index into NETWORKS
    modal_label = np.where(valid, LABEL_INTS[argmax], 0).astype(np.int32)

    # restrict to cortex (belt-and-suspenders; subcortex left unlabeled = 0)
    cortical = np.zeros(n_grayord, dtype=bool)
    cortical[l_slc] = True
    cortical[r_slc] = True
    modal_label[~cortical] = 0

    winfrac = np.divide(counts.max(axis=0), total, out=np.zeros(n_grayord),
                        where=valid).astype(np.float32)
    winfrac[~cortical] = np.nan

    # ── report ────────────────────────────────────────────────────────────────
    log('\nModal atlas composition (cortical vertices):')
    n_cort = int(cortical.sum())
    for n in NETWORKS:
        frac = np.mean(modal_label[cortical] == NET_LABEL[n])
        log(f'  {n:6s} (label {NET_LABEL[n]:2d}): {frac*100:5.2f}% of cortex')
    log(f'  unlabeled : {np.mean(modal_label[cortical] == 0)*100:5.2f}%')
    log(f'\nWinning-fraction (consensus) over cortex: '
        f'mean={np.nanmean(winfrac):.3f}, median={np.nanmedian(winfrac):.3f}, '
        f'min={np.nanmin(winfrac):.3f}')

    # ── write dlabel (MPM) with canonical colour table ────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_dict = {0: ('???', (0.0, 0.0, 0.0, 0.0))}
    for n in NETWORKS:
        label_dict[NET_LABEL[n]] = (n, hex2rgba(NET_HEX[n]))
    lax = cifti2_axes.LabelAxis(['modal_network'], [label_dict])
    hdr = nib.Cifti2Header.from_axes((lax, bm))
    img = nib.Cifti2Image(modal_label[np.newaxis, :].astype(np.float32), header=hdr)
    out_dlabel = OUT_DIR / f'population_modal_atlas_{TIMEPOINT}.dlabel.nii'
    nib.save(img, str(out_dlabel))
    log(f'\n  → {out_dlabel.name}')

    # ── write winning-fraction dscalar (free byproduct) ───────────────────────
    sax = cifti2_axes.ScalarAxis([f'winning_fraction_n{n_ok}'])
    hdr2 = nib.Cifti2Header.from_axes((sax, bm))
    img2 = nib.Cifti2Image(winfrac[np.newaxis, :].astype(np.float32), header=hdr2)
    out_wf = OUT_DIR / f'population_modal_atlas_winfrac_{TIMEPOINT}.dscalar.nii'
    nib.save(img2, str(out_wf))
    log(f'  → {out_wf.name}')

    log("""
Connectome Workbench:
  Load population_modal_atlas_baseline.dlabel.nii as a label layer — every cortical
  vertex is coloured by its modal network (colours already baked in, match the
  matplotlib panels). For an optional confidence-weighted look, drive layer opacity
  with population_modal_atlas_winfrac_baseline.dscalar.nii (vivid = high consensus).
""")
    log('Done.')


if __name__ == '__main__':
    main()
