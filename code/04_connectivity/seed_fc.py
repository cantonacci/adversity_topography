#!/usr/bin/env python3
"""
gen_scan_seed_fc.py — vertexwise SCAN-seed FC, high vs low threat group means.

Fixed seed = GROUP-TEMPLATE SCAN (atlas label 18), identical for every subject, so
the analysis is NOT circular with the individual parcellation / encroachment (an
individually-expanded SCAN can't trivially inflate its own coupling).

For each high- (threat ≥ +1 SD) or low-threat (≤ −1 SD) subject:
  load denoisedSmoothed dtseries → FD<0.2 mask → cortical vertices
  seed ts = mean of group-template SCAN vertices
  vertexwise FC = corr(seed, every cortical vertex), Fisher-z
Accumulate group sums → high_mean, low_mean, diff (high − low) surfaces.

Outputs (outputs/cifti_for_workbench/):
  scan_seed_fc_groupmean.dscalar.nii   (3 maps: high, low, high_minus_low)
and (outputs/tables/): scan_seed_fc_groupmean.npz  (raw 59412-vectors + n)
"""
import sys, glob, warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import DAT_DIR, ATLAS_DIR, FC_DTSERIES_DIR, OUT_DIR, TAB_DIR

FD_THRESH, MIN_FRAMES, N_CORT, N_FULL = 0.2, 375, 59412, 91282
SCAN_LABEL, HIGH, LOW = 18, 1.0, -1.0
N_JOBS = min(16, cpu_count())
FC_DIR = FC_DTSERIES_DIR
ATLAS  = ATLAS_DIR / 'abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii'
OUT_C  = OUT_DIR / 'cifti_for_workbench'   # repo outputs/ (was code/outputs/ — stray path)
OUT_T  = TAB_DIR                            # repo outputs/tables
DTGLOB = '*_task-rest_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
MOGLOB = '*_task-rest_motion.tsv'

_atlas_img = nib.load(str(ATLAS))
_atlas = _atlas_img.get_fdata()[0].astype(int)[:N_CORT]
SEED_IDX = np.where(_atlas == SCAN_LABEL)[0]


def log(m): print(m, flush=True)


def _find(d, p):
    h = glob.glob(str(Path(d) / p))
    return h[0] if h else None


def seed_fc_map(args):
    """(sub, group) → (group, Fisher-z seed-FC vector (N_CORT,)) or None."""
    sub, group = args
    func = FC_DIR / sub / 'ses-00A' / 'func'
    dt, mo = _find(func, DTGLOB), _find(func, MOGLOB)
    if dt is None or mo is None:
        return None
    try:
        fd = pd.read_csv(mo, sep='\t')['framewise_displacement'].values.astype(float)
        m = (fd < FD_THRESH) & ~np.isnan(fd)
        if m.sum() < MIN_FRAMES:
            return None
        cort = nib.load(dt).get_fdata()[:, :N_CORT].astype(np.float32)
        nT = cort.shape[0]
        if nT == len(m):
            mm = m
        else:
            log(f'  WARNING frame/motion mismatch {sub}/ses-00A: '
                f'n_TRs={nT} n_mot={len(m)} — truncating/padding mask to n_TRs')
            mm = (m[:nT] if nT < len(m)
                  else np.concatenate([m, np.zeros(nT - len(m), bool)]))
        cort = cort[mm]
        if cort.shape[0] < MIN_FRAMES:
            return None
        seed = cort[:, SEED_IDX].mean(1)
        s = seed - seed.mean()
        ss = np.sqrt((s * s).sum())
        cz = cort - cort.mean(0)
        denom = np.sqrt((cz * cz).sum(0)) * ss
        with np.errstate(invalid='ignore', divide='ignore'):
            r = (cz * s[:, None]).sum(0) / denom
        np.clip(r, -0.9999, 0.9999, out=r)
        return (group, np.arctanh(r).astype(np.float32))
    except Exception as e:
        log(f'  SKIP {sub}: {repr(e)[:80]}')
        return None


def main():
    df = pd.read_csv(DAT_DIR / 'df_base.csv')[['sub_ID', 'threat_composite']].dropna()
    hi = df[df['threat_composite'] >= HIGH]['sub_ID'].tolist()
    lo = df[df['threat_composite'] <= LOW]['sub_ID'].tolist()
    log(f'SCAN-seed FC | seed vertices={len(SEED_IDX)} | high={len(hi)} low={len(lo)} | workers={N_JOBS}')
    args = [(s, 'high') for s in hi] + [(s, 'low') for s in lo]

    sums = {'high': np.zeros(N_CORT, np.float64), 'low': np.zeros(N_CORT, np.float64)}
    # per-vertex count of finite contributions (denominator for the group mean)
    counts = {'high': np.zeros(N_CORT, np.int64), 'low': np.zeros(N_CORT, np.int64)}
    ns = {'high': 0, 'low': 0}
    done = 0
    with Pool(N_JOBS) as pool:
        for res in pool.imap_unordered(seed_fc_map, args, chunksize=4):
            done += 1
            if done % 100 == 0:
                log(f'  {done}/{len(args)} (high n={ns["high"]}, low n={ns["low"]})')
            if res is None:
                continue
            grp, vec = res
            ok = np.isfinite(vec)
            sums[grp][ok] += vec[ok]
            counts[grp][ok] += 1
            ns[grp] += 1
    log(f'Loaded: high {ns["high"]}/{len(hi)}, low {ns["low"]}/{len(lo)}')

    def group_mean(grp):
        # divide each vertex's sum by its per-vertex finite count; where a vertex
        # got zero finite contributions -> NaN (avoid divide-by-zero).
        c = counts[grp]
        out = np.full(N_CORT, np.nan, np.float64)
        nz = c > 0
        out[nz] = sums[grp][nz] / c[nz]
        return out.astype(np.float32)

    high_mean = group_mean('high')
    low_mean = group_mean('low')
    diff = (high_mean - low_mean).astype(np.float32)
    # seed region itself → NaN (coupling to the rest of cortex)
    for arr in (high_mean, low_mean, diff):
        arr[SEED_IDX] = np.nan

    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_C.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_T / 'scan_seed_fc_groupmean.npz', high=high_mean, low=low_mean,
             diff=diff, n_high=ns['high'], n_low=ns['low'], seed_idx=SEED_IDX)

    bm = _atlas_img.header.get_axis(1)

    def to_full(v):
        full = np.full(N_FULL, np.nan, np.float32)
        full[:N_CORT] = v
        return full
    data = np.stack([to_full(high_mean), to_full(low_mean), to_full(diff)], 0)
    sax = nib.cifti2.ScalarAxis(['SCANseedFC_high', 'SCANseedFC_low', 'SCANseedFC_high_minus_low'])
    hdr = nib.Cifti2Header.from_axes((sax, bm))
    nib.save(nib.Cifti2Image(data, hdr), str(OUT_C / 'scan_seed_fc_groupmean.dscalar.nii'))
    log(f'wrote scan_seed_fc_groupmean.dscalar.nii  diff range [{np.nanmin(diff):+.3f},{np.nanmax(diff):+.3f}]')
    log('Done.')


if __name__ == '__main__':
    main()
