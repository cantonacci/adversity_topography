#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_fc.py  —  Subject-level 15-network FC matrices from XCP-D data.

Works for any ABCD timepoint by accepting --session and deriving the subject
list from the corresponding topography dataframe.

Session → dataframe mapping:
  ses-00A → df_base.csv   (~4,525 subjects)
  ses-02A → df_y2.csv     (~4,347 subjects)
  ses-04A → df_y4.csv     (~3,983 subjects)
  ses-06A → df_y6.csv     (~2,687 subjects)

For each subject:
  1. Load boldmap (REPRO_DIR, session-matched) → vertex-to-network labels
  2. Load motion.tsv → FD<0.2 mask, n_usable_frames, mean_FD
  3. Load denoisedSmoothed dtseries → cortical vertices → apply FD mask
  4. Compute 15 mean network timeseries → Pearson r → Fisher-z
  5. Write row to per-chunk CSV

Columns: sub_ID, session, n_usable_frames, n_total_frames, mean_FD,
         fc_{N1}_{N2} x120  (upper triangle incl. diagonal; diagonal=NaN)

Usage:
  python compute_fc.py --session ses-02A --chunk-idx 0 --n-chunks 20
  python compute_fc.py --session ses-00A --chunk-idx 0 --n-chunks 20  # baseline
"""

import sys, glob, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import DAT_DIR, NETWORKS, REPRO_DIR, FC_DTSERIES_DIR

# ── constants ──────────────────────────────────────────────────────────────────
FD_THRESH  = 0.2
MIN_FRAMES = 375   # 5 min at TR=0.8s
N_CORT     = 59412
N_FULL     = 91282

FC_DIR    = FC_DTSERIES_DIR

BOLDMAP_GLOB  = (
    '*_task-rest_space-fsLR_den-91k_desc-denoised-spatially-interpolated-'
    'smoothed-2.25mm-censor-ReproTM_template-ABCC-a3-9to16_refine-SCAN_'
    'minsize-30_boldmap.dlabel.nii'
)
DTSERIES_GLOB = '*_task-rest_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
MOTION_GLOB   = '*_task-rest_motion.tsv'

# Session → subject-list dataframe
SESSION_DF = {
    'ses-00A': 'df_base.csv',
    'ses-02A': 'df_y2.csv',
    'ses-04A': 'df_y4.csv',
    'ses-06A': 'df_y6.csv',
}

NET_LABEL = {
    'DMN': 1, 'VIS': 2, 'FP': 3, 'DAN': 5, 'VAN': 7,
    'SAL': 8, 'CO': 9, 'SMD': 10, 'SML': 11, 'AUD': 12,
    'Tpole': 13, 'MTL': 14, 'PMN': 15, 'PON': 16, 'SCAN': 18,
}
NET_NAMES = list(NETWORKS)  # 15 networks, canonical order

FC_COLS = [
    f'fc_{NET_NAMES[i]}_{NET_NAMES[j]}'
    for i in range(len(NET_NAMES))
    for j in range(i, len(NET_NAMES))
]  # 120 pairs: upper triangle including diagonal (diagonal will be NaN)


def log(msg): print(msg, flush=True)


def find_file(directory, pattern):
    hits = glob.glob(str(Path(directory) / pattern))
    return Path(hits[0]) if hits else None


def load_fd_mask(motion_path):
    mot = pd.read_csv(motion_path, sep='\t')
    fd  = mot['framewise_displacement'].values.astype(float)
    mask = (fd < FD_THRESH) & ~np.isnan(fd)
    return mask, int(mask.sum()), float(np.nanmean(fd))


def load_network_labels(boldmap_path):
    img  = nib.load(str(boldmap_path))
    data = img.get_fdata()[0].astype(np.int16)
    return data[:N_CORT]


def load_cortical_ts(dtseries_path, fd_mask):
    img    = nib.load(str(dtseries_path))
    data   = img.get_fdata()
    cort   = data[:, :N_CORT].astype(np.float32)
    n_TRs  = cort.shape[0]
    n_mot  = len(fd_mask)

    if n_TRs == n_mot:
        mask = fd_mask
    elif n_TRs < n_mot:
        mask = fd_mask[:n_TRs]
    else:
        mask = np.concatenate([fd_mask, np.zeros(n_TRs - n_mot, dtype=bool)])

    return cort[mask]


def compute_fc(cort_ts, net_labels):
    """
    (n_frames, N_CORT) + (N_CORT,) labels → (15, 15) Fisher-z FC matrix.
    Diagonal = NaN (autocorrelation of mean timeseries is not meaningful FC).
    """
    n = len(NET_NAMES)
    mean_ts = np.full((n, cort_ts.shape[0]), np.nan, dtype=np.float32)
    for k, net in enumerate(NET_NAMES):
        idx = np.where(net_labels == NET_LABEL[net])[0]
        if len(idx) > 0:
            mean_ts[k] = cort_ts[:, idx].mean(axis=1)

    valid = np.where(~np.any(np.isnan(mean_ts), axis=1))[0]
    fc    = np.full((n, n), np.nan, dtype=np.float32)

    if len(valid) >= 2:
        ts_v = mean_ts[valid]
        corr = np.corrcoef(ts_v).astype(np.float32)
        np.clip(corr, -0.9999, 0.9999, out=corr)
        np.fill_diagonal(corr, np.nan)  # diagonal not meaningful
        fz = np.where(np.isnan(corr), np.nan, np.arctanh(corr))
        for a, i in enumerate(valid):
            for b, j in enumerate(valid):
                fc[i, j] = fz[a, b]

    return fc


def process_subject(sub_id, session):
    func_repro = REPRO_DIR / sub_id / session / 'func'
    func_fc    = FC_DIR    / sub_id / session / 'func'

    boldmap_p  = find_file(func_repro, BOLDMAP_GLOB)
    dtseries_p = find_file(func_fc,    DTSERIES_GLOB)
    motion_p   = find_file(func_fc,    MOTION_GLOB)

    if boldmap_p is None:
        log(f'  SKIP {sub_id}: boldmap not found in {func_repro}')
        return None
    if dtseries_p is None:
        log(f'  SKIP {sub_id}: dtseries not found in {func_fc}')
        return None
    if motion_p is None:
        log(f'  SKIP {sub_id}: motion not found in {func_fc}')
        return None

    try:
        fd_mask, n_usable, mean_fd = load_fd_mask(motion_p)
    except Exception as e:
        log(f'  SKIP {sub_id}: motion error — {e}')
        return None

    if n_usable < MIN_FRAMES:
        log(f'  SKIP {sub_id}: {n_usable} usable frames < {MIN_FRAMES}')
        return None

    try:
        net_labels = load_network_labels(boldmap_p)
    except Exception as e:
        log(f'  SKIP {sub_id}: boldmap error — {e}')
        return None

    try:
        cort_ts = load_cortical_ts(dtseries_p, fd_mask)
    except Exception as e:
        log(f'  SKIP {sub_id}: dtseries error — {e}')
        return None

    if cort_ts.shape[0] < MIN_FRAMES:
        log(f'  SKIP {sub_id}: {cort_ts.shape[0]} frames after mask < {MIN_FRAMES}')
        return None

    fc_mat  = compute_fc(cort_ts, net_labels)
    fc_flat = [fc_mat[i, j] for i in range(len(NET_NAMES)) for j in range(i, len(NET_NAMES))]

    row = {
        'sub_ID':          sub_id,
        'session':         session,
        'n_usable_frames': n_usable,
        'n_total_frames':  len(fd_mask),
        'mean_FD':         round(mean_fd, 5),
    }
    for col, val in zip(FC_COLS, fc_flat):
        row[col] = float(val) if not np.isnan(val) else np.nan

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session',   type=str, required=True,
                        choices=list(SESSION_DF.keys()),
                        help='BIDS session label (e.g. ses-02A)')
    parser.add_argument('--chunk-idx', type=int, required=True)
    parser.add_argument('--n-chunks',  type=int, default=20)
    args = parser.parse_args()

    session = args.session
    df_name = SESSION_DF[session]

    out_dir = DAT_DIR / f'fc_chunks_{session}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'fc_chunk_{args.chunk_idx:03d}.csv'

    log('=' * 60)
    log(f'FC COMPUTATION  session={session}  chunk {args.chunk_idx} of {args.n_chunks}')
    log('=' * 60)

    df      = pd.read_csv(DAT_DIR / df_name)
    sub_ids = df['sub_ID'].tolist()
    chunks  = np.array_split(sub_ids, args.n_chunks)
    chunk   = list(chunks[args.chunk_idx])
    log(f'Subjects: {len(chunk)}  ({chunk[0]} .. {chunk[-1]})')

    rows = []
    for i, sub_id in enumerate(chunk):
        if i % 25 == 0:
            log(f'  [{i}/{len(chunk)}] {sub_id}')
        row = process_subject(sub_id, session)
        if row is not None:
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    log(f'\nSaved {len(rows)} subjects → {out_path}')
    log('Done.')


if __name__ == '__main__':
    main()
