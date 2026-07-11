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

import warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd

from adtopo.config import cfg
from adtopo.fc_utils import (
    MIN_FRAMES,
    BOLDMAP_GLOB, DTSERIES_GLOB, MOTION_GLOB,
    NET_NAMES, FC_COLS,
    log, find_file, load_fd_mask, load_network_labels,
    load_cortical_ts, compute_fc,
)

# ── constants ──────────────────────────────────────────────────────────────────
FC_DIR    = cfg.FC_DTSERIES_DIR

# Session → subject-list dataframe
SESSION_DF = {
    'ses-00A': 'df_base.csv',
    'ses-02A': 'df_y2.csv',
    'ses-04A': 'df_y4.csv',
    'ses-06A': 'df_y6.csv',
}


def process_subject(sub_id, session):
    func_repro = cfg.REPRO_DIR / sub_id / session / 'func'
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
        cort_ts = load_cortical_ts(dtseries_p, fd_mask, sub_id=sub_id, session=session)
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

    out_dir = cfg.DAT_DIR / f'fc_chunks_{session}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'fc_chunk_{args.chunk_idx:03d}.csv'

    log('=' * 60)
    log(f'FC COMPUTATION  session={session}  chunk {args.chunk_idx} of {args.n_chunks}')
    log('=' * 60)

    df      = pd.read_csv(cfg.DAT_DIR / df_name)
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
