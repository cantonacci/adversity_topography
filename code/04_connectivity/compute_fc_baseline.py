#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_fc_baseline.py  —  Subject-level 15-network FC matrices from XCP-D data.

For each subject in df_base.csv:
  1. Load boldmap (REPRO_DIR) for vertex-to-network assignment
  2. Load motion.tsv -> FD<0.2 mask, n_usable_frames, mean_FD
  3. Load denoisedSmoothed dtseries -> cortical vertices -> apply FD mask
  4. Compute 15 mean network timeseries -> Pearson r -> Fisher-z
  5. Output row to per-chunk CSV

Columns: sub_ID, n_usable_frames, n_total_frames, mean_FD,
         fc_{N1}_{N2} x120  (upper triangle incl. diagonal; diagonal=NaN)

Usage:
  python compute_fc_baseline.py --chunk-idx 0 --n-chunks 20
"""

import sys, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import DAT_DIR, REPRO_DIR, FC_DTSERIES_DIR
from lib.fc_utils import (
    MIN_FRAMES,
    BOLDMAP_GLOB, DTSERIES_GLOB, MOTION_GLOB,
    NET_NAMES, FC_COLS,
    log, find_file, load_fd_mask, load_network_labels,
    load_cortical_ts, compute_fc,
)

# ── constants ──────────────────────────────────────────────────────────────────
FC_DIR    = FC_DTSERIES_DIR
OUT_DIR   = Path(__file__).parent.parent / 'outputs' / 'data_processed' / 'fc_chunks'


def process_subject(sub_id):
    session    = 'ses-00A'
    func_repro = REPRO_DIR / sub_id / 'ses-00A' / 'func'
    func_fc    = FC_DIR    / sub_id / 'ses-00A' / 'func'

    boldmap_p  = find_file(func_repro, BOLDMAP_GLOB)
    dtseries_p = find_file(func_fc,    DTSERIES_GLOB)
    motion_p   = find_file(func_fc,    MOTION_GLOB)

    if boldmap_p is None:
        log(f'  SKIP {sub_id}: boldmap not found')
        return None
    if dtseries_p is None:
        log(f'  SKIP {sub_id}: dtseries not found')
        return None
    if motion_p is None:
        log(f'  SKIP {sub_id}: motion not found')
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
        'n_usable_frames': n_usable,
        'n_total_frames':  len(fd_mask),
        'mean_FD':         round(mean_fd, 5),
    }
    for col, val in zip(FC_COLS, fc_flat):
        row[col] = float(val) if not np.isnan(val) else np.nan

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk-idx', type=int, required=True)
    parser.add_argument('--n-chunks',  type=int, default=20)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'fc_chunk_{args.chunk_idx:03d}.csv'

    log('=' * 60)
    log(f'FC COMPUTATION  chunk {args.chunk_idx} of {args.n_chunks}')
    log('=' * 60)

    df      = pd.read_csv(DAT_DIR / 'df_base.csv')
    sub_ids = df['sub_ID'].tolist()
    chunks  = np.array_split(sub_ids, args.n_chunks)
    chunk   = list(chunks[args.chunk_idx])
    log(f'Subjects: {len(chunk)}  ({chunk[0]} .. {chunk[-1]})')

    rows = []
    for i, sub_id in enumerate(chunk):
        if i % 25 == 0:
            log(f'  [{i}/{len(chunk)}] {sub_id}')
        row = process_subject(sub_id)
        if row is not None:
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    log(f'\nSaved {len(rows)} subjects → {out_path}')
    log('Done.')


if __name__ == '__main__':
    main()
