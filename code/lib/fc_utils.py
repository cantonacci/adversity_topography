#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fc_utils.py — shared helpers for subject-level 15-network FC computation.

Extracted verbatim from compute_fc.py / compute_fc_baseline.py, which were
~95% identical. Both scripts now import these functions instead of defining
them locally; the only legitimate script-level differences (compute_fc.py's
--session handling vs compute_fc_baseline.py hardcoding ses-00A, and their
output paths) remain in the scripts themselves.

Requires `config.py` to be importable (scripts do sys.path.insert of the code
dir before importing lib.fc_utils).
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

from config import NETWORKS

# ── constants (identical across both FC scripts) ─────────────────────────────
FD_THRESH  = 0.2
MIN_FRAMES = 375   # 5 min at TR=0.8s
N_CORT     = 59412
N_FULL     = 91282

BOLDMAP_GLOB  = (
    '*_task-rest_space-fsLR_den-91k_desc-denoised-spatially-interpolated-'
    'smoothed-2.25mm-censor-ReproTM_template-ABCC-a3-9to16_refine-SCAN_'
    'minsize-30_boldmap.dlabel.nii'
)
DTSERIES_GLOB = '*_task-rest_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii'
MOTION_GLOB   = '*_task-rest_motion.tsv'

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


def load_cortical_ts(dtseries_path, fd_mask, sub_id=None, session=None):
    img    = nib.load(str(dtseries_path))
    data   = img.get_fdata()
    cort   = data[:, :N_CORT].astype(np.float32)
    n_TRs  = cort.shape[0]
    n_mot  = len(fd_mask)

    if n_TRs == n_mot:
        mask = fd_mask
    else:
        who = f'{sub_id or "?"}/{session or "?"}'
        log(f'  WARNING frame/motion mismatch {who}: '
            f'n_TRs={n_TRs} n_mot={n_mot} — truncating/padding mask to n_TRs')
        if n_TRs < n_mot:
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
