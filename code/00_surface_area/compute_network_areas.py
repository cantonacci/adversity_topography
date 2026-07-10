#!/usr/bin/env python3
"""
Compute network surface area metrics for ABCD Template Matching outputs.

For each subject-session, reads:
  - REPRO_DIR/.../func/*_boldmap.dlabel.nii  (per-vertex network labels, 91k CIFTI)
  - XCP_DIR/.../anat/*midthickness*.surf.gii  (cortical surface geometry)

Outputs two CSVs per timepoint (in --out_dir):
  {session}_network_areas_cortical.csv        -- cortical surface area only (mm^2)
  {session}_network_areas_with_subcortical.csv -- cortical mm^2 + subcortical volume (mm^3)

Both include proportion columns ([network]_proportion = area / total_area).
Missing subjects (no dlabel or no surf.gii) appear as NaN rows.

Usage:
  python compute_network_areas.py --session ses-00A --n_jobs 16
"""

import os
import glob
import argparse
import traceback
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from multiprocessing import Pool

from adtopo.config import REPRO_DIR, XCP_DIR, DATA_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Label integer → CSV column name, confirmed from dlabel.nii label table.
# Labels 0, 4, 6, 17 are unlabeled/background and excluded.
# SCAN (18) does appear in subcortical structures (~97% of subjects; confirmed with MIDB).
LABEL_TO_NET = {
    1:  'DMN',
    2:  'VIS',
    3:  'FP',
    5:  'DAN',
    7:  'VAN',
    8:  'SAL',
    9:  'CO',
    10: 'SMD',
    11: 'SML',
    12: 'AUD',
    13: 'Tpole',
    14: 'MTL',
    15: 'PMN',
    16: 'PON',
    18: 'SCAN',
}

# Ordered network columns matching 2_baselinev6area.csv format
NETWORKS = ['DMN', 'VIS', 'FP', 'DAN', 'VAN', 'SAL', 'CO',
            'SMD', 'SML', 'AUD', 'Tpole', 'MTL', 'PMN', 'PON', 'SCAN']

# Subcortical voxel volume: 2x2x2 mm isotropic = 8 mm^3
VOXEL_VOLUME_MM3 = 8.0

DEFAULT_REPRO_DIR = str(REPRO_DIR)
DEFAULT_XCP_DIR   = str(XCP_DIR)
DEFAULT_OUT_DIR   = str(DATA_DIR / 'network_areas')

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_vertex_areas(surf_path):
    """Load midthickness surf.gii; return per-vertex area array (mm^2)."""
    gii = nib.load(surf_path)
    coords = gii.darrays[0].data.astype(np.float64)  # (N_verts, 3)
    faces  = gii.darrays[1].data.astype(np.int64)    # (N_faces, 3)
    v0 = coords[faces[:, 0]]
    v1 = coords[faces[:, 1]]
    v2 = coords[faces[:, 2]]
    cross     = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.sqrt(np.sum(cross ** 2, axis=1))
    vertex_areas = np.zeros(len(coords), dtype=np.float64)
    np.add.at(vertex_areas, faces[:, 0], tri_areas / 3.0)
    np.add.at(vertex_areas, faces[:, 1], tri_areas / 3.0)
    np.add.at(vertex_areas, faces[:, 2], tri_areas / 3.0)
    return vertex_areas


def process_subject_session(args):
    """
    Process one subject-session.

    Returns:
        (sub_id, cortical_areas, combined_areas, status_str)
        cortical_areas / combined_areas: dict {net: float} or None on failure
    """
    sub_id, session, repro_dir, xcp_dir = args
    try:
        # ---- locate dlabel.nii ------------------------------------------------
        func_dir = os.path.join(repro_dir, sub_id, session, 'func')
        dlabel_files = glob.glob(os.path.join(func_dir, '*_boldmap.dlabel.nii'))
        if not dlabel_files:
            return sub_id, None, None, 'missing_dlabel'
        dlabel_path = dlabel_files[0]

        # ---- locate surf.gii files (session-matched) -------------------------
        anat_dir = os.path.join(xcp_dir, sub_id, session, 'anat')
        if not os.path.isdir(anat_dir):
            return sub_id, None, None, 'missing_anat_dir'
        surf_L_files = sorted(glob.glob(os.path.join(anat_dir, '*hemi-L*midthickness*.surf.gii')))
        surf_R_files = sorted(glob.glob(os.path.join(anat_dir, '*hemi-R*midthickness*.surf.gii')))
        if not surf_L_files or not surf_R_files:
            return sub_id, None, None, 'missing_surf_gii'
        surf_L_path = surf_L_files[0]
        surf_R_path = surf_R_files[0]

        # ---- load CIFTI dlabel -----------------------------------------------
        img      = nib.load(dlabel_path)
        labels   = img.get_fdata(dtype=np.float32).squeeze().astype(np.int32)
        brain_ax = img.header.get_axis(1)  # BrainModelAxis

        # extract cortical structure slices and their surf.gii vertex indices
        left_slc = right_slc = None
        left_verts = right_verts = None
        for name, slc, bm in brain_ax.iter_structures():
            if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
                left_slc   = slc
                left_verts = bm.vertex   # surf.gii vertex index for each CIFTI position
            elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
                right_slc   = slc
                right_verts = bm.vertex

        if left_slc is None or right_slc is None:
            return sub_id, None, None, 'missing_cortex_structure'

        # ---- compute per-vertex areas from midthickness surfaces -------------
        vert_areas_L = compute_vertex_areas(surf_L_path)
        vert_areas_R = compute_vertex_areas(surf_R_path)

        # labels and areas aligned to CIFTI cortical positions
        labels_L  = labels[left_slc]
        labels_R  = labels[right_slc]
        cifti_areas_L = vert_areas_L[left_verts]   # area of each left-cortex CIFTI vertex
        cifti_areas_R = vert_areas_R[right_verts]  # area of each right-cortex CIFTI vertex

        # ---- subcortical labels (everything after right cortex ends) ---------
        subcortical_labels = labels[right_slc.stop:]

        # ---- aggregate per network -------------------------------------------
        cortical_areas = {}
        combined_areas = {}
        for label, net in LABEL_TO_NET.items():
            # cortical surface area (mm^2)
            cort_area = float(
                cifti_areas_L[labels_L == label].sum() +
                cifti_areas_R[labels_R == label].sum()
            )
            cortical_areas[net] = cort_area

            # subcortical volume contribution (n_voxels × 8 mm^3)
            sub_vol = float(np.sum(subcortical_labels == label)) * VOXEL_VOLUME_MM3
            combined_areas[net] = cort_area + sub_vol

        return sub_id, cortical_areas, combined_areas, 'ok'

    except Exception:
        return sub_id, None, None, f'error: {traceback.format_exc(limit=2)}'


# ---------------------------------------------------------------------------
# CSV assembly
# ---------------------------------------------------------------------------

def build_dataframe(ok_rows, nan_rows, network_cols, add_proportions=True):
    """
    ok_rows  : list of (sub_id, area_dict)
    nan_rows : list of (sub_id, reason_str)
    Returns a sorted DataFrame with area columns + optional proportion columns.
    """
    records = []
    for sub_id, areas in ok_rows:
        row = {'sub_ID': sub_id}
        row.update({net: areas[net] for net in network_cols})
        records.append(row)
    for sub_id, _ in nan_rows:
        row = {'sub_ID': sub_id}
        row.update({net: np.nan for net in network_cols})
        records.append(row)

    df = pd.DataFrame(records, columns=['sub_ID'] + network_cols)
    df = df.sort_values('sub_ID').reset_index(drop=True)

    if add_proportions:
        total = df[network_cols].sum(axis=1)
        for net in network_cols:
            df[f'{net}_proportion'] = df[net] / total

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute ABCD network surface areas from Template Matching outputs.')
    parser.add_argument('--session', required=True,
                        choices=['ses-00A', 'ses-02A', 'ses-04A', 'ses-06A'],
                        help='Timepoint to process')
    parser.add_argument('--repro_dir', default=DEFAULT_REPRO_DIR,
                        help='Path to REPRO_DIR directory')
    parser.add_argument('--xcp_dir', default=DEFAULT_XCP_DIR,
                        help='Path to XCP_DIR directory')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                        help='Output directory for CSVs')
    parser.add_argument('--n_jobs', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    args = parser.parse_args()

    session   = args.session
    repro_dir = args.repro_dir
    xcp_dir   = args.xcp_dir
    out_dir   = args.out_dir

    print(f'[compute_network_areas] Session: {session}')
    print(f'  ReproTM dir : {repro_dir}')
    print(f'  XCP-D dir   : {xcp_dir}')
    print(f'  Output dir  : {out_dir}')
    print(f'  Workers     : {args.n_jobs}')

    # ---- find all subjects with this session in either dataset ---------------
    def subs_with_session(base_dir):
        subs = set()
        try:
            for entry in os.scandir(base_dir):
                if entry.is_dir() and entry.name.startswith('sub-'):
                    if os.path.isdir(os.path.join(entry.path, session)):
                        subs.add(entry.name)
        except OSError as e:
            print(f'  Warning: could not scan {base_dir}: {e}')
        return subs

    repro_subs = subs_with_session(repro_dir)
    xcp_subs   = subs_with_session(xcp_dir)
    all_subs   = sorted(repro_subs | xcp_subs)

    print(f'\n  Found in ReproTM : {len(repro_subs):6d} subjects')
    print(f'  Found in XCP-D   : {len(xcp_subs):6d} subjects')
    print(f'  Union (total)    : {len(all_subs):6d} subjects')

    # ---- build task list and process -----------------------------------------
    tasks = [(sub, session, repro_dir, xcp_dir) for sub in all_subs]

    cortical_ok, combined_ok, nan_rows = [], [], []
    n_done = 0

    print(f'\nProcessing with {args.n_jobs} workers...')
    with Pool(args.n_jobs) as pool:
        for sub_id, cort, comb, status in pool.imap_unordered(
                process_subject_session, tasks, chunksize=20):
            n_done += 1
            if n_done % 500 == 0:
                print(f'  {n_done}/{len(tasks)} processed...')
            if status == 'ok':
                cortical_ok.append((sub_id, cort))
                combined_ok.append((sub_id, comb))
            else:
                nan_rows.append((sub_id, status))

    n_ok  = len(cortical_ok)
    n_nan = len(nan_rows)
    print(f'\nDone. OK: {n_ok}, NaN (missing/error): {n_nan}')

    # ---- write CSVs ----------------------------------------------------------
    session_tag = session.replace('ses-', '')

    df_cortical = build_dataframe(cortical_ok, nan_rows, NETWORKS)
    path_cortical = os.path.join(out_dir, f'{session_tag}_network_areas_cortical.csv')
    df_cortical.to_csv(path_cortical, index=False)
    print(f'Saved: {path_cortical}  ({len(df_cortical)} rows)')

    df_combined = build_dataframe(combined_ok, nan_rows, NETWORKS)
    path_combined = os.path.join(out_dir, f'{session_tag}_network_areas_with_subcortical.csv')
    df_combined.to_csv(path_combined, index=False)
    print(f'Saved: {path_combined}  ({len(df_combined)} rows)')

    # ---- write failure log ---------------------------------------------------
    if nan_rows:
        fail_path = os.path.join(out_dir, f'{session_tag}_network_areas_failures.txt')
        with open(fail_path, 'w') as f:
            for sub_id, reason in sorted(nan_rows):
                f.write(f'{sub_id}\t{reason}\n')
        print(f'Failure log: {fail_path}')


if __name__ == '__main__':
    main()
