#!/usr/bin/env python3
"""
Analysis D — positioning SCAN's adversity-driven expansion on the macroscale
sensorimotor->association cortical gradient (Margulies et al. 2016, principal
functional gradient, fsLR 32k).

Question: does threat drive SCAN to expand *down* the cortical hierarchy, toward the
unimodal/sensorimotor end of the principal gradient?

Vertexwise map of the adversity effect = SCAN_density_threat_baseline_diff.dscalar.nii
(high-threat minus low-threat probability that a vertex is labelled SCAN). We correlate
this with the principal gradient G1 across cortical vertices, with spin-test inference
(Alexander-Bloch spatial null) using neuromaps.

Outputs:
  outputs/tables/D_gradient_positioning_summary.txt
"""
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('NEUROMAPS_DATA', str(ROOT / 'data/neuromaps_cache'))
(ROOT / 'data/neuromaps_cache').mkdir(parents=True, exist_ok=True)

from neuromaps.datasets import fetch_annotation
from neuromaps.nulls import alexander_bloch
from neuromaps.stats import compare_images

DIFF  = ROOT / 'outputs/cifti_for_workbench/SCAN_density_threat_baseline_diff.dscalar.nii'
DLAB  = ROOT / 'data/atlas_files/abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii'
OUT   = ROOT / 'outputs/tables/D_gradient_positioning_summary.txt'
NVERT = 32492  # per hemi, fsLR 32k
L = []
def log(s=''): print(s); L.append(s)


def cifti_to_full(img, nvert=NVERT):
    """Map a CIFTI dscalar's cortical data onto full L+R 32k vertex arrays (medial wall = NaN)."""
    data = np.asarray(img.get_fdata())[0]
    ax = img.header.get_axis(1)
    full = {'CIFTI_STRUCTURE_CORTEX_LEFT': np.full(nvert, np.nan),
            'CIFTI_STRUCTURE_CORTEX_RIGHT': np.full(nvert, np.nan)}
    for name, slc, bm in ax.iter_structures():
        if name in full:
            full[name][bm.vertex] = data[slc]
    return np.concatenate([full['CIFTI_STRUCTURE_CORTEX_LEFT'],
                           full['CIFTI_STRUCTURE_CORTEX_RIGHT']])


def dlabel_scan_mask(path, nvert=NVERT):
    """Boolean mask (L+R 32k) of vertices labelled SCAN in the group template."""
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata())[0].astype(int)
    ax = img.header.get_axis(0)
    lut = ax.label[0]  # {key: (name, rgba)}
    scan_keys = [k for k, v in lut.items() if 'SCAN' in str(v[0]).upper()]
    bm = img.header.get_axis(1)
    full = {'CIFTI_STRUCTURE_CORTEX_LEFT': np.zeros(nvert, bool),
            'CIFTI_STRUCTURE_CORTEX_RIGHT': np.zeros(nvert, bool)}
    for name, slc, b in bm.iter_structures():
        if name in full:
            vals = data[slc]
            full[name][b.vertex] = np.isin(vals, scan_keys)
    return np.concatenate([full['CIFTI_STRUCTURE_CORTEX_LEFT'],
                           full['CIFTI_STRUCTURE_CORTEX_RIGHT']]), scan_keys


log('Analysis D — SCAN adversity-expansion on the principal cortical gradient')
log('')

# principal gradient (G1) -----------------------------------------------------
grad = fetch_annotation(source='margulies2016', desc='fcgradient01', space='fsLR', den='32k')
gL, gR = (grad if isinstance(grad, (list, tuple)) else (grad,))[:2] if isinstance(grad, (list, tuple)) else (grad[0], grad[1])
g_full = np.concatenate([nib.load(str(gL)).darrays[0].data,
                         nib.load(str(gR)).darrays[0].data]).astype(float)

# adversity-driven SCAN probability change -----------------------------------
d_full = cifti_to_full(nib.load(str(DIFF)))

# medial wall: vertices absent from the dscalar are NaN -> drives the valid mask.
# also treat exact-zero gradient (fsLR medial wall convention) as missing.
g_full[g_full == 0] = np.nan
valid = np.isfinite(g_full) & np.isfinite(d_full)
log(f'valid cortical vertices: {valid.sum()} of {len(valid)}')
log(f'diff map sign check: mean(high-low) over SCAN-gaining vertices should be >0; '
    f'overall mean diff = {np.nanmean(d_full):+.5f}, max = {np.nanmax(d_full):+.4f}')
log('')

gv, dv = g_full[valid], d_full[valid]
r_p, _ = stats.pearsonr(gv, dv)
r_s, _ = stats.spearmanr(gv, dv)
log(f'Correlation of principal gradient (G1) with adversity-driven SCAN change:')
log(f'  Pearson  r = {r_p:+.3f}')
log(f'  Spearman r = {r_s:+.3f}')
log('  (G1 low = unimodal/sensorimotor, high = transmodal/association;')
log('   NEGATIVE r => SCAN expands toward the sensorimotor end under threat.)')
log('')

# spin-test inference (Alexander-Bloch) --------------------------------------
log('Spin-test inference (Alexander-Bloch spatial null, 1000 rotations)...')
nulls = alexander_bloch(g_full, atlas='fsLR', density='32k', n_perm=1000, seed=1234)
r_spin, p_spin = compare_images(g_full, d_full, nulls=nulls, metric='pearsonr')
log(f'  whole-map gradient x expansion correlation: spin-tested r = {r_spin:+.3f}, p_spin = {p_spin:.4g}')
log('')

# descriptive: where does the expansion sit on the gradient? ------------------
g_pct = stats.rankdata(gv) / len(gv) * 100  # gradient percentile among valid vertices
thr_hi = np.nanpercentile(dv, 90)
hot = dv >= thr_hi   # top-decile adversity-driven SCAN gain ("encroachment hotspot")
log('Gradient position (percentile of G1, 0=sensorimotor pole, 100=association pole):')
log(f'  whole cortex                         : {np.mean(g_pct):.1f}')
log(f'  adversity SCAN-gain hotspot (top 10%): {np.mean(g_pct[hot]):.1f}')

# spin test for the hotspot localization (reuse rotations) --------------------
# Is the hotspot's mean gradient lower than expected under the spatial null?
valid_idx = np.where(valid)[0]
hot_full  = valid_idx[hot]
gz = (g_full - np.nanmean(g_full)) / np.nanstd(g_full)
obs_hot = np.nanmean(gz[hot_full])
nulls_z = (nulls - np.nanmean(g_full)) / np.nanstd(g_full)
null_hot = np.nanmean(nulls_z[hot_full, :], axis=0)
p_hot = (np.sum(null_hot <= obs_hot) + 1) / (nulls.shape[1] + 1)
log(f'  hotspot localization spin test: observed mean G1(z) at hotspot = {obs_hot:+.3f}, '
    f'null mean = {np.mean(null_hot):+.3f}, p_spin(lower) = {p_hot:.4g}')
try:
    scan_mask, scan_keys = dlabel_scan_mask(DLAB)
    sm_valid = scan_mask[valid]
    if sm_valid.sum() > 0:
        log(f'  SCAN template territory               : {np.mean(g_pct[sm_valid]):.1f}  '
            f'(n_vert={int(sm_valid.sum())}, label keys={scan_keys})')
        log('  -> hotspot percentile < SCAN-territory percentile would mean SCAN expands')
        log('     DOWNHILL (further toward sensorimotor) than its own normative territory.')
except Exception as e:
    log(f'  (SCAN template territory: could not parse dlabel: {repr(e)[:100]})')

OUT.write_text('\n'.join(L))
log('')
log(f'wrote {OUT}')
