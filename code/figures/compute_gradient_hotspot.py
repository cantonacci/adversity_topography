"""
Precompute the per-vertex data for Fig 2c (gradient localization), reusing the
EXACT loading/definitions from analysis_D_gradient_positioning.py so the figure
matches the reported stats. Emits a small CSV the panel script reads (keeps
fig2_panels.py free of the neuromaps dependency).

Output: outputs/tables/D_gradient_hotspot_vertices.csv
  columns: g1, g1_z, diff, gradient_pct, is_hotspot, is_scan_territory
  (one row per valid cortical vertex; hotspot = top-decile adversity SCAN gain)
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('NEUROMAPS_DATA', str(ROOT / 'data/neuromaps_cache'))
from neuromaps.datasets import fetch_annotation

DIFF = ROOT / 'outputs/cifti_for_workbench/SCAN_density_threat_baseline_diff.dscalar.nii'
DLAB = ROOT / 'data/atlas_files/abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii'
OUT  = ROOT / 'outputs/tables/D_gradient_hotspot_vertices.csv'
NVERT = 32492


def cifti_to_full(img, nvert=NVERT):
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
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata())[0].astype(int)
    lut = img.header.get_axis(0).label[0]
    scan_keys = [k for k, v in lut.items() if 'SCAN' in str(v[0]).upper()]
    bm = img.header.get_axis(1)
    full = {'CIFTI_STRUCTURE_CORTEX_LEFT': np.zeros(nvert, bool),
            'CIFTI_STRUCTURE_CORTEX_RIGHT': np.zeros(nvert, bool)}
    for name, slc, b in bm.iter_structures():
        if name in full:
            full[name][b.vertex] = np.isin(data[slc], scan_keys)
    return np.concatenate([full['CIFTI_STRUCTURE_CORTEX_LEFT'],
                           full['CIFTI_STRUCTURE_CORTEX_RIGHT']])


grad = fetch_annotation(source='margulies2016', desc='fcgradient01', space='fsLR', den='32k')
gL, gR = grad
g_full = np.concatenate([nib.load(str(gL)).darrays[0].data,
                         nib.load(str(gR)).darrays[0].data]).astype(float)
d_full = cifti_to_full(nib.load(str(DIFF)))
g_full[g_full == 0] = np.nan
valid = np.isfinite(g_full) & np.isfinite(d_full)

gv, dv = g_full[valid], d_full[valid]
g1_z = (gv - gv.mean()) / gv.std()
g_pct = stats.rankdata(gv) / len(gv) * 100
hot = dv >= np.percentile(dv, 90)
scan_terr = dlabel_scan_mask(DLAB)[valid]

df = pd.DataFrame({'g1': gv, 'g1_z': g1_z, 'diff': dv, 'gradient_pct': g_pct,
                   'is_hotspot': hot, 'is_scan_territory': scan_terr})
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f'wrote {OUT}  (n_valid={len(df)}, n_hotspot={int(hot.sum())}, '
      f'hotspot mean G1z={g1_z[hot].mean():+.3f}, hotspot mean pct={g_pct[hot].mean():.1f})')
