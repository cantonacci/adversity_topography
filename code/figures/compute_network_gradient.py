"""
Per-network mean position on the principal cortical gradient (Margulies 2016 G1,
the sensorimotor→association / S-A axis), fsLR 32k. Used to order the Fig 1a
network columns along the S-A axis and to draw the axis annotation.

G1 low  = unimodal / sensorimotor pole
G1 high = transmodal / association pole

Output: outputs/tables/network_gradient_G1.csv  (network, meanG1, sd, n_vert),
sorted ascending (sensorimotor → association).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("NEUROMAPS_DATA", str(ROOT / "data/neuromaps_cache"))
from neuromaps.datasets import fetch_annotation

DLAB = ROOT / "data/atlas_files/abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii"
NVERT = 32492
NET_MAP = {  # network -> atlas label integer (from brain_surface_figures.py)
    "DMN": 1, "VIS": 2, "FP": 3, "DAN": 5, "VAN": 7,
    "SAL": 8, "CO": 9, "SMD": 10, "SML": 11, "AUD": 12,
    "Tpole": 13, "MTL": 14, "PMN": 15, "PON": 16, "SCAN": 18,
}

# principal gradient (two fsLR 32k gii hemispheres)
grad = fetch_annotation(source="margulies2016", desc="fcgradient01",
                        space="fsLR", den="32k")
gL, gR = grad
g_full = np.concatenate([nib.load(str(gL)).darrays[0].data,
                         nib.load(str(gR)).darrays[0].data]).astype(float)
g_full[g_full == 0] = np.nan   # fsLR medial-wall convention

# atlas labels onto the full L+R 32k vertex array
img = nib.load(str(DLAB))
data = np.asarray(img.get_fdata())[0].astype(int)
bm = img.header.get_axis(1)
full = {"CIFTI_STRUCTURE_CORTEX_LEFT": np.full(NVERT, -1, int),
        "CIFTI_STRUCTURE_CORTEX_RIGHT": np.full(NVERT, -1, int)}
for name, slc, b in bm.iter_structures():
    if name in full:
        full[name][b.vertex] = data[slc]
lab = np.concatenate([full["CIFTI_STRUCTURE_CORTEX_LEFT"],
                      full["CIFTI_STRUCTURE_CORTEX_RIGHT"]])

rows = []
for net, key in NET_MAP.items():
    m = (lab == key) & np.isfinite(g_full)
    rows.append({"network": net, "meanG1": float(np.nanmean(g_full[m])),
                 "sd": float(np.nanstd(g_full[m])), "n_vert": int(m.sum())})
df = pd.DataFrame(rows).sort_values("meanG1").reset_index(drop=True)
df.to_csv(ROOT / "outputs/tables/network_gradient_G1.csv", index=False)
print(df.to_string(index=False))
print("\nSensorimotor → Association order:",
      " ".join(df["network"].tolist()))
