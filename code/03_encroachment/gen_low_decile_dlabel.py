# -*- coding: utf-8 -*-
"""Fill the one missing SCAN extent map: the bottom-decile (p10p90 low) thresholded
SCAN dlabel — the twin of SCAN_dlabel_high_adversity_p10p90_t25_baseline that
generate_scan_borders.py never produced (its low branch runs only at ≤ -1 SD).

Input already on disk: SCAN_density_threat_baseline_p10p90_low.dscalar.nii
Output: SCAN_dlabel_low_adversity_p10p90_t25_baseline.dlabel.nii (+ magenta border)

Reuses the helpers in generate_scan_borders so format/label/colour match the twin.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import nibabel as nib
import generate_scan_borders as G

# generate_scan_borders.OUT_DIR resolves to code/outputs (a stray-path artefact);
# the real CIFTI live in the repo outputs/. Point the module at the correct dir.
REPO = Path(__file__).resolve().parents[2]
G.OUT_DIR    = REPO / 'outputs' / 'cifti_for_workbench'
G.BORDER_DIR = G.OUT_DIR / 'borders'

THRESH  = 0.25
thr_tag = f't{int(THRESH * 100):02d}'

# BrainModelAxis + cortical mask, identical to generate_scan_borders.main()
atlas_img = nib.load(str(G.ATLAS_PATH))
bm_ax     = atlas_img.header.get_axis(1)
cort_mask = np.zeros(G.N_FULL, dtype=bool)
for name, slc, _ in bm_ax.iter_structures():
    if 'CORTEX' in name:
        cort_mask[slc] = True
G.log(f'cortical vertices: {cort_mask.sum():,}')

low_src = G.OUT_DIR / 'SCAN_density_threat_baseline_p10p90_low.dscalar.nii'
if not low_src.exists():
    raise SystemExit(f'missing required input: {low_src}')

out = G.OUT_DIR / f'SCAN_dlabel_low_adversity_p10p90_{thr_tag}_baseline.dlabel.nii'
G.log(f'Building {out.name} from {low_src.name} at P(SCAN) >= {THRESH}')
G.save_dlabel(
    G.dscalar_to_scan_dlabel(low_src, cort_mask, bm_ax, 'SCAN_low_p10p90', THRESH),
    out,
)

# matching magenta border (twin of SCAN_border_high_adversity_p10p90_t25_*)
G.generate_borders(out, f'SCAN_border_low_adversity_p10p90_{thr_tag}')
G.log('DONE — bottom-decile SCAN extent dlabel + border written')
