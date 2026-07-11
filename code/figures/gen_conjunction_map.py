"""
Conjunction map: where adversity-driven SCAN EXPANSION (encroachment) and
adversity-driven SCAN COUPLING GAIN (seed FC) co-localise, vertexwise.

Self-contained pipeline (raw diff maps -> cleaned, coloured dlabel):
  1 = FC-gain only   (top-decile coupling increase, NOT top-decile expansion)
  2 = expansion only (top-decile expansion,          NOT top-decile FC gain)
  3 = OVERLAP        (top-decile of BOTH)            <- the headline (SCAN magenta)

Top decile = >= 90th percentile of each signed diff map among vertices valid in
both maps (same vertex set the spin test used: r_spin = 0.69, p = 0.001).
Each class is then cluster-cleaned by true surface area on the fsLR midthickness
(drop clusters < MIN_AREA mm^2). The two "only" classes render as subdued Wong
tints so the magenta overlap stays the foreground.

Output:
  outputs/cifti_for_workbench/scan_structfunc_conjunction_clean.dscalar.nii  (codes 0-3)
  outputs/cifti_for_workbench/scan_structfunc_conjunction_clean.dlabel.nii   (coloured)
"""
import numpy as np
import nibabel as nib
from pathlib import Path
from nibabel.cifti2 import cifti2_axes
from scipy.stats import hypergeom
import subprocess, tempfile, os

ROOT = Path(__file__).resolve().parents[2]
ENC = ROOT / 'outputs/cifti_for_workbench/SCAN_density_threat_baseline_diff.dscalar.nii'
FC  = ROOT / 'outputs/cifti_for_workbench/scan_seed_fc_groupmean.dscalar.nii'
OUTD = ROOT / 'outputs/cifti_for_workbench'
SURF_L = ROOT / 'data/neuromaps_cache/atlases/fsLR/tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii'
SURF_R = ROOT / 'data/neuromaps_cache/atlases/fsLR/tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii'
CLEAN_DSCALAR = OUTD / 'scan_structfunc_conjunction_clean.dscalar.nii'
CLEAN_DLABEL  = OUTD / 'scan_structfunc_conjunction_clean.dlabel.nii'
MIN_AREA = 50.0   # mm^2; drop clusters smaller than this (per class)
NVERT = 32492

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print('CMD FAILED:', ' '.join(map(str, cmd))); print(r.stdout, r.stderr); raise SystemExit(1)
    return r

def cifti_to_full(img, mapidx=0):
    data = np.asarray(img.get_fdata())[mapidx]
    ax = img.header.get_axis(1)
    full = {'CIFTI_STRUCTURE_CORTEX_LEFT': np.full(NVERT, np.nan),
            'CIFTI_STRUCTURE_CORTEX_RIGHT': np.full(NVERT, np.nan)}
    for name, slc, bm in ax.iter_structures():
        if name in full:
            full[name][bm.vertex] = data[slc]
    return np.concatenate([full['CIFTI_STRUCTURE_CORTEX_LEFT'],
                           full['CIFTI_STRUCTURE_CORTEX_RIGHT']])

def main():
    # ---- compute the 3-class codes from the raw signed diff maps ----
    enc_img = nib.load(str(ENC)); fc_img = nib.load(str(FC))
    enc = cifti_to_full(enc_img, 0)   # high-low SCAN density (expansion)
    fc  = cifti_to_full(fc_img, 2)    # high-low SCAN-seed FC (map 3)
    valid = np.isfinite(enc) & np.isfinite(fc)
    n = valid.sum()
    hi_enc = valid & (enc >= np.percentile(enc[valid], 90))
    hi_fc  = valid & (fc  >= np.percentile(fc[valid], 90))
    overlap = hi_enc & hi_fc

    code_full = np.zeros(enc.shape[0], dtype=np.float32)
    code_full[hi_fc & ~hi_enc] = 1
    code_full[hi_enc & ~hi_fc] = 2
    code_full[overlap] = 3

    n_enc, n_fc, n_ov = int(hi_enc.sum()), int(hi_fc.sum()), int(overlap.sum())
    exp_chance = n_fc * (n_enc / n)
    print(f'valid vertices (both maps): {n}')
    print(f'top-decile expansion / FC-gain / overlap: {n_enc} / {n_fc} / {n_ov}')
    print(f'  overlap = {100*n_ov/n_fc:.1f}% of FC-gain, {100*n_ov/n_enc:.1f}% of expansion')
    print(f'  enrichment over chance: {n_ov/exp_chance:.2f}x; hypergeom p = {hypergeom.sf(n_ov-1,n,n_enc,n_fc):.3g}')

    # ---- map codes onto the encroachment file's brain-model axis (compact) ----
    ax = enc_img.header.get_axis(1)
    offsets = {'CIFTI_STRUCTURE_CORTEX_LEFT': 0, 'CIFTI_STRUCTURE_CORTEX_RIGHT': NVERT}
    codes = np.zeros(enc_img.shape[1], dtype=np.float32)
    for name, slc, bm in ax.iter_structures():
        if name in offsets:
            codes[slc] = code_full[offsets[name] + bm.vertex]

    # ---- cluster-clean each class by surface area, recombine ----
    def write_binary(mask, path):
        nib.Cifti2Image(mask.astype(np.float32)[None, :],
                        header=(cifti2_axes.ScalarAxis(['m']), ax)).to_filename(str(path))

    clean = np.zeros_like(codes)
    for k in (1, 2, 3):
        with tempfile.TemporaryDirectory() as td:
            binp = Path(td) / 'b.dscalar.nii'; clp = Path(td) / 'c.dscalar.nii'
            write_binary(codes == k, binp)
            run(['wb_command', '-cifti-find-clusters', str(binp),
                 '0.5', str(MIN_AREA), '0.5', str(MIN_AREA), 'COLUMN', str(clp),
                 '-left-surface', str(SURF_L), '-right-surface', str(SURF_R)])
            kept = np.asarray(nib.load(str(clp)).get_fdata())[0] > 0
            clean[kept] = k
            print(f'class {k}: {int((codes==k).sum())} -> {int(kept.sum())} vtx after >= {MIN_AREA:.0f} mm^2')

    nib.Cifti2Image(clean[None, :], header=(cifti2_axes.ScalarAxis(['structfunc_conjunction_clean']), ax)
                    ).to_filename(str(CLEAN_DSCALAR))

    # ---- colour: subdued Wong tints for "only" classes, full SCAN magenta for overlap ----
    lut = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
    lut.write(
        "FC-gain only\n1 90 160 205 255\n"       # Wong-blue, ~45% toward full from pale
        "Expansion only\n2 242 185 85 255\n"     # Wong-orange, ~45% toward full from pale
        "Overlap\n3 142 0 103 255\n"             # SCAN magenta #8E0067 (full)
    )
    lut.close()
    run(['wb_command', '-cifti-label-import', str(CLEAN_DSCALAR), lut.name, str(CLEAN_DLABEL)])
    os.unlink(lut.name)
    print('wrote', CLEAN_DLABEL if CLEAN_DLABEL.exists() else 'NOTHING')


if __name__ == '__main__':
    main()
