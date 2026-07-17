"""
De-speckle the SCAN-seed FC high-minus-low difference map for Fig 3a.

wb_view has no cluster-size slider, so the diff surface is cleaned offline: each
SIGN is clustered separately on the fsLR-32k midthickness by TRUE surface area
(wb_command -cifti-find-clusters), tiny isolated specks are dropped, and the
ORIGINAL continuous Fisher-z values are kept inside surviving clusters (no value
threshold is imposed on the display — the cluster-clean is the only cleanup).

Mirrors the cluster-clean recipe in gen_conjunction_map.py (per-sign, true mm^2,
left/right midthickness). This script exists so that de-speckle step is finally
reproducible and version-controlled, and so it runs against the CORRECTED seed-FC
group-mean (seed_fc.py per-vertex-denominator fix, Jul 10) rather than the stale
Jun-26 offline output.

Source : outputs/cifti_for_workbench/scan_seed_fc_groupmean.dscalar.nii
         map index 2 = 'SCANseedFC_high_minus_low' (Fisher-z, high - low threat)
Outputs: outputs/cifti_for_workbench/scan_seed_fc_diff_clusterclean_min{100,200}.dscalar.nii
         (single map each -> loads as a scalar layer, no map dropdown)

Also prints the corrected map's robust distribution (percentiles + % cortex beyond
+/-0.02) so the ROY-BIG-BL User-Scale palette caps can be re-derived on real
numbers rather than the pre-fix values the old locked spec was tuned to.
"""
import subprocess, tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.cifti2 import cifti2_axes

ROOT = Path(__file__).resolve().parents[2]
FC       = ROOT / 'outputs/cifti_for_workbench/scan_seed_fc_groupmean.dscalar.nii'
OUTD     = ROOT / 'outputs/cifti_for_workbench'
SURF_L   = ROOT / 'data/neuromaps_cache/atlases/fsLR/tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii'
SURF_R   = ROOT / 'data/neuromaps_cache/atlases/fsLR/tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii'

DIFF_MAP    = 2        # 'SCANseedFC_high_minus_low'
CLUSTER_T   = 0.02     # value that defines a suprathreshold vertex, per sign
MIN_AREAS   = (100.0, 200.0)   # mm^2; keep clusters at/above each -> one file each


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print('CMD FAILED:', ' '.join(map(str, cmd)))
        print(r.stdout, r.stderr)
        raise SystemExit(1)
    return r


def report_distribution(diff):
    fin = diff[np.isfinite(diff)]   # seed vertices are already NaN in the source
    print(f'\n=== corrected diff map: distribution over {fin.size} finite cortical vertices ===')
    print(f'  min/max      : {fin.min():+.4f} / {fin.max():+.4f}')
    print(f'  mean / sd    : {fin.mean():+.4f} / {fin.std():.4f}')
    for p in (1, 2, 5, 10, 90, 95, 98, 99):
        print(f'  p{p:<2d}         : {np.percentile(fin, p):+.4f}')
    print(f'  % vertices > +0.02 : {100*np.mean(fin > 0.02):.1f}%')
    print(f'  % vertices < -0.02 : {100*np.mean(fin < -0.02):.1f}%')
    print('  (palette caps: set Pos-Max ~ p98, Neg-Max ~ p2, User Scale, ROY-BIG-BL, Display Zero OFF)\n')


def despeckle(diff, ax, min_area):
    """Return continuous diff kept only inside per-sign clusters >= min_area mm^2."""
    def cluster_mask(binary, tag):
        with tempfile.TemporaryDirectory() as td:
            binp = Path(td) / 'b.dscalar.nii'
            clp  = Path(td) / 'c.dscalar.nii'
            nib.Cifti2Image(binary.astype(np.float32)[None, :],
                            header=(cifti2_axes.ScalarAxis(['m']), ax)).to_filename(str(binp))
            run(['wb_command', '-cifti-find-clusters', str(binp),
                 '0.5', str(min_area), '0.5', str(min_area), 'COLUMN', str(clp),
                 '-left-surface', str(SURF_L), '-right-surface', str(SURF_R)])
            lab = np.asarray(nib.load(str(clp)).get_fdata())[0]
        kept = lab > 0
        n_before = int(binary.sum())
        n_clusters = int(np.unique(lab[kept]).size)
        print(f'  {tag}: {n_before} suprathreshold vtx -> {int(kept.sum())} vtx '
              f'in {n_clusters} clusters >= {min_area:.0f} mm^2 '
              f'({100*kept.sum()/max(n_before,1):.1f}% of suprathreshold vtx kept)')
        return kept

    pos = cluster_mask(diff >= CLUSTER_T,  f'pos (>= +{CLUSTER_T})')
    neg = cluster_mask(diff <= -CLUSTER_T, f'neg (<= -{CLUSTER_T})')
    out = np.full_like(diff, np.nan, dtype=np.float32)
    surviving = pos | neg
    out[surviving] = diff[surviving]
    return out


def main():
    img = nib.load(str(FC))
    ax = img.header.get_axis(1)
    diff = np.asarray(img.get_fdata())[DIFF_MAP].astype(np.float64)

    report_distribution(diff)

    for min_area in MIN_AREAS:
        print(f'--- de-speckle at min {min_area:.0f} mm^2 ---')
        clean = despeckle(diff, ax, min_area)
        out = OUTD / f'scan_seed_fc_diff_clusterclean_min{int(min_area)}.dscalar.nii'
        nib.Cifti2Image(clean[None, :],
                        header=(cifti2_axes.ScalarAxis(['SCANseedFC_high_minus_low_clean']), ax)
                        ).to_filename(str(out))
        print(f'  wrote {out.name}\n')
    print('Done.')


if __name__ == '__main__':
    main()
