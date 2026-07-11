"""
Group-mean SCAN-FC surface overlays for Workbench (the 'connectivity fingerprint'
on the brain): each non-SCAN network's territory is painted by SCAN's MEAN
functional coupling to it, separately for high- and low-threat youth, plus the
high−low difference. Render high vs low side by side in Workbench to see SCAN's
connectivity profile shift; the difference map mirrors the Fig 3a β surface.

Output: outputs/cifti_for_workbench/fc_scan_groupmean_fingerprint.dscalar.nii
  (3 maps: SCANfc_high, SCANfc_low, SCANfc_high_minus_low)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

ROOT = Path(__file__).resolve().parents[2]
from adtopo.config import cfg

FC_PATH = cfg.DAT_DIR / "fc_ses-00A.csv"
BASE    = ROOT / "outputs/data_processed/df_base.csv"
ATLAS   = cfg.ATLAS_DIR / "abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii"
OUT     = ROOT / "outputs/cifti_for_workbench"
N_CORT, N_FULL = 59412, 91282
HIGH, LOW = 1.0, -1.0
NET_LABEL = {"DMN": 1, "VIS": 2, "FP": 3, "DAN": 5, "VAN": 7, "SAL": 8, "CO": 9,
             "SMD": 10, "SML": 11, "AUD": 12, "Tpole": 13, "MTL": 14, "PMN": 15,
             "PON": 16, "SCAN": 18}
NETS = [n for n in NET_LABEL if n != "SCAN"]

def scan_col(x):
    for c in (f"fc_{x}_SCAN", f"fc_SCAN_{x}"):
        if c in fc.columns:
            return c
    raise KeyError(f"no SCAN-{x} column")


def paint(valdict):
    surf = np.full(N_FULL, np.nan, dtype=np.float32)
    cort = adata[:N_CORT]
    for net, val in valdict.items():
        surf[:N_CORT][cort == NET_LABEL[net]] = val
    return surf


def main():
    global fc, adata
    fc = pd.read_csv(FC_PATH)
    base = pd.read_csv(BASE)[["sub_ID", "threat_composite"]]
    m = fc.merge(base, on="sub_ID")
    hi = m[m["threat_composite"] >= HIGH]
    lo = m[m["threat_composite"] <= LOW]
    print(f"N FC={len(fc)}  high={len(hi)}  low={len(lo)}")

    hi_mean = {x: hi[scan_col(x)].mean() for x in NETS}
    lo_mean = {x: lo[scan_col(x)].mean() for x in NETS}
    diff = {x: hi_mean[x] - lo_mean[x] for x in NETS}
    print("  SMD  high={:.3f} low={:.3f} diff={:+.3f}".format(hi_mean["SMD"], lo_mean["SMD"], diff["SMD"]))
    print("  SAL  high={:.3f} low={:.3f} diff={:+.3f}".format(hi_mean["SAL"], lo_mean["SAL"], diff["SAL"]))

    atlas_img = nib.load(str(ATLAS))
    bm = atlas_img.header.get_axis(1)
    adata = atlas_img.get_fdata()[0].astype(np.int16)

    data = np.stack([paint(hi_mean), paint(lo_mean), paint(diff)], axis=0)
    sax = nib.cifti2.ScalarAxis(["SCANfc_high", "SCANfc_low", "SCANfc_high_minus_low"])
    hdr = nib.Cifti2Header.from_axes((sax, bm))
    out = OUT / "fc_scan_groupmean_fingerprint.dscalar.nii"
    nib.save(nib.Cifti2Image(data.astype(np.float32), hdr), str(out))
    print(f"wrote {out.name}  (3 maps: high, low, high-minus-low)")


if __name__ == '__main__':
    main()
