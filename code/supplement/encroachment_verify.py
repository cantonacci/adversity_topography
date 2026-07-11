"""One-off verification of the encroachment Results-section descriptives.
Replicates the exact definitions in code/figures/fig2_panels.py so we can
reconcile the docx numbers (composition shares, zone shares, zone x network
interaction) against the current data. Prints only; writes nothing.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from adtopo.config import cfg

NETS14 = ["DMN", "VIS", "FP", "DAN", "VAN", "SAL", "CO", "SMD", "SML", "AUD",
          "Tpole", "MTL", "PMN", "PON"]


def composition(hi, zone=None):
    suf = "" if zone is None else f"_{zone}"
    tot = hi[f"total_encroach_count{suf}"].values.astype(float)
    keep = tot > 0
    out = {n: (hi[f"encroach_count_{n}{suf}"].values.astype(float)[keep]
               / tot[keep]) * 100 for n in NETS14}
    return out, int(keep.sum())


def interaction(hi, medial_net, lateral_net):
    a_m = hi[f"encroach_frac_{medial_net}_medial"].fillna(0).values
    a_l = hi[f"encroach_frac_{medial_net}_lateral"].fillna(0).values
    b_m = hi[f"encroach_frac_{lateral_net}_medial"].fillna(0).values
    b_l = hi[f"encroach_frac_{lateral_net}_lateral"].fillna(0).values
    D = (a_m - b_m) - (a_l - b_l)
    t, p = ttest_1samp(D, 0.0)
    return t, p, D.mean() / D.std(ddof=1), len(D)


def within_net_zone_dz(hi, net, direction):
    m = hi[f"encroach_frac_{net}_medial"].fillna(0).values
    l = hi[f"encroach_frac_{net}_lateral"].fillna(0).values
    D = (m - l) if direction == "medial" else (l - m)
    return D.mean() / D.std(ddof=1)


def main():
    enc = pd.read_csv(cfg.BASE_DIR / "outputs/encroachment/encroachment_baseline.csv")
    base = pd.read_csv(cfg.DAT_DIR / "df_base.csv")[["sub_ID", "threat_composite"]]
    hi = enc.merge(base, on="sub_ID")
    hi = hi[hi["threat_composite"] >= 1].copy()
    print(f"high-threat (>=+1SD) n = {len(hi)}")

    for zone in [None, "medial", "lateral"]:
        comp, n = composition(hi, zone)
        order = sorted(NETS14, key=lambda x: comp[x].mean(), reverse=True)
        label = zone or "overall"
        top = ", ".join(f"{x} {comp[x].mean():.1f}%" for x in order[:5])
        print(f"\n[{label}] n={n}  top: {top}")

    print("\n--- zone x network interaction (D = (X_med - Y_med) - (X_lat - Y_lat)) ---")
    for lat in ["SML", "SMD"]:
        t, p, dz, n = interaction(hi, "CO", lat)
        print(f"  CO vs {lat}: t({n-1})={t:.2f}  p={p:.2e}  dz={dz:.3f}")

    print("\n--- within-network zone preference (Cohen's dz) ---")
    print(f"  CO  (medial - lateral):  dz={within_net_zone_dz(hi, 'CO', 'medial'):.3f}")
    for net in ["SMD", "SML"]:
        print(f"  {net} (lateral - medial): dz={within_net_zone_dz(hi, net, 'lateral'):.3f}")
    # somatomotor combined (SMD+SML) lateral preference
    smd_m = hi["encroach_frac_SMD_medial"].fillna(0).values
    smd_l = hi["encroach_frac_SMD_lateral"].fillna(0).values
    sml_m = hi["encroach_frac_SML_medial"].fillna(0).values
    sml_l = hi["encroach_frac_SML_lateral"].fillna(0).values
    for label, sm in [("SMD+SML mean", ((smd_l + sml_l) / 2 - (smd_m + sml_m) / 2)),
                      ("SMD+SML sum", ((smd_l + sml_l) - (smd_m + sml_m)))]:
        print(f"  somatomotor {label} (lateral - medial): dz={sm.mean()/sm.std(ddof=1):.3f}")


if __name__ == '__main__':
    main()
