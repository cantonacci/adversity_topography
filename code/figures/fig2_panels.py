"""
Figure 2 — Adversity drives SCAN to encroach into specific neighbours (Pillar 2).
Each panel a separate vector PDF (+ PNG preview) under outputs/figures/nn/.

Code panels:
  fig2a_encroach_spectrum  which networks SCAN displaces in high-threat youth (bar)
  fig2b_zone_dissociation  medial→CO vs lateral→somatomotor double dissociation
  fig2c_gradient           expansion hotspot localises to the sensorimotor pole

Workbench panels (made separately by user; staged in cifti_for_workbench/):
  expansion/difference surface, displaced-network map, SCAN-border overlay.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ttest_1samp

import figsrc as F

fs = F.fs
fs.set_style()
OUT = str(F.FIG_OUT / "fig2")
F.Path(OUT).mkdir(parents=True, exist_ok=True)
_rng = np.random.default_rng(0)

NETS14 = ["DMN", "VIS", "FP", "DAN", "VAN", "SAL", "CO", "SMD", "SML", "AUD",
          "Tpole", "MTL", "PMN", "PON"]


def _load_hi_encroach():
    """Subject-level encroachment rows for high-threat (≥+1 SD) youth."""
    enc = pd.read_csv(F.ROOT / "outputs/encroachment/encroachment_baseline.csv")
    base = pd.read_csv(F.DAT_DIR / "df_base.csv")[["sub_ID", "threat_composite"]]
    hi = enc.merge(base, on="sub_ID")
    return hi[hi["threat_composite"] >= 1].copy()


def _composition(hi, zone=None):
    """Lynch-style encroachment profile: per subject, each network's share (%) of
    the TOTAL encroaching SCAN area (sums to ~100% across networks). zone in
    {None, 'medial', 'lateral'}. Returns dict net→per-subject % (subjects with
    no encroachment in that zone dropped) and the n used."""
    suf = "" if zone is None else f"_{zone}"
    tot = hi[f"total_encroach_count{suf}"].values.astype(float)
    keep = tot > 0
    out = {n: (hi[f"encroach_count_{n}{suf}"].values.astype(float)[keep]
               / tot[keep]) * 100 for n in NETS14}
    return out, int(keep.sum())


def _lynch_points(ax, comp, order, ymax=100):
    """Lynch Fig-2e style: per-subject points, a thin black 95% CI bar, and a
    network-coloured dot at the mean."""
    for i, net in enumerate(order):
        v = comp[net]
        ax.scatter(i + _rng.normal(0, 0.09, len(v)), v, s=1.6,
                   color=F.GREY["fill"], alpha=0.18, linewidths=0,
                   rasterized=True, zorder=1)
        m = v.mean()
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar(i, m, yerr=ci, fmt="none", ecolor="black", elinewidth=0.6,
                    capsize=1.8, zorder=3)
        ax.scatter(i, m, s=26, color=F.net_color(net), edgecolors="black",
                   linewidths=0.5, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, ymax)


def _star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def _sig_bracket(ax, x1, x2, y, stars, h=2.5):
    """Lynch-style significance bracket: horizontal line at y with short down-ticks
    over the two compared columns, star(s) above."""
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color="black", lw=0.6, zorder=5)
    ax.text((x1 + x2) / 2, y, stars, ha="center", va="bottom", fontsize=6, zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2a — encroachment spectrum (which networks SCAN displaces)
# ─────────────────────────────────────────────────────────────────────────────
def panel_encroach_spectrum():
    # Lynch-style encroachment profile: each network's share of the total
    # encroaching SCAN area in high-threat youth. Grey = subjects; black = 95% CI;
    # coloured dot = mean (atlas colour of the encroached-upon network).
    hi = _load_hi_encroach()
    comp, n = _composition(hi)
    order = sorted(NETS14, key=lambda x: comp[x].mean(), reverse=True)[:3]  # SMD,SML,CO

    fig, ax = fs.figure("single", 62)
    _lynch_points(ax, comp, order, ymax=60)   # y capped; sparse upper tail clipped
    ax.set_ylabel("Encroachment\n(% of expanded SCAN area)")
    fs.despine(ax)
    fs.panel_label(ax, "a")
    fs.save(fig, "fig2a_encroach_spectrum", outdir=OUT)
    print(f"  fig2a_encroach_spectrum  (n={n}) top:",
          ", ".join(f"{x} {comp[x].mean():.1f}%" for x in order[:4]))


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2b — medial vs lateral double dissociation
# ─────────────────────────────────────────────────────────────────────────────
def _zone_interaction_stat(hi):
    """Double-dissociation (zone × network) contrast over high-threat youth: the
    #1 medial network (CO) vs the #1 lateral network (SML), crossed with zone, on
    the per-network displaced fraction (matches the manuscript's validated stat).
       D = (CO_med − SML_med) − (CO_lat − SML_lat), paired t vs 0.
    SML has no medial cortex → frac is 0/0, imputed to 0 (structurally absent)."""
    co_m = hi["encroach_frac_CO_medial"].fillna(0).values
    co_l = hi["encroach_frac_CO_lateral"].fillna(0).values
    sml_m = hi["encroach_frac_SML_medial"].fillna(0).values
    sml_l = hi["encroach_frac_SML_lateral"].fillna(0).values
    D = (co_m - sml_m) - (co_l - sml_l)
    t, p = ttest_1samp(D, 0.0)
    return t, p, D.mean() / D.std(ddof=1), len(D)


def panel_zone_dissociation():
    # Lynch-style, zone-resolved: each network's share of the encroaching SCAN
    # area WITHIN the medial vs lateral zone. Top-3 networks per zone, with
    # FDR-corrected pairwise significance brackets (Lynch Fig-2e style).
    from scipy.stats import ttest_rel
    from statsmodels.stats.multitest import multipletests
    hi = _load_hi_encroach()
    comp_med, n_med = _composition(hi, "medial")
    comp_lat, n_lat = _composition(hi, "lateral")
    order_med = sorted(NETS14, key=lambda x: comp_med[x].mean(), reverse=True)[:3]
    order_lat = sorted(NETS14, key=lambda x: comp_lat[x].mean(), reverse=True)[:3]

    fig, axes = fs.grid(118, 62, nrows=1, ncols=2, sharey=True)
    _lynch_points(axes[0], comp_med, order_med, ymax=125)
    _lynch_points(axes[1], comp_lat, order_lat, ymax=125)
    axes[0].set_title("Medial zone\n(cingulate)", fontsize=6)
    axes[1].set_title("Lateral zone\n(insula / operculum)", fontsize=6)
    axes[0].set_ylabel("Encroachment\n(% within zone)")
    for ax in axes:
        fs.despine(ax)

    # pairwise significance brackets (paired t, BH-FDR within zone)
    # medial: CO vs DAN (short, lower), CO vs VIS (long, higher)
    # lateral: SML vs AUD (short, lower), SMD vs AUD (long, higher)
    mx = {n: k for k, n in enumerate(order_med)}
    lx = {n: k for k, n in enumerate(order_lat)}
    p_med = [ttest_rel(comp_med["CO"], comp_med["DAN"])[1],
             ttest_rel(comp_med["CO"], comp_med["VIS"])[1]]
    p_lat = [ttest_rel(comp_lat["SML"], comp_lat["AUD"])[1],
             ttest_rel(comp_lat["SMD"], comp_lat["AUD"])[1]]
    q_med = multipletests(p_med, method="fdr_bh")[1]
    q_lat = multipletests(p_lat, method="fdr_bh")[1]
    _sig_bracket(axes[0], mx["CO"], mx["DAN"], 104, _star(q_med[0]))
    _sig_bracket(axes[0], mx["CO"], mx["VIS"], 116, _star(q_med[1]))
    _sig_bracket(axes[1], lx["SML"], lx["AUD"], 104, _star(q_lat[0]))
    _sig_bracket(axes[1], lx["SMD"], lx["AUD"], 116, _star(q_lat[1]))

    t, p, dz, n = _zone_interaction_stat(hi)
    ptxt = "p < 10⁻¹⁰⁰" if p < 1e-100 else f"p = {p:.0e}"
    axes[1].text(0.97, 0.40,
                 f"zone × network\ninteraction\nt({n-1}) = {t:.1f},  dz = {dz:.2f}\n{ptxt}",
                 transform=axes[1].transAxes, ha="right", va="center", fontsize=5)
    fs.panel_label(axes[0], "b", x=-0.30, y=1.06)
    fs.save(fig, "fig2b_zone_dissociation", outdir=OUT)
    print(f"  fig2b_zone_dissociation  medial→{order_med[0]} {comp_med[order_med[0]].mean():.0f}%, "
          f"lateral→{order_lat[0]} {comp_lat[order_lat[0]].mean():.0f}%; "
          f"interaction t({n-1})={t:.1f} dz={dz:.2f} p={p:.1e}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2c — gradient localisation (expansion hotspot at the sensorimotor pole)
# ─────────────────────────────────────────────────────────────────────────────
def panel_gradient():
    g = pd.read_csv(F.TAB_DIR / "D_gradient_hotspot_vertices.csv")
    allz = g["g1_z"].values
    hotz = g.loc[g["is_hotspot"], "g1_z"].values
    hot_pct = g.loc[g["is_hotspot"], "gradient_pct"].mean()

    xs = np.linspace(allz.min(), allz.max(), 240)
    ka = gaussian_kde(allz)(xs)
    kh = gaussian_kde(hotz)(xs)

    fig, ax = fs.figure("single", 60)
    ax.fill_between(xs, ka, color=F.GREY["muted"], alpha=0.45, lw=0)
    ax.fill_between(xs, kh, color=F.SCAN_COLOR, alpha=0.40, lw=0)
    ax.plot(xs, ka, color=F.GREY["line"], lw=0.9)
    ax.plot(xs, kh, color=F.SCAN_COLOR, lw=1.2)
    ax.axvline(allz.mean(), color=F.GREY["muted"], lw=0.4, ls=(0, (2, 3)))
    ax.axvline(hotz.mean(), color=F.SCAN_COLOR, lw=0.5, ls=(0, (2, 3)), alpha=0.7)

    # legend-ish labels on the curves
    ax.text(allz.mean() + 0.1, ka.max() * 0.92, "all cortex", fontsize=5,
            color=F.GREY["line"], ha="left", va="top")
    ax.text(hotz.mean() - 0.1, kh.max() * 1.02, "SCAN-expansion\nhotspot",
            fontsize=5.2, color=F.SCAN_COLOR, ha="right", va="bottom")
    # pole hints
    ax.text(0.015, -0.16, "← sensorimotor", transform=ax.transAxes, fontsize=5,
            ha="left", va="top", color=F.GREY["muted"])
    ax.text(0.985, -0.16, "association →", transform=ax.transAxes, fontsize=5,
            ha="right", va="top", color=F.GREY["muted"])
    ax.text(0.60, 0.94, f"hotspot at {hot_pct:.0f}th gradient pct\n"
            f"mean G1 = {hotz.mean():.2f} SD\nspin p = 0.001",
            transform=ax.transAxes, fontsize=5.2, ha="left", va="top")

    ax.set_xlabel("Principal cortical gradient  G1 (z)")
    ax.set_ylabel("Vertex density")
    ax.set_ylim(0, None)
    fs.despine(ax)
    fs.panel_label(ax, "c")
    fs.save(fig, "fig2c_gradient", outdir=OUT)
    print(f"  fig2c_gradient  hotspot G1z={hotz.mean():+.2f} pct={hot_pct:.1f} "
          f"(cortex mean {allz.mean():+.2f})")


if __name__ == "__main__":
    print("Building Figure 2 panels →", OUT)
    panel_encroach_spectrum()
    panel_zone_dissociation()
    panel_gradient()
    print("Done.")
