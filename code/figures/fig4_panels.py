"""
Figure 4 — Cognitive cost of an enlarged SCAN (Pillar 4). Each panel a separate
vector PDF (+ PNG) under outputs/figures/nn/fig4/.

  fig4a_mediation     threat → SCAN → crystallized path diagram (standardised a/b)
  fig4b_crosswave     forest of the crystallised indirect effect across waves
  fig4c_withinperson  within-person ΔSCAN → Δcrystallized (+ cryst/fluid inset)
  fig4d_prediction    out-of-sample CV-R² gain from baseline SCAN (Analysis H)

Run prep_fig4_withinperson.py first (writes the fig4c_* tables).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

import figsrc as F

fs = F.fs
fs.set_style()
OUT = str(F.FIG_OUT / "fig4")
F.Path(OUT).mkdir(parents=True, exist_ok=True)

ZERO = "#d6d6d6"   # subtle zero-reference line (matches Fig 3 tweaks)

# Cognitive-domain colours (Wong, colourblind-safe), consistent across Fig 4.
# Domain = colour; significance = opacity (n.s. elements faded). SCAN's own
# magenta accent is reserved for the SCAN node in 4a only.
CRYST_COL = "#0072B2"   # crystallized — Wong blue
FLUID_COL = "#E69F00"   # fluid — Wong orange
NS_ALPHA = 0.35         # opacity for non-significant elements


def _within_inset(ax, c, rect=(0.60, 0.71, 0.37, 0.26)):
    """Cryst-vs-fluid within-person coefficient ± 95% CI inset (top-right).
    Colour = domain; opacity = significance."""
    ax.add_patch(Rectangle((rect[0] - 0.05, rect[1] - 0.05),
                           rect[2] + 0.09, rect[3] + 0.10, transform=ax.transAxes,
                           facecolor="white", edgecolor="none", zorder=4.4))
    axin = ax.inset_axes(list(rect), zorder=5)
    axin.set_facecolor("white"); axin.patch.set_alpha(1.0)
    order = ["Crystallized", "Fluid"]
    for j, name in enumerate(order):
        rr = c[c["outcome"] == name].iloc[0]
        col = CRYST_COL if name == "Crystallized" else FLUID_COL
        al = 1.0 if rr["p"] < 0.05 else NS_ALPHA
        axin.plot([rr["ci_lo"], rr["ci_hi"]], [j, j], color=col, lw=1.0,
                  alpha=al, zorder=2)
        axin.scatter([rr["beta"]], [j], s=11, color=col, edgecolors="black",
                     linewidths=0.4, alpha=al, zorder=3)
    axin.axvline(0, color=F.GREY["muted"], lw=0.5, zorder=1)
    axin.set_yticks([0, 1]); axin.set_yticklabels(order)
    axin.set_ylim(-0.6, 1.6); axin.invert_yaxis()
    axin.tick_params(labelsize=4.0, length=2)
    axin.set_xlabel("β (ΔSCAN→Δcog)", fontsize=4.2, labelpad=1)
    for s in ("top", "right"):
        axin.spines[s].set_visible(False)
    return axin


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4a — mediation path diagram (standardised coefficients)
# ─────────────────────────────────────────────────────────────────────────────
def panel_mediation():
    m = pd.read_csv(F.TAB_DIR / "phase6_mediation_SCAN.csv")
    r = m[m["outcome"] == "nihtb_cryst_y6"].iloc[0]
    base = pd.read_csv(F.ROOT / "outputs" / "data_processed" / "df_base.csv")
    sub = base.dropna(subset=["threat_composite", "prop_SCAN", "nihtb_cryst_y6"])
    sd_t, sd_s, sd_c = (sub["threat_composite"].std(), sub["prop_SCAN"].std(),
                        sub["nihtb_cryst_y6"].std())
    a_std = r["beta_a"] * sd_t / sd_s          # threat → SCAN
    b_std = r["beta_b"] * sd_s / sd_c          # SCAN → cryst (| threat)
    cp_std = r["beta_c_prime"] * sd_t / sd_c   # direct threat → cryst

    fig, ax = fs.figure("single", 60)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(0.12, 1.0); ax.axis("off")

    def box(xy, text, fc, ec, tc="black"):
        x, y = xy
        ax.add_patch(FancyBboxPatch((x - 0.16, y - 0.085), 0.32, 0.17,
                     boxstyle="round,pad=0.006,rounding_size=0.03",
                     linewidth=0.8, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=5.8,
                color=tc, fontweight="bold", zorder=4)

    def arrow(p0, p1, color, label, lab_xy, lw=1.2, ls="-", alpha=1.0, lc=None):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7,
                     lw=lw, color=color, ls=ls, alpha=alpha,
                     shrinkA=1, shrinkB=1, zorder=2))
        ax.text(*lab_xy, label, ha="center", va="center", fontsize=5.0,
                color=lc or color, zorder=5)

    T, S, C = (0.14, 0.60), (0.5, 0.88), (0.86, 0.60)
    box(T, "Threat", "white", "black")
    box(S, "SCAN\nexpansion", F.SCAN_COLOR, "black", tc="white")
    box(C, "Crystallized\ncognition", CRYST_COL, "black", tc="white")
    arrow((0.24, 0.69), (0.37, 0.83), "black", f"a = {a_std:+.2f}***", (0.24, 0.83))
    arrow((0.63, 0.83), (0.76, 0.69), "black", f"b = {b_std:+.2f}*", (0.78, 0.83))
    arrow((0.30, 0.57), (0.70, 0.57), F.GREY["muted"],
          f"c′ = {cp_std:+.2f} (direct)", (0.5, 0.52), lw=0.9, ls=(0, (4, 2)))
    ax.text(0.5, 0.27,
            f"Indirect (a×b):  {r['indirect']:.2f} cognition points\n"
            f"95% CI [{r['boot_ci_lo']:.2f}, {r['boot_ci_hi']:.2f}],  "
            f"p = {r['boot_p']:.3f}",
            ha="center", va="center", fontsize=5.4,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#cccccc", lw=0.5))
    fs.panel_label(ax, "a", x=0.0, y=1.0)
    fs.save(fig, "fig4a_mediation", outdir=OUT)
    print(f"  fig4a_mediation  a*={a_std:+.3f} b*={b_std:+.3f} cprime*={cp_std:+.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4b — cross-wave forest of the crystallised indirect effect
# ─────────────────────────────────────────────────────────────────────────────
def panel_crosswave():
    t = pd.read_csv(F.TAB_DIR / "phase6_cognition_timepoints.csv")
    # (label, indirect, lo, hi, n, colour, significant?)  sig = CI excludes 0
    rows = []
    for _, r in t[t["measure"] == "Crystallized"].iterrows():
        rows.append((f"Crystallized {r['wave'].upper()}", r["indirect"], r["ci_lo"],
                     r["ci_hi"], int(r["n"]), CRYST_COL, bool(r["ci_excl_0"])))
    fr = t[(t["measure"] == "Fluid") & (t["wave"] == "y6")].iloc[0]
    rows.append(("Fluid Y6", fr["indirect"], fr["ci_lo"], fr["ci_hi"],
                 int(fr["n"]), FLUID_COL, bool(fr["ci_excl_0"])))
    rows = rows[::-1]                       # Crystallized Y2 at top

    fig, ax = fs.figure("single", 56)
    ax.axvline(0, color=F.GREY["muted"], lw=0.6, zorder=1)
    for i, (lab, ind, lo, hi, nn, col, sig) in enumerate(rows):
        al = 1.0 if sig else NS_ALPHA
        ax.plot([lo, hi], [i, i], color=col, lw=1.3, alpha=al,
                solid_capstyle="round", zorder=2)
        # marker kept solid (full opacity) so the CI line doesn't show through it
        ax.scatter([ind], [i], s=22, color=col, edgecolors="black",
                   linewidths=0.4, zorder=3)
        ax.text(hi + 0.02, i, f"n={nn}", va="center", ha="left",
                fontsize=4.4, color=F.GREY["muted"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-0.80, 0.22)
    ax.set_xlabel("Indirect effect on cognition\n(threat → SCAN → cognition)")
    fs.despine(ax)
    fs.panel_label(ax, "b")
    fs.save(fig, "fig4b_crosswave", outdir=OUT)
    print("  fig4b_crosswave  done")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4c — within-person ΔSCAN → Δcrystallized (+ cryst/fluid inset)
# ─────────────────────────────────────────────────────────────────────────────
def panel_withinperson():
    p = pd.read_csv(F.TAB_DIR / "fig4c_within_person_partial.csv")
    c = pd.read_csv(F.TAB_DIR / "fig4c_within_person_coefs.csv")
    cr = c[c["outcome"] == "Crystallized"].iloc[0]
    x, y = p["resid_dSCAN"].values, p["resid_dcryst"].values
    xlo, xhi = np.percentile(x, [1, 99])

    fig, ax = fs.figure("single", 60)
    ax.axhline(0, color=ZERO, lw=0.3, zorder=0)
    ax.axvline(0, color=ZERO, lw=0.3, zorder=0)
    ax.scatter(x, y, s=2.5, color=F.GREY["fill"], alpha=0.13,
               edgecolors="none", zorder=1)
    edges = np.linspace(xlo, xhi, 11); ctr = 0.5 * (edges[:-1] + edges[1:])
    ib = np.clip(np.digitize(x, edges[1:-1]), 0, len(ctr) - 1)
    by = np.array([y[ib == b].mean() if (ib == b).any() else np.nan
                   for b in range(len(ctr))])
    xs = np.linspace(xlo, xhi, 50)
    ax.plot(xs, cr["beta"] * xs, color=CRYST_COL, lw=1.2, zorder=3)
    ax.scatter(ctr, by, s=14, color=CRYST_COL, edgecolors="white",
               linewidths=0.3, zorder=4)
    ax.text(0.035, 0.045,
            f"β = {cr['beta']:.2f}  [{cr['ci_lo']:.2f}, {cr['ci_hi']:.2f}]\n"
            f"p < .001    n = {int(cr['n_subj'])} children",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=5.2)
    ax.set_xlabel("Δ SCAN  (within-person, residualized)")
    ax.set_ylabel("Δ Crystallized  (residualized)")
    ax.set_xlim(xlo, xhi)
    fs.despine(ax)
    fs.panel_label(ax, "c")
    _within_inset(ax, c)
    fs.save(fig, "fig4c_withinperson", outdir=OUT)
    print(f"  fig4c_withinperson  cryst β={cr['beta']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4c (ALT) — binned-means version (trend-forward; no dominating cloud)
# ─────────────────────────────────────────────────────────────────────────────
def panel_withinperson_binned(nbins=10):
    p = pd.read_csv(F.TAB_DIR / "fig4c_within_person_partial.csv")
    c = pd.read_csv(F.TAB_DIR / "fig4c_within_person_coefs.csv")
    cr = c[c["outcome"] == "Crystallized"].iloc[0]
    x, y = p["resid_dSCAN"].values, p["resid_dcryst"].values

    # equal-count (quantile) bins of ΔSCAN → mean Δcryst ± 95% CI per bin
    q = pd.qcut(x, nbins, labels=False, duplicates="drop")
    bx, by, be = [], [], []
    for b in range(q.max() + 1):
        m = q == b
        bx.append(x[m].mean()); by.append(y[m].mean())
        be.append(1.96 * y[m].std(ddof=1) / np.sqrt(m.sum()))
    bx, by, be = np.array(bx), np.array(by), np.array(be)

    sd_pts = float(cr["sd_draw"])          # SD of raw Δcrystallized (points)

    fig, ax = fs.figure("single", 60)
    ax.axhline(0, color=ZERO, lw=0.3, zorder=0)
    xs = np.linspace(bx.min(), bx.max(), 50)
    ax.plot(xs, cr["beta"] * xs, color=CRYST_COL, lw=1.2, zorder=2)
    ax.errorbar(bx, by, yerr=be, fmt="o", ms=4, color=CRYST_COL,
                ecolor="black", elinewidth=0.5, capsize=1.8, capthick=0.5,
                markeredgecolor="black", markeredgewidth=0.5, zorder=3)
    pad = 0.04
    ax.set_ylim(min(by - be) - pad, max(by + be) + pad)
    ax.text(0.035, 0.06,
            f"β = {cr['beta']:.2f}  [{cr['ci_lo']:.2f}, {cr['ci_hi']:.2f}]\n"
            f"p < .001    n = {int(cr['n_subj'])} children",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=5.2)
    ax.set_xlabel("Δ SCAN  (within-person, residualized;\nlow → high SCAN growth)")
    ax.set_ylabel("Δ Crystallized  (residualized, SD)\nmean ± 95% CI per decile")
    # interpretable right axis in crystallized-score points (1 SD = sd_pts points)
    sax = ax.secondary_yaxis("right", functions=(lambda v: v * sd_pts,
                                                 lambda v: v / sd_pts))
    sax.set_ylabel("Δ Crystallized  (points)")
    fs.despine(ax)
    fs.panel_label(ax, "c")
    _within_inset(ax, c, rect=(0.55, 0.70, 0.33, 0.25))
    fs.save(fig, "fig4c_withinperson_binned", outdir=OUT)
    print(f"  fig4c_withinperson_binned  {len(bx)} bins, "
          f"range {by.max():+.3f}→{by.min():+.3f} SD "
          f"({(by.max()-by.min())*sd_pts:.1f} pts span)")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4c (ALT 2) — quartile group contrast (interpretable points)
# ─────────────────────────────────────────────────────────────────────────────
def panel_withinperson_quartile():
    p = pd.read_csv(F.TAB_DIR / "fig4c_within_person_partial.csv")
    c = pd.read_csv(F.TAB_DIR / "fig4c_within_person_coefs.csv")
    cr = c[c["outcome"] == "Crystallized"].iloc[0]
    sd_pts = float(cr["sd_draw"])
    x, y = p["resid_dSCAN"].values, p["resid_dcryst"].values * sd_pts  # → points

    q = pd.qcut(x, 4, labels=False)
    mu = np.array([y[q == b].mean() for b in range(4)])
    ci = np.array([1.96 * y[q == b].std(ddof=1) / np.sqrt((q == b).sum())
                   for b in range(4)])
    labels = ["Q1\n(least)", "Q2", "Q3", "Q4\n(most)"]

    fig, ax = fs.figure("single", 60)
    ax.axhline(0, color=ZERO, lw=0.3, zorder=0)
    ax.errorbar(range(4), mu, yerr=ci, fmt="o", ms=5, color=CRYST_COL,
                ecolor="black", elinewidth=0.5, capsize=1.8, capthick=0.5,
                markeredgecolor="black", markeredgewidth=0.5, zorder=3)
    ax.plot(range(4), mu, color=CRYST_COL, lw=0.8, alpha=0.5, zorder=2)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_xlim(-0.4, 3.4)
    ax.set_xlabel("SCAN growth quartile  (within-person ΔSCAN)")
    ax.set_ylabel("Δ Crystallized vs. sample average\n(points, covariate-adjusted)")
    ax.text(0.035, 0.06,
            f"Q4 − Q1 = {mu[-1] - mu[0]:.1f} pts\n"
            f"β = {cr['beta']:.2f}, p < .001   n = {int(cr['n_subj'])} children",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=5.2)
    fs.despine(ax)
    fs.panel_label(ax, "c")
    _within_inset(ax, c, rect=(0.60, 0.80, 0.37, 0.24))
    fs.save(fig, "fig4c_withinperson_quartile", outdir=OUT)
    print(f"  fig4c_withinperson_quartile  Q1={mu[0]:+.2f} → Q4={mu[-1]:+.2f} pts")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4d — out-of-sample prediction (Analysis H)
# ─────────────────────────────────────────────────────────────────────────────
def panel_prediction():
    # values from outputs/tables/H_cv_prediction_summary.txt (Analysis H, finalised)
    cov_r2, scan_r2 = 0.0574, 0.0682          # M0_cov, Mscan_cov+SCAN
    incr, perm_p = 0.011, 0.001               # incremental CV-R2 + permutation p

    fig, ax = fs.figure("single", 58)
    bars = [("Covariates", cov_r2, F.GREY["fill"]),
            ("+ baseline\nSCAN", scan_r2, CRYST_COL)]
    for i, (lab, v, col) in enumerate(bars):
        ax.bar(i, v, width=0.62, color=col, edgecolor="black", linewidth=0.4, zorder=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels([b[0] for b in bars])
    ax.set_xlim(-0.6, 2.5)
    ax.set_ylim(0, scan_r2 * 1.22)
    ax.set_ylabel("Cross-validated R²\n(year-6 crystallized)")
    # dashed covariate-baseline reference + increment bracket to the right of the bar
    ax.plot([0, 1.55], [cov_r2, cov_r2], color=F.GREY["muted"], lw=0.6,
            ls=(0, (3, 2)), zorder=1)
    ax.annotate("", xy=(1.5, scan_r2), xytext=(1.5, cov_r2),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="black"))
    ax.text(1.62, (cov_r2 + scan_r2) / 2,
            f"ΔR² = +{incr:.3f}\npermutation p = {perm_p:.3f}",
            va="center", ha="left", fontsize=5.0)
    fs.despine(ax)
    fs.panel_label(ax, "d")
    fs.save(fig, "fig4d_prediction", outdir=OUT)
    print("  fig4d_prediction  done")


if __name__ == "__main__":
    print("Building Figure 4 panels →", OUT)
    panel_mediation()
    panel_crosswave()
    panel_withinperson()            # scatter version
    panel_withinperson_binned()     # binned-means alt
    panel_withinperson_quartile()   # quartile-contrast alt
    panel_prediction()
    print("Done.")
