"""
Figure 1 — Adversity selectively expands the SCAN.
Generates each panel as a separate vector PDF (+ PNG preview) under
outputs/figures/nn/, to be assembled in Illustrator. House style via figstyle.

Panels built here (code):
  fig1a_matrix      10 ELA factors x 15 networks partial-r matrix (SCAN boxed)
  fig1b_selectivity pseudo-ΔR² across 15 networks (SCAN dwarfs the rest)
  fig1c_scatters    threat / deprivation / unpredictability vs SCAN (bivariate)
  fig1d_splithalf   discovery vs replication threat→SCAN β across 20 splits
  fig1e_sibling     between- vs within-family threat→SCAN estimates (forest)
  fig1f_dose        SCAN share in high (≥+1 SD) vs low (≤−1 SD) threat youth

Workbench panels (made separately by user): SCAN axis-position inset;
+1 SD vs −1 SD threat SCAN surface juxtaposition.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
import statsmodels.api as sm

import figsrc as F
from adtopo.config import cfg

fs = F.fs
fs.set_style()

NET_ORDER = F.NET_ORDER
OUT = str(F.FIG_OUT / "fig1")
F.Path(OUT).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1a — ELA factor × network partial-r matrix
# ─────────────────────────────────────────────────────────────────────────────
def panel_matrix():
    r = pd.read_csv(F.TAB_DIR / "phase2_individual_r_matrix_baseline.csv", index_col=0)
    q = pd.read_csv(F.TAB_DIR / "phase2_individual_q_matrix_baseline.csv", index_col=0)
    p = pd.read_csv(F.TAB_DIR / "phase2_individual_p_matrix_baseline.csv", index_col=0)

    # Display family anger in the adversity-aligned direction. The raw ABCD factor
    # is reverse-valenced (higher raw = less anger; it correlates -0.52/-0.71 with
    # family conflict/physical trauma), so we negate it here to match the other
    # threat factors and the sign with which it enters the threat composite.
    # Sign-only: FDR q and p (significance markers) are unchanged.
    if "ELA_family_anger" in r.index:
        r.loc["ELA_family_anger"] = -r.loc["ELA_family_anger"]

    # columns ordered along the S-A axis (principal gradient): sensorimotor→assoc
    sa = pd.read_csv(F.TAB_DIR / "network_gradient_G1.csv")
    col_order = sa.sort_values("meanG1")["network"].tolist()

    # rows grouped by composite domain (threat, deprivation, unpredictability)
    row_order = cfg.ELA_THREAT_COLS + cfg.ELA_DEPRIVATION_COLS + cfg.ELA_UNPRED_COLS
    r = r.loc[row_order, col_order]
    q = q.loc[row_order, col_order]
    p = p.loc[row_order, col_order]

    vmax = np.ceil(np.nanmax(np.abs(r.values)) * 20) / 20  # round up to .05

    fig, ax = fs.figure(150, 86)
    im = ax.imshow(r.values, cmap=fs.DIVERGING, vmin=-vmax, vmax=vmax,
                   aspect="auto")

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([cfg.ELA_LABELS_SHORT[c] for c in row_order])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # significance markers — FDR q thresholds (consistent across all stars):
    # * q<.05, ** q<.01, *** q<.001
    def _fdr_star(qv):
        return "***" if qv < 0.001 else "**" if qv < 0.01 else "*" if qv < 0.05 else ""
    for i in range(len(row_order)):
        for j in range(len(col_order)):
            mk = _fdr_star(q.values[i, j])
            if mk:
                col = "white" if abs(r.values[i, j]) > 0.55 * vmax else "black"
                ax.text(j, i, mk, ha="center", va="center", color=col,
                        fontsize=5)

    # horizontal separators between composite domains
    for boundary in (len(cfg.ELA_THREAT_COLS) - 0.5,
                     len(cfg.ELA_THREAT_COLS) + len(cfg.ELA_DEPRIVATION_COLS) - 0.5):
        ax.axhline(boundary, color="white", lw=1.2)

    # box the SCAN column
    j = col_order.index("SCAN")
    ax.add_patch(mpatches.Rectangle((j - 0.5, -0.5), 1, len(row_order),
                 fill=False, edgecolor=F.SCAN_COLOR, lw=1.4, clip_on=False))

    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Partial r", rotation=90)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)

    fs.panel_label(ax, "a", x=-0.30, y=1.02)
    fs.save(fig, "fig1a_matrix", outdir=OUT)
    print("  fig1a_matrix  vmax=", vmax)


def panel_saaxis():
    # S-A axis legend strip to sit BENEATH the 1a matrix: a smooth left→right
    # gradient (sensorimotor pole → association pole) with a directional arrow
    # and pole labels. The 1a columns are ordered along this axis (verified:
    # ascending mean principal-gradient G1, SMD/SCAN/SML at the sensorimotor
    # pole … DMN at the association pole). Even spacing here matches the matrix's
    # evenly-spaced columns — it encodes ORDER, not the uneven true G1 spacing.
    # Align under the column block in Illustrator (matrix has left y-labels).
    ramp = np.linspace(0, 1, 256).reshape(1, -1)
    fig, ax = fs.figure(150, 18)
    ax.imshow(ramp, cmap="cividis", aspect="auto", extent=[0, 1, 0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-2.4, 1)
    ax.axis("off")
    # directional arrow + pole labels beneath the gradient bar
    ax.annotate("", xy=(1.0, -0.55), xytext=(0.0, -0.55), annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black"))
    ax.text(0.0, -1.35, "Sensorimotor pole", ha="left", va="top", fontsize=6)
    ax.text(1.0, -1.35, "Association pole", ha="right", va="top", fontsize=6)
    ax.text(0.5, -1.35, "S–A axis position", ha="center", va="top", fontsize=6)
    fs.save(fig, "fig1a_SAaxis", outdir=OUT)
    print("  fig1a_SAaxis  smooth sensorimotor→association gradient strip")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1b — network selectivity (pseudo-ΔR²)
# ─────────────────────────────────────────────────────────────────────────────
def panel_selectivity():
    d = pd.read_csv(F.TAB_DIR / "phase3_composites_delta_r2_baseline.csv")
    d = d.set_index("network").loc[NET_ORDER].reset_index()
    d = d.sort_values("delta_R2", ascending=True)  # barh: largest on top
    colors = [F.net_color(n) for n in d["network"]]

    fig, ax = fs.figure("single", 82)
    ax.barh(range(len(d)), d["delta_R2"].values, color=colors,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["network"].values)
    ax.set_xlabel("Incremental variance explained\nby adversity (pseudo-ΔR²)")
    scan_v = float(d.loc[d["network"] == "SCAN", "delta_R2"].iloc[0])
    fs.despine(ax)
    fs.panel_label(ax, "b")
    fs.save(fig, "fig1b_selectivity", outdir=OUT)
    print("  fig1b_selectivity  SCAN ΔR²=", round(scan_v, 4))


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1c — composite × SCAN bivariate scatters (small multiple, shared y)
# ─────────────────────────────────────────────────────────────────────────────
def panel_scatters():
    df = pd.read_csv(F.DAT_DIR / "df_base.csv")
    # per-dimension accent: threat / deprivation / unpredictability (matches the
    # ELA-dimension colour bars used across the figures)
    ELA_COLORS = ["#D55E00", "#0072B2", "#009E73"]
    fig, axes = fs.grid(120, 44, nrows=1, ncols=3, sharey=True)
    for k, (comp, ax) in enumerate(zip(cfg.COMPOSITE_COLS, axes)):
        col = ELA_COLORS[k]
        sub = df[[comp, "prop_SCAN"]].dropna()
        x = sub[comp].values
        y = sub["prop_SCAN"].values * 100.0
        ax.scatter(x, y, s=2, color=F.GREY["fill"], alpha=0.08, linewidths=0,
                   rasterized=True)
        # highlight bottom vs top decile of exposure (Kilian's suggestion) so the
        # association reads from the point cloud, not only the binned-mean trend
        p10, p90 = np.percentile(x, [10, 90])
        lo_m, hi_m = x <= p10, x >= p90
        ax.scatter(x[lo_m], y[lo_m], s=3, color="#4C72B0", alpha=0.30,
                   linewidths=0, rasterized=True, zorder=2)   # bottom 10% (low adversity)
        ax.scatter(x[hi_m], y[hi_m], s=3, color=col, alpha=0.30,
                   linewidths=0, rasterized=True, zorder=2)   # top 10% (high adversity)
        # regression line + 95% CI of the fit (grey, understated)
        X = sm.add_constant(x)
        m = sm.OLS(y, X).fit()
        xx = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 100)
        pr = m.get_prediction(sm.add_constant(xx)).summary_frame(alpha=0.05)
        ax.fill_between(xx, pr["mean_ci_lower"], pr["mean_ci_upper"],
                        color=F.GREY["muted"], alpha=0.25, linewidth=0)
        ax.plot(xx, pr["mean"], color=F.GREY["line"], lw=0.9, zorder=4)
        # binned means ± SEM (decile bins) in the SCAN accent — the trend signal
        bins = np.quantile(x, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(x, bins[1:-1]), 0, 9)
        bx = np.array([x[idx == b].mean() for b in range(10)])
        by = np.array([y[idx == b].mean() for b in range(10)])
        be = np.array([y[idx == b].std(ddof=1) / np.sqrt((idx == b).sum())
                       for b in range(10)])
        # thin line through the binned means makes the graded trend read clearly
        ax.plot(bx, by, color=col, lw=0.8, alpha=0.55, zorder=4)
        ax.errorbar(bx, by, yerr=be, fmt="o", ms=3, color=col,
                    ecolor=col, elinewidth=0.7, capsize=1.5,
                    markeredgecolor="white", markeredgewidth=0.3, zorder=5)
        r, pval = pearsonr(x, y)
        ptxt = "p < 0.001" if pval < 1e-3 else f"p = {pval:.3f}"
        ax.text(0.04, 0.97, f"r = {r:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=6.5, fontweight="bold")
        ax.text(0.04, 0.86, ptxt, transform=ax.transAxes,
                va="top", ha="left", fontsize=5.5)
        ax.set_xlabel(f"{cfg.COMPOSITE_LABELS[comp]} (z)")
        ax.set_xlim(-2.5, 4)
        ax.set_ylim(1.5, 4.0)
        if k == 0:
            ax.set_ylabel("SCAN cortical share (%)")
        fs.despine(ax)
    # legend spelling out what each mark is — esp. the magenta binned-mean trend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", ls="none", ms=2.2, color=F.GREY["fill"],
               alpha=0.6, label="individual youth"),
        Line2D([], [], color=F.GREY["line"], lw=0.9, label="linear fit (95% CI)"),
        Line2D([], [], marker="o", ms=3, lw=0.8, color=ELA_COLORS[-1],
               label="mean per adversity decile (± SEM)"),
    ]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=4.2,
                    frameon=False, handlelength=1.1, handletextpad=0.4,
                    labelspacing=0.3, borderpad=0.2)
    fs.panel_label(axes[0], "c", x=-0.38, y=1.06)
    fs.save(fig, "fig1c_scatters", outdir=OUT)
    print("  fig1c_scatters  done")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1d — split-half replication (discovery vs replication β)
# ─────────────────────────────────────────────────────────────────────────────
def panel_splithalf():
    # Across-network ΔR² profile for a representative split: discovery vs
    # replication, one point per network. SCAN sits top-right; all 15 points
    # hug the identity line (the r≈0.94 reproducibility the text cites). This
    # avoids the complementary-half see-saw artefact of a per-split β scatter.
    prof = pd.read_csv(F.TAB_DIR / "A_splithalf_profile_repsplit.csv")
    prof = prof.set_index("network").loc[NET_ORDER].reset_index()
    x = prof["disc_dr2"].values * 100   # express as % variance for readability
    y = prof["rep_dr2"].values * 100
    r_split = float(prof["dr2_profile_r"].iloc[0])
    hi = max(x.max(), y.max()) * 1.12

    fig, ax = fs.figure("single", 70)
    ax.plot([0, hi], [0, hi], ls="--", lw=0.5, color=F.GREY["muted"], zorder=0)
    for net, xv, yv in zip(prof["network"], x, y):
        ax.scatter(xv, yv, s=14 if net == "SCAN" else 9,
                   color=F.net_color(net), alpha=0.9,
                   edgecolors="black", linewidths=0.3,
                   zorder=3 if net == "SCAN" else 2)
    j = list(prof["network"]).index("SCAN")
    ax.annotate("SCAN", (x[j], y[j]), xytext=(-3, 3),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=6, color=F.SCAN_COLOR)
    ax.set_xlabel("Discovery half  ΔR² (%)")
    ax.set_ylabel("Replication half  ΔR² (%)")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.text(0.04, 0.96,
            f"profile r = {r_split:.2f}\nSCAN ΔR² rank #1\n& sig. in 20/20 halves",
            transform=ax.transAxes, va="top", ha="left", fontsize=6)
    fs.despine(ax)
    fs.panel_label(ax, "d")
    fs.save(fig, "fig1d_splithalf", outdir=OUT)
    print(f"  fig1d_splithalf  rep-split profile r={r_split:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1e — between- vs within-family threat→SCAN (forest)
# ─────────────────────────────────────────────────────────────────────────────
def panel_sibling():
    # Take-home: the threat→SCAN association is the SAME size estimated BETWEEN
    # families as it is WITHIN families (siblings discordant in threat — the
    # more-exposed sibling has the larger SCAN, with shared genes/SES/
    # neighbourhood held constant). A between-family confound would null the
    # within-family estimate; instead it is if anything larger. Both estimates
    # bracket the full-sample β → robust, not a family-level artefact.
    # Numbers from B_sibling_discordance_summary.txt:
    #   full-sample multivariate β = +0.00232 (phase3 reference)
    #   between-family            β = +0.001861 se 0.000241  p=1.1e-14 (Mundlak)
    #   within-family sibling FE  β = +0.003234 se 0.001567  z=2.06 p=0.039  ← primary
    #   (FE = family-demeaned OLS, cluster-robust SE; the canonical within est.
    #    NOT the Mundlak SE, which is anticonservative.)
    full_b = 0.002320
    rows = [   # label, beta, se, p-text, colour  (top → bottom)
        ("Between\nfamilies",          0.001861, 0.000241, "p < 10⁻¹⁴", F.GREY["muted"]),
        ("Within families\n(siblings)", 0.003234, 0.001567, "p = 0.039",  F.SCAN_COLOR),
    ]
    fig, ax = fs.figure(86, 52)
    ys = [1, 0]
    # solid black null line — the only full-height reference
    ax.axvline(0, color="black", lw=0.8, zorder=0)
    ax.text(0, 1.78, "null", ha="center", va="bottom", fontsize=4.6,
            color="black")
    # full-sample β: a faint, thin, SHORT dotted tick spanning only the data
    # rows, so it reads as a local reference and never as the zero line
    ax.plot([full_b, full_b], [-0.15, 1.15], color="#b9b9b9", lw=0.5,
            ls=(0, (1, 2.5)), zorder=0)
    ax.text(full_b, 1.30, "full-sample β", ha="center", va="bottom",
            fontsize=4.4, color="#9a9a9a")
    for y, (lab, b, se, ptxt, col) in zip(ys, rows):
        bold = col == F.SCAN_COLOR
        ax.errorbar(b, y, xerr=1.96 * se, fmt="o", color=col, ecolor=col,
                    ms=5.5 if bold else 5, elinewidth=1.1, capsize=2.5, zorder=3)
        ax.text(b + 1.96 * se + 0.00022, y, ptxt, ha="left", va="center",
                fontsize=5.5, color=col if bold else "black",
                fontweight="bold" if bold else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    for tl, (_, _, _, _, col) in zip(ax.get_yticklabels(), rows):
        if col == F.SCAN_COLOR:
            tl.set_color(col)
            tl.set_fontweight("bold")
    # plain-language take-home in the empty upper-right
    ax.text(0.97, 0.96, "effect persists\nwithin families",
            transform=ax.transAxes, ha="right", va="top", fontsize=5,
            color=F.SCAN_COLOR)
    ax.set_ylim(-0.7, 2.25)
    ax.set_xlim(-0.0011, 0.0072)
    ax.set_xlabel("Threat → SCAN  β  (95% CI)")
    fs.despine(ax)
    fs.panel_label(ax, "e", x=-0.46, y=1.08)
    fs.save(fig, "fig1e_sibling", outdir=OUT)
    print("  fig1e_sibling  between vs within-family forest (within FE p=0.039)")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1f — dose-response: SCAN share by threat group
# ─────────────────────────────────────────────────────────────────────────────
def panel_dose():
    df = pd.read_csv(F.DAT_DIR / "df_base.csv")
    t = df["threat_composite"]
    s = df["prop_SCAN"] * 100.0
    low = s[t <= -1].dropna().values
    mid = s[(t > -0.5) & (t < 0.5)].dropna().values   # average-threat youth
    high = s[t >= 1].dropna().values
    # Low and High are the emphasised contrast; Average sits subtly in between
    # (grey) to show the two groups straddle the sample's middle (dose-response).
    # Single-hue purple severity ramp (light→dark = low→high dose); purple avoids
    # clashing with the threat/deprivation/unpredictability dimension bars
    # (vermillion/blue/teal), SCAN magenta, and the fluid-cognition orange.
    DOSE_LOW, DOSE_HIGH = "#807DBA", "#3F007D"
    groups = [("Low\n(≤ −1 SD)",  low,  DOSE_LOW,  7, 3),
              ("Average\n(≈ 0 SD)", mid, F.GREY["line"], 4.5, 2),
              ("High\n(≥ +1 SD)", high, DOSE_HIGH, 7, 3)]
    means = [g[1].mean() for g in groups]
    sems = [g[1].std(ddof=1) / np.sqrt(len(g[1])) for g in groups]
    ci = [1.96 * s for s in sems]
    pct_inc = (means[2] - means[0]) / means[0] * 100   # high vs low

    # mean ± 95% CI points (not zero-baseline bars) so the y-axis can be zoomed
    # to the data band honestly — points don't imply area-from-zero. This makes
    # the +24% high-vs-low gap read as pronounced as it actually is.
    fig, ax = fs.figure(52, 62)
    xs = [0, 1, 2]
    # faint individual youth behind the means (y-axis is zoomed, so distribution
    # tails clip — these convey spread/overlap, not the full range)
    rng = np.random.default_rng(0)
    for xp, (lab, vals, col, ms, zo) in zip(xs, groups):
        jx = xp + rng.normal(0, 0.06, len(vals))
        ax.scatter(jx, vals, s=1.0, color=col, alpha=0.05, linewidths=0,
                   zorder=1, clip_on=True, rasterized=True)
    ax.plot(xs, means, color=F.GREY["muted"], lw=0.9, zorder=2)
    ax.plot([0, 2], [means[0], means[0]], ls=(0, (3, 2)), lw=0.6,
            color=F.GREY["muted"], zorder=0)   # low-mean reference
    for xp, (lab, vals, col, ms, zo) in zip(xs, groups):
        ax.errorbar(xp, means[xp], yerr=ci[xp], fmt="o", ms=ms, color=col,
                    ecolor=col, elinewidth=1.3, capsize=3.5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        ax.text(xp, means[xp] + ci[xp] + 0.03, f"{means[xp]:.2f}%",
                ha="center", va="bottom", fontsize=5,
                color=col if col != F.GREY["line"] else F.GREY["muted"])
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("SCAN cortical share (%)")
    ax.set_xlabel("Threat exposure")
    ax.set_xlim(-0.55, 2.55)
    lo = min(m - c for m, c in zip(means, ci))
    hi = max(m + c for m, c in zip(means, ci))
    pad = (hi - lo) * 0.40
    ax.set_ylim(lo - pad, hi + pad)
    ax.text(1.0, hi + pad * 0.6, f"+{pct_inc:.0f}%   d = 0.52",
            ha="center", va="top", fontsize=5.5, fontweight="bold")
    fs.despine(ax)
    fs.panel_label(ax, "f")
    fs.save(fig, "fig1f_dose", outdir=OUT)
    print(f"  fig1f_dose  low={means[0]:.3f}% avg={means[1]:.3f}% "
          f"high={means[2]:.3f}% (+{pct_inc:.1f}%, n_low={len(low)}, "
          f"n_mid={len(mid)}, n_high={len(high)})")


def panel_sizes():
    # Distribution of all 15 network cortical shares at baseline, ordered
    # largest→smallest; SCAN (one of the smallest) highlighted in the accent.
    df = pd.read_csv(F.DAT_DIR / "df_base.csv")
    prop_cols = [f"prop_{n}" for n in F.NET_ORDER]
    data = {n: (df[f"prop_{n}"].dropna().values * 100.0) for n in F.NET_ORDER}
    order = sorted(F.NET_ORDER, key=lambda n: np.median(data[n]), reverse=True)
    series = [data[n] for n in order]

    fig, ax = fs.figure("double", 62)
    parts = ax.violinplot(series, positions=range(len(order)), widths=0.85,
                          showextrema=False, showmedians=False)
    for n, body in zip(order, parts["bodies"]):
        body.set_facecolor(F.net_color(n))
        body.set_edgecolor("black")
        body.set_linewidth(0.3)
        body.set_alpha(0.55 if n != "SCAN" else 0.95)
    # IQR box + median tick inside each violin
    for i, n in enumerate(order):
        q1, med, q3 = np.percentile(data[n], [25, 50, 75])
        ax.plot([i, i], [q1, q3], color="black", lw=2.2, solid_capstyle="butt",
                zorder=3)
        ax.scatter(i, med, s=5, color="white", zorder=4, linewidths=0)

    ax.set_xticks(range(len(order)))
    labels = ax.set_xticklabels(order, rotation=45, ha="right")
    for lab in labels:
        if lab.get_text() == "SCAN":
            lab.set_color(F.SCAN_COLOR)
            lab.set_fontweight("bold")
    ax.set_ylabel("Cortical share (%)")
    ax.set_ylim(0, None)
    fs.despine(ax)
    fs.panel_label(ax, "g")
    fs.save(fig, "fig1g_network_sizes", outdir=OUT)
    scan_rank = order.index("SCAN") + 1
    print(f"  fig1g_network_sizes  SCAN size rank {scan_rank}/15 "
          f"(median {np.median(data['SCAN']):.2f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1h — cross-wave replication (threat–network correlation by wave)
# ─────────────────────────────────────────────────────────────────────────────
def panel_waves():
    # The baseline result is not a fluke: SCAN is the single most threat-
    # sensitive network at baseline, Year 2 AND Year 4 (rank #1/15 at each),
    # riding above the grey pack of the other 14 networks. The association
    # attenuates with age and by Year 6 SCAN settles back into the pack
    # (rank #6) — an honest developmental-attenuation story, not hidden.
    # Uses bivariate threat→network r (phase2), which is robust at every wave
    # (the Year-6 multivariate model has known convergence/attrition issues).
    waves = ["baseline", "year2", "year4", "year6"]
    wlab = ["Baseline", "Year 2", "Year 4", "Year 6"]
    M = pd.DataFrame({w: pd.read_csv(
        F.TAB_DIR / f"phase2_composites_r_matrix_{w}.csv",
        index_col=0).loc["threat_composite"] for w in waves})
    xs = list(range(len(waves)))

    fig, ax = fs.figure("single", 62)
    ax.axhline(0, color="black", lw=0.4, zorder=0)
    for net in F.NET_ORDER:
        if net == "SCAN":
            continue
        ax.plot(xs, M.loc[net, waves].values, color=F.GREY["muted"],
                lw=0.6, alpha=0.5, zorder=1)
    ax.plot(xs, M.loc["SCAN", waves].values, color=F.SCAN_COLOR, lw=1.7,
            marker="o", ms=4.5, markeredgecolor="white", markeredgewidth=0.4,
            zorder=3)
    # flag the waves where SCAN ranks #1 by |r|
    for i, w in enumerate(waves):
        if M[w].abs().rank(ascending=False)["SCAN"] == 1:
            ax.text(i, M.loc["SCAN", w] + 0.010, "#1", ha="center", va="bottom",
                    fontsize=5.5, color=F.SCAN_COLOR, fontweight="bold")
    ax.text(0.97, 0.97, "SCAN", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=F.SCAN_COLOR, fontweight="bold")
    ax.text(0.97, 0.88, "other networks", transform=ax.transAxes, ha="right",
            va="top", fontsize=5, color=F.GREY["muted"])
    ax.set_xticks(xs)
    ax.set_xticklabels(wlab)
    ax.set_ylabel("Correlation with threat  (r)")
    ax.set_xlim(-0.2, len(waves) - 0.8)
    fs.despine(ax)
    fs.panel_label(ax, "h")
    fs.save(fig, "fig1h_waves", outdir=OUT)
    print("  fig1h_waves  SCAN r by wave: " +
          ", ".join(f"{w}={M.loc['SCAN', w]:+.3f}" for w in waves))


if __name__ == "__main__":
    print("Building Figure 1 panels →", OUT)
    panel_matrix()
    panel_saaxis()
    panel_selectivity()
    panel_scatters()
    panel_splithalf()
    panel_sibling()
    panel_dose()
    panel_sizes()
    panel_waves()
    print("Done.")
