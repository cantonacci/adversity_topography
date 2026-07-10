"""
figstyle.py — Publication-figure toolkit for Nature/Science-grade panels.
================================================================================
Drop this file (plus nature.mplstyle, same folder) into your repo. Then in any
analysis/plotting script:

    import figstyle as fs
    fs.set_style()                          # apply the house style ONCE, at top

    fig, ax = fs.figure("single", 50)       # real-world sizing in mm
    ax.plot(x, y, color=fs.WONG["blue"])
    ax.set_xlabel("Time (s)")               # ALWAYS units in parentheses
    ax.set_ylabel("Response (a.u.)")
    fs.panel_label(ax, "a")                 # 8 pt bold lowercase, top-left
    fs.save(fig, "fig2a")                   # -> fig2a.pdf (vector) + fig2a.png

Design philosophy: this produces *clean, correct, editable* single panels at
final print size. Intricacy and multi-panel composition happen afterward in
Illustrator. Aim for "boring-but-pristine vector panel," not "finished figure."

Everything here encodes a specific Nature Portfolio requirement (2026 specs).
Nothing is decorative.
"""
from __future__ import annotations

import os
import warnings
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ============================================================================ #
# 1. CANONICAL DIMENSIONS  (Nature Portfolio, 2026)
# ============================================================================ #
MM = 1.0 / 25.4  # multiply a millimetre value by MM to get inches

# Printed column widths — build at ONE of these, never an arbitrary width.
WIDTH_SINGLE = 89.0   # mm — single column (most panels / simple plots)
WIDTH_ONEHALF = 120.0  # mm — 1.5 column (wide single panels, ~120–136 mm)
WIDTH_DOUBLE = 183.0  # mm — double column (full-width multi-panel figures)
MAX_HEIGHT = 170.0    # mm — max height for a Nature main figure (leaves room
#                              for the legend below). Keep panels compact.

_WIDTH_ALIASES = {
    "single": WIDTH_SINGLE, "1": WIDTH_SINGLE, "column": WIDTH_SINGLE,
    "1.5": WIDTH_ONEHALF, "onehalf": WIDTH_ONEHALF, "1.5col": WIDTH_ONEHALF,
    "double": WIDTH_DOUBLE, "2": WIDTH_DOUBLE, "full": WIDTH_DOUBLE,
}

# ============================================================================ #
# 2. COLOUR  —  Wong colourblind-safe palette (Nature's own recommendation)
#    Wong, B. Points of view: Colour blindness. Nat. Methods 8, 441 (2011).
#    Use these. Avoid red/green pairings and rainbow/jet scales.
# ============================================================================ #
WONG = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",  # "bluish green"
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermillion": "#D55E00",
    "purple":  "#CC79A7",  # "reddish purple"
}
# Ordered cycle used by default (high-contrast pairs come first).
WONG_CYCLE = [WONG["blue"], WONG["orange"], WONG["green"], WONG["purple"],
              WONG["skyblue"], WONG["vermillion"], WONG["yellow"], WONG["black"]]

# Perceptually-uniform sequential / diverging defaults for heatmaps & density.
# (viridis/magma/cividis are colourblind-safe; never use 'jet' or 'rainbow'.)
SEQUENTIAL = "viridis"
SEQUENTIAL_ALT = "magma"
DIVERGING = "RdBu_r"   # acceptable; for strict CB-safety prefer "vlag" via seaborn

GREY = {"line": "#333333", "fill": "#BBBBBB", "muted": "#888888"}


def palette(n: int) -> list[str]:
    """Return the first n colourblind-safe hex colours from the Wong cycle."""
    if n > len(WONG_CYCLE):
        warnings.warn(
            f"{n} colours requested but only {len(WONG_CYCLE)} colourblind-safe "
            "hues exist. Distinguish extra series by marker/linestyle, or use a "
            "sequential colourmap — do not add ambiguous colours."
        )
    return (WONG_CYCLE * (n // len(WONG_CYCLE) + 1))[:n]


# ============================================================================ #
# 3. STYLE ACTIVATION
# ============================================================================ #
_STYLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "nature.mplstyle")


def _resolve_sans() -> None:
    """Make sure a real sans-serif is used; warn if Arial/Helvetica absent."""
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    preferred = ["Arial", "Helvetica", "Helvetica Neue", "Nimbus Sans",
                 "Liberation Sans", "DejaVu Sans"]
    chosen = next((f for f in preferred if f in available), None)
    if chosen not in ("Arial", "Helvetica"):
        warnings.warn(
            "Arial/Helvetica not found; falling back to "
            f"'{chosen}'. For true Nature compliance install Arial or Helvetica "
            "on this machine (the text must be a standard sans-serif). The PDF "
            "will still have editable text."
        )


def set_style(style_path: str | None = None) -> None:
    """Apply the Nature house style. Call ONCE near the top of a script.

    Loads nature.mplstyle if present (preferred); otherwise applies an
    equivalent rcParams set in code so the module is self-contained.
    """
    path = style_path or _STYLE_PATH
    if os.path.exists(path):
        plt.style.use(path)
    else:  # self-contained fallback (keeps the most load-bearing rules)
        mpl.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans",
                                 "Liberation Sans", "DejaVu Sans"],
            "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
            "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
            "axes.linewidth": 0.5, "axes.spines.top": False,
            "axes.spines.right": False, "axes.grid": False,
            "axes.titlelocation": "left", "axes.titleweight": "bold",
            "xtick.direction": "out", "ytick.direction": "out",
            "xtick.major.size": 2.5, "ytick.major.size": 2.5,
            "xtick.major.width": 0.5, "ytick.major.width": 0.5,
            "lines.linewidth": 1.0, "lines.markersize": 3.0,
            "errorbar.capsize": 2.0, "legend.frameon": False,
            "figure.dpi": 150, "savefig.dpi": 600,
            "savefig.transparent": True, "pdf.fonttype": 42,
            "ps.fonttype": 42, "svg.fonttype": "none",
            "figure.constrained_layout.use": True,
            "axes.prop_cycle": mpl.cycler(color=WONG_CYCLE),
        })
    _resolve_sans()


# ============================================================================ #
# 4. FIGURE CONSTRUCTION  (sized in real millimetres)
# ============================================================================ #
def figure(width: str | float = "single", height_mm: float | None = None,
           **subplot_kw) -> tuple[Figure, Axes]:
    """Create a single-axes figure at a real printed size.

    width     : "single" (89 mm), "1.5" (120 mm), "double" (183 mm), or a number
                in mm. Build at the size the panel will actually be PRINTED so
                7 pt type stays 7 pt — wrong size is the #1 amateur tell.
    height_mm : height in mm. Defaults to a balanced ratio for the chosen width.
    """
    w_mm = _WIDTH_ALIASES.get(str(width).lower(), None)
    if w_mm is None:
        w_mm = float(width)  # treat as explicit mm
    if height_mm is None:
        height_mm = w_mm * 0.72  # pleasant default; adjust per panel
    if height_mm > MAX_HEIGHT:
        warnings.warn(f"height {height_mm} mm exceeds Nature max ({MAX_HEIGHT} mm).")
    fig, ax = plt.subplots(figsize=(w_mm * MM, height_mm * MM), **subplot_kw)
    return fig, ax


def grid(width: str | float = "double", height_mm: float | None = None,
         nrows: int = 1, ncols: int = 2, **kw):
    """Multi-axes figure at real size. Prefer assembling separate single-panel
    PDFs in Illustrator, but this is handy for tightly-coupled subplots."""
    w_mm = _WIDTH_ALIASES.get(str(width).lower(), None) or float(width)
    if height_mm is None:
        height_mm = min(w_mm * 0.4 * nrows, MAX_HEIGHT)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w_mm * MM, height_mm * MM), **kw)
    return fig, axes


# ============================================================================ #
# 5. PANEL POLISH
# ============================================================================ #
def panel_label(ax: Axes, letter: str, x: float = -0.18, y: float = 1.08,
                **kw) -> None:
    """Add a panel label: 8 pt BOLD, lowercase, upright (Nature spec).
    Place in the panel's top-left. Tweak x/y for panels with wide tick labels."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", fontstyle="normal", va="bottom", ha="right",
            **kw)


def despine(ax: Axes, trim: bool = True, offset: float = 0.0) -> None:
    """Remove top/right spines (the style does this already) and, if trim=True,
    clip remaining spines to the data range — the clean 'open axes' look."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if offset:
        for s in ("left", "bottom"):
            ax.spines[s].set_position(("outward", offset))
    if trim:
        xt, yt = ax.get_xticks(), ax.get_yticks()
        if len(xt):
            x0, x1 = ax.get_xlim()
            vis = [t for t in xt if x0 <= t <= x1]
            if vis:
                ax.spines["bottom"].set_bounds(min(vis), max(vis))
        if len(yt):
            y0, y1 = ax.get_ylim()
            vis = [t for t in yt if y0 <= t <= y1]
            if vis:
                ax.spines["left"].set_bounds(min(vis), max(vis))


def sig_bracket(ax: Axes, x1: float, x2: float, y: float, text: str = "*",
                height: float = 0.02, lw: float = 0.5, fontsize: float = 6,
                color: str = "#000000") -> None:
    """Draw a significance bracket spanning x1..x2 at height y (data coords),
    with a centred annotation (e.g. '*', '**', 'n.s.'). height is the tick drop
    as a fraction of the y-range."""
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = height * yr
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, color=color,
            clip_on=False, solid_capstyle="butt")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=fontsize, color=color)


def add_scalebar(ax: Axes, length: float, label: str, loc: str = "lower right",
                 pad: float = 0.05, lw: float = 1.5, color: str = "white",
                 fontsize: float = 6) -> None:
    """Add a scale bar to an image panel. Keep it (and its label) as separate
    artists — Nature wants scale bars editable, never flattened into pixels."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    yb = min(y0, y1)
    yr = abs(y1 - y0)
    xr = abs(x1 - x0)
    px = pad * xr
    py = pad * yr
    if "right" in loc:
        xend = max(x0, x1) - px
        xstart = xend - length
    else:
        xstart = min(x0, x1) + px
        xend = xstart + length
    ybar = (yb + py) if "lower" in loc else (max(y0, y1) - py)
    ax.plot([xstart, xend], [ybar, ybar], lw=lw, color=color,
            solid_capstyle="butt", clip_on=False)
    ax.text((xstart + xend) / 2, ybar + 0.015 * yr, label, color=color,
            ha="center", va="bottom", fontsize=fontsize)


def direct_label(ax: Axes, x: float, y: float, text: str, color: str,
                 **kw) -> None:
    """Label a series directly at its end instead of using a legend (cleaner,
    avoids the eye round-tripping to a key). Colour the text to match the line
    ONLY when it carries meaning; otherwise prefer black + a key swatch."""
    ax.text(x, y, text, color=color, va="center", ha="left", fontsize=6,
            clip_on=False, **kw)


# ============================================================================ #
# 6. EXPORT  (vector-first, editable text, RGB)
# ============================================================================ #
def save(fig: Figure, name: str, outdir: str = "figures",
         formats: Sequence[str] = ("pdf", "png"), dpi: int = 600,
         transparent: bool = True, tight: bool = False,
         close: bool = True) -> list[str]:
    """Export a panel.

    - PDF/EPS are vector with EDITABLE text (pdf.fonttype 42) — submit these.
    - PNG at 600 dpi is a raster preview for slides / quick looks.
    - tight=False (default) preserves your EXACT requested width. Set tight=True
      only when stray whitespace must be cropped (it changes the final width).

    Returns the list of written paths.
    """
    os.makedirs(outdir, exist_ok=True)
    paths = []
    save_kw = dict(dpi=dpi, transparent=transparent)
    if tight:
        save_kw.update(bbox_inches="tight", pad_inches=0.01)
    for ext in formats:
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, **save_kw)
        paths.append(p)
    if close:
        plt.close(fig)
    return paths


# ============================================================================ #
# 7. SELF-TEST / DEMO
# ============================================================================ #
def demo(outdir: str = "figures") -> list[str]:
    """Render a couple of example panels so you can eyeball the house style."""
    import numpy as np
    set_style()
    rng = np.random.default_rng(0)

    # --- Panel a: grouped bar + individual points + SEM + significance --------
    figa, ax = figure("single", 50)
    groups = ["Ctrl", "Drug A", "Drug B"]
    data = [rng.normal(m, 1.0, 12) for m in (5.0, 6.6, 5.4)]
    means = [d.mean() for d in data]
    sems = [d.std(ddof=1) / len(d) ** 0.5 for d in data]
    xpos = range(len(groups))
    cols = palette(len(groups))
    ax.bar(xpos, means, width=0.62, color=cols, edgecolor="black",
           linewidth=0.5, zorder=1)
    ax.errorbar(xpos, means, yerr=sems, fmt="none", ecolor="black",
                elinewidth=0.5, capsize=2, zorder=2)
    for xp, d in zip(xpos, data):
        ax.scatter(rng.normal(xp, 0.05, len(d)), d, s=4, color="black",
                   alpha=0.6, linewidths=0, zorder=3)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Response (a.u.)")
    ax.set_ylim(0, 9)
    sig_bracket(ax, 0, 1, 8.0, "**")
    despine(ax)
    panel_label(ax, "a")
    paths = save(figa, "demo_a", outdir=outdir)

    # --- Panel b: line series with shaded error + direct labels ---------------
    figb, ax = figure("single", 50)
    x = np.linspace(0, 10, 100)
    for name_, c, k in (("WT", WONG["blue"], 1.0), ("Mutant", WONG["orange"], 0.6)):
        y = k * (1 - np.exp(-x / 2)) + rng.normal(0, 0.01, x.size)
        sd = 0.05 * np.ones_like(x)
        ax.plot(x, y, color=c, lw=1.0)
        ax.fill_between(x, y - sd, y + sd, color=c, alpha=0.18, linewidth=0)
        direct_label(ax, x[-1] + 0.1, y[-1], name_, c)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Signal (ΔF/F)")
    ax.set_xlim(0, 11.5)
    despine(ax)
    panel_label(ax, "b")
    paths += save(figb, "demo_b", outdir=outdir)
    return paths


if __name__ == "__main__":
    out = demo()
    print("Wrote:")
    for p in out:
        print("  ", p)
