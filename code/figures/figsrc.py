"""
figsrc.py — project-specific figure layer for the ELA × SCAN paper.

Sits ON TOP of the house style (figstyle.py + nature.mplstyle, in this folder).
figstyle owns the Nature mechanics (fonts, sizes, spines, export); this module
owns only the domain choices figstyle cannot know: the canonical 15-network
order and colours, the SCAN accent, composite/ELA label maps, paths, and a few
small helpers. Import this, not figstyle, from panel scripts.

    import figsrc as F
    F.fs.set_style()
    fig, ax = F.fs.figure("single", 50)
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# ── locate the house style (co-located here) and the analysis config ─────────
ROOT       = Path(__file__).resolve().parents[2]

import figstyle as fs   # noqa: E402  (the house style toolkit)
from adtopo.config import (    # noqa: E402
    NETWORKS, NET_GROUPS, NET_GROUP_MAP,
    COMPOSITE_COLS, COMPOSITE_LABELS,
    ELA_COLS, ELA_LABELS_SHORT,
)

# ── paths ────────────────────────────────────────────────────────────────────
TAB_DIR = ROOT / "outputs" / "tables"
DAT_DIR = ROOT / "outputs" / "data_processed"
WP_DIR  = ROOT / "code" / "05_behavior" / "within_person" / "derived"
FIG_OUT = ROOT / "outputs" / "figures" / "nn"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# ── network order & colour key (consistent across ALL figures) ───────────────
# Group order chosen so related systems sit together; SCAN sits last for emphasis.
GROUP_ORDER = ["Transmodal", "Limbic/Salience", "Unimodal", "Other/Assoc"]
NET_ORDER   = [n for g in GROUP_ORDER for n in NET_GROUPS[g]]   # 15, SCAN last

# Wong colourblind-safe hues mapped onto the four functional groups (kept as a
# fallback / optional grouping key).
GROUP_COLOR = {
    "Transmodal":      fs.WONG["blue"],
    "Limbic/Salience": fs.WONG["orange"],
    "Unimodal":        fs.WONG["skyblue"],
    "Other/Assoc":     fs.WONG["green"],
}

# Canonical ABCD Template-Matching-V2 atlas colours (match the dlabel RGBA used
# in the Connectome-Workbench surface renders, so violin/bar/scatter colours
# correspond to the brain figures). These are the network identity colours.
ATLAS_NET_COLORS = {
    "DMN": "#FF0000", "VIS": "#000099", "FP": "#FFFF00",
    "DAN": "#00FF00", "VAN": "#0D85A0", "SAL": "#000000",
    "CO":  "#6600CC", "SMD": "#66FFFF", "SML": "#FF8000",
    "AUD": "#B266FF", "Tpole": "#006699", "MTL": "#66FF66",
    "PMN": "#3C3CFB", "PON": "#EFEFEF", "SCAN": "#8E0067",
}

SCAN_COLOR   = ATLAS_NET_COLORS["SCAN"]   # #8E0067 — SCAN accent = its atlas colour
THREAT_HIGH  = fs.WONG["vermillion"]      # high adversity (threat-contrast semantic)
THREAT_LOW   = fs.WONG["blue"]            # low adversity
NEUTRAL_BAR  = fs.GREY["fill"]            # #BBBBBB for de-emphasised bars


def net_color(net: str) -> str:
    """Canonical atlas colour for a network (matches the surface renders)."""
    return ATLAS_NET_COLORS[net]


# Networks whose atlas colour is near-white (PON = #EFEFEF) and so vanish on a
# white figure background. Filled markers (bars/dots/violins) already carry a thin
# black edge, so they read fine; for the few line-drawn glyphs that look bad with a
# stroke (the fig3 chord arcs) we instead render pale networks a touch darker.
PALE_NETS = {"PON"}


def group_color(net: str) -> str:
    """Functional-group colour (Wong) — optional alternative to atlas colours."""
    return GROUP_COLOR[NET_GROUP_MAP[net]]


def star(q: float, p: float | None = None, bonf: float | None = None) -> str:
    """Significance marker: '**' Bonferroni, '*' FDR q<0.05, else ''."""
    if bonf is not None and p is not None and p < bonf:
        return "**"
    if q is not None and q < 0.05:
        return "*"
    return ""


# convenience re-exports
WONG = fs.WONG
GREY = fs.GREY
