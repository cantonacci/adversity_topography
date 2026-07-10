#!/usr/bin/env python3
"""
figure_within_person.py  (v2)
Publication-quality figures for within-person analyses.

Outputs (PDF + 300 DPI PNG in outputs/figures/within_person/):
  panelA_SCAN_trajectories           — SCAN proportion × wave × ELA tertile (no legend)
  panelB_crystallized_trajectories   — Crystallized cognition × wave × ELA tertile (no legend)
  panelC_within_person_path_diagram  — Path diagram (redesigned, clean layout)
  legend_trajectory                  — Shared legend for Panels A & B
"""

import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as _fm
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).parent
DERIV  = HERE / 'derived'
BASE   = Path(__file__).resolve().parents[3]
FIGOUT = BASE / 'outputs' / 'figures' / 'within_person'
FIGOUT.mkdir(parents=True, exist_ok=True)

# ── global style ───────────────────────────────────────────────────────────────
_FONT_PREF = ['Source Sans Pro', 'Source Sans 3', 'Liberation Sans',
              'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
_AVAIL = {f.name for f in _fm.fontManager.ttflist}
_FONT  = next((f for f in _FONT_PREF if f in _AVAIL), 'DejaVu Sans')
print(f'Font: {_FONT}')

mpl.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    _FONT_PREF + ['DejaVu Sans'],
    'font.size':          8,
    'axes.titlesize':     9,
    'axes.labelsize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    8,
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   3.0,
    'ytick.major.size':   3.0,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.08,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})

# ── colour palette (Okabe-Ito) ─────────────────────────────────────────────────
LOW_COLOR   = '#0072B2'   # blue
MID_COLOR   = '#999999'   # grey
HIGH_COLOR  = '#D55E00'   # vermillion
SCAN_COLOR  = '#8E0067'   # atlas SCAN magenta
CRYST_COLOR = '#009E73'   # green
FLUID_COLOR = '#CC79A7'   # pink

TERTILE_COLORS = {'Low': LOW_COLOR, 'Medium': MID_COLOR, 'High': HIGH_COLOR}
TERTILE_LABELS = {'Low': 'Low adversity', 'Medium': 'Sample mean', 'High': 'High adversity'}
TERTILE_ORDER  = ['Low', 'Medium', 'High']

WAVE_YR     = [0, 2, 4, 6]
WAVE_CODES  = {0: '00A', 2: '02A', 4: '04A', 6: '06A'}
WAVE_LABELS = {
    0: 'Baseline\n(~9 yr)',
    2: 'Year 2\n(~11 yr)',
    4: 'Year 4\n(~13 yr)',
    6: 'Year 6\n(~15 yr)',
}


def _tint(hex_color, factor=0.18):
    r, g, b = mpl.colors.to_rgb(hex_color)
    return (1 - factor*(1-r), 1 - factor*(1-g), 1 - factor*(1-b))


def _stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def _fmt(val, p):
    sign = '+' if val >= 0 else ''
    return f'β = {sign}{val:.3f}{_stars(p)}'


def save_fig(fig, name):
    for ext in ('pdf', 'png'):
        fp = FIGOUT / f'{name}.{ext}'
        fig.savefig(fp)
        print(f'  Saved: {fp}')
    plt.close(fig)


# ── load & prep data ──────────────────────────────────────────────────────────
print('Loading data ...')
long = pd.read_csv(DERIV / 'scan_topo_long.csv')
ela  = pd.read_csv(DERIV / 'ela_scores.csv')

long['usable'] = long['usable'].isin([True, 'True', 'TRUE'])
long = long.merge(ela, on='src_subject_id', how='left')

sub_ela  = long.groupby('src_subject_id')['ela_threat'].first().dropna()
q33, q67 = sub_ela.quantile([1/3, 2/3])


def _assign_tert(v):
    if pd.isna(v): return np.nan
    if v <= q33:   return 'Low'
    if v <= q67:   return 'Medium'
    return 'High'


long['tert'] = long['ela_threat'].apply(_assign_tert)
print(f'Tertile cutoffs (z-score): Low ≤ {q33:.2f} < Medium ≤ {q67:.2f} < High')

for t in TERTILE_ORDER:
    ns = [long[(long.wave == WAVE_CODES[yr]) & long.usable &
               (long.tert == t) & long.scan_prop.notna()]
          .src_subject_id.nunique() for yr in WAVE_YR]
    print(f'  {t}: {ns}')


# ── trajectory helper ─────────────────────────────────────────────────────────
def _traj(val_col):
    rows = []
    for yr in WAVE_YR:
        wc  = WAVE_CODES[yr]
        sub = long[(long.wave == wc) & long.usable & long[val_col].notna()]
        for t in TERTILE_ORDER:
            g = sub[sub.tert == t][val_col]
            n = len(g)
            if n < 2:
                rows.append(dict(wave=yr, tert=t, m=np.nan, lo=np.nan, hi=np.nan, n=n))
                continue
            m   = g.mean()
            sem = g.sem()
            rows.append(dict(wave=yr, tert=t, m=m, lo=m-1.96*sem, hi=m+1.96*sem, n=n))
    return pd.DataFrame(rows)


def _draw_traj(ax, val_col):
    traj = _traj(val_col)
    for t in TERTILE_ORDER:
        c  = TERTILE_COLORS[t]
        s  = traj[traj.tert == t].sort_values('wave')
        ls = '--' if t == 'Medium' else '-'
        ax.plot(s['wave'], s['m'], color=c, linewidth=2.0, linestyle=ls,
                marker='o', markersize=4.5, zorder=3, clip_on=False)
        ax.fill_between(s['wave'], s['lo'], s['hi'], color=c, alpha=0.12, zorder=2)

    ax.set_xticks(WAVE_YR)
    ax.set_xticklabels([WAVE_LABELS[w] for w in WAVE_YR])
    ax.set_xlabel('Study wave', labelpad=4)
    return traj


# ═══════════════════════════════════════════════════════════════════════════════
# Shared legend (saved separately)
# ═══════════════════════════════════════════════════════════════════════════════
def panel_legend():
    print('\nLegend')
    handles = []
    for t in TERTILE_ORDER:
        c  = TERTILE_COLORS[t]
        ls = '--' if t == 'Medium' else '-'
        # baseline N from scan_prop (representative)
        n0 = long[(long.wave == '00A') & long.usable & long.scan_prop.notna() &
                  (long.tert == t)].src_subject_id.nunique()
        handles.append(Line2D([0], [0], color=c, linewidth=2, linestyle=ls,
                               marker='o', markersize=5,
                               label=f'{TERTILE_LABELS[t]}  (n = {n0:,})'))

    fig, ax = plt.subplots(figsize=(2.8, 0.9))
    ax.axis('off')
    ax.legend(handles=handles, frameon=False, loc='center',
              handlelength=2.2, handletextpad=0.5, labelspacing=0.45)
    fig.tight_layout(pad=0.1)
    save_fig(fig, 'legend_trajectory')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel A — SCAN proportion trajectories (no legend)
# ═══════════════════════════════════════════════════════════════════════════════
def panel_a():
    print('\nPanel A: SCAN trajectories')
    fig, ax = plt.subplots(figsize=(3.46, 3.15))
    _draw_traj(ax, 'scan_prop')
    ax.set_ylabel('SCAN proportion', labelpad=4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    fig.tight_layout()
    save_fig(fig, 'panelA_SCAN_trajectories')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel B — Crystallized cognition trajectories (no legend)
# ═══════════════════════════════════════════════════════════════════════════════
def panel_b():
    print('\nPanel B: Crystallized cognition trajectories')
    fig, ax = plt.subplots(figsize=(3.46, 3.15))
    _draw_traj(ax, 'cog_cryst')
    ax.set_ylabel('Crystallized cognition\n(age-corrected standard score)', labelpad=4)
    fig.tight_layout()
    save_fig(fig, 'panelB_crystallized_trajectories')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel C — within-person path diagram (redesigned)
#
# Layout (wider figure, generous spacing):
#
#                       [ΔSCAN]
#                      /       \
#   [ELA] ────────────          ──────> [ΔCryst]
#              \      (indirect)        /
#               \                      /
#                \_____________________/ (direct, R3)
#                      \
#                       ────────────────> [ΔFluid]
#
# Node positions chosen to maximise breathing room and avoid arrow crossings.
# ═══════════════════════════════════════════════════════════════════════════════
def panel_c():
    print('\nPanel C: within-person path diagram')

    # All-waves LME results (cluster-robust SE)
    R = {
        'r1':  dict(beta=+0.041, p=0.002),   # ELA  → ΔSCAN     N=6,129
        'r2c': dict(beta=-0.094, p=0.0001),  # ΔSCAN→ ΔCryst    N=3,879  q<0.001
        'r2f': dict(beta=-0.076, p=0.033),   # ΔSCAN→ ΔFluid    N=1,032  q=0.033
        'r3':  dict(beta=-0.119, p=0.0001),  # ELA  → ΔCryst    N=3,879  (direct)
    }

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ── node geometry: (cx, cy, half-width, half-height) ──────────────────────
    # Generous horizontal spacing; SCAN elevated; CRYST/FLUID well separated
    NB = {
        'ELA':   (0.11, 0.51, 0.095, 0.105),
        'SCAN':  (0.50, 0.82, 0.115, 0.095),
        'CRYST': (0.88, 0.72, 0.110, 0.090),
        'FLUID': (0.88, 0.22, 0.110, 0.090),
    }
    # Edge shortcuts
    def left(k):  return NB[k][0] - NB[k][2]
    def right(k): return NB[k][0] + NB[k][2]
    def top(k):   return NB[k][1] + NB[k][3]
    def bot(k):   return NB[k][1] - NB[k][3]
    def cx(k):    return NB[k][0]
    def cy(k):    return NB[k][1]

    # ── draw boxes ─────────────────────────────────────────────────────────────
    def _box(key, lines, fc, ec, lw=1.7):
        x0, y0, w, h = (left(key), bot(key),
                        2*NB[key][2], 2*NB[key][3])
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle='round,pad=0.018', lw=lw,
            edgecolor=ec, facecolor=fc, zorder=3, clip_on=False))
        ax.text(cx(key), cy(key), lines, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#1a1a1a',
                zorder=4, multialignment='center')

    _box('ELA',   'Early\nThreat',        _tint(HIGH_COLOR),  HIGH_COLOR)
    _box('SCAN',  'ΔSCAN\nProportion',    _tint(SCAN_COLOR),  SCAN_COLOR)
    _box('CRYST', 'Δ Crystallized\nCognition', _tint(CRYST_COLOR), CRYST_COLOR)
    _box('FLUID', 'Δ Fluid\nCognition',   _tint(FLUID_COLOR), FLUID_COLOR)

    # ── draw arrows ────────────────────────────────────────────────────────────
    def _arrow(xy_from, xy_to, color, lw=1.7, rad=0.0, ls='->'):
        ax.annotate('', xy=xy_to, xytext=xy_from,
                    arrowprops=dict(arrowstyle=ls, color=color, lw=lw,
                                    connectionstyle=f'arc3,rad={rad}'),
                    zorder=5)

    def _blabel(x, y, txt, color, fs=8.0):
        ax.text(x, y, txt, ha='center', va='center',
                fontsize=fs, color=color, fontweight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.20', facecolor='white',
                          edgecolor='none', alpha=0.95))

    # 1. ELA → SCAN  (R1: β=+0.041**)
    _arrow((right('ELA'), cy('ELA')+0.05),
           (left('SCAN'),  cy('SCAN')-0.04),
           HIGH_COLOR, rad=-0.12)
    _blabel(0.280, 0.730, _fmt(R['r1']['beta'], R['r1']['p']), HIGH_COLOR)

    # 2. SCAN → CRYST  (R2: β=-0.094***)
    _arrow((right('SCAN'), cy('SCAN')-0.01),
           (left('CRYST'), cy('CRYST')+0.03),
           SCAN_COLOR, rad=0.10)
    _blabel(0.695, 0.840, _fmt(R['r2c']['beta'], R['r2c']['p']), SCAN_COLOR)

    # 3. SCAN → FLUID  (R2: β=-0.076*)
    _arrow((right('SCAN'), cy('SCAN')-0.07),
           (left('FLUID'), cy('FLUID')+0.06),
           SCAN_COLOR, lw=1.3, rad=-0.18)
    _blabel(0.760, 0.560, _fmt(R['r2f']['beta'], R['r2f']['p']), SCAN_COLOR, fs=7.5)

    # 4. ELA → CRYST  (R3: β=-0.119*** direct effect, separate model)
    # Curved below SCAN box; arc midpoint ≈ (0.50, 0.46) — clear of SCAN
    _arrow((right('ELA'), cy('ELA')-0.06),
           (left('CRYST'), cy('CRYST')-0.05),
           HIGH_COLOR, lw=1.2, rad=-0.28)
    # Label placed along the arc, below center of figure
    _blabel(0.490, 0.400,
            _fmt(R['r3']['beta'], R['r3']['p']),
            HIGH_COLOR, fs=7.5)
    # Small "direct" annotation near label
    ax.text(0.490, 0.358, '(direct effect, Result 3)',
            ha='center', va='top', fontsize=5.8, color=HIGH_COLOR,
            style='italic', zorder=6,
            bbox=dict(boxstyle='round,pad=0.10', facecolor='white',
                      edgecolor='none', alpha=0.90))

    # ── method note (sole line at bottom, well clear of all elements) ──────────
    ax.text(0.50, 0.03,
            'All-waves MixedLM, cluster-robust SE.  '
            '*** p < 0.001   ** p < 0.01   * p < 0.05',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=5.8, color='#888888', style='italic')

    fig.tight_layout()
    save_fig(fig, 'panelC_within_person_path_diagram')


# ── run ───────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('Within-person publication figures  (v2)')
print('='*60)

panel_legend()
panel_a()
panel_b()
panel_c()

print(f'\nAll figures saved to: {FIGOUT}')
