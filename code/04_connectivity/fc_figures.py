#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fc_figures.py  —  Publication-quality figures for FC ~ ELA analyses.

Produces:
  1. fc_matrix_beta.pdf/png        — 15x15 beta (threat→FC) heatmap; SCAN highlighted
  2. fc_matrix_group_means.pdf/png — 3-panel: high-ELA / low-ELA / difference FC
  3. fc_chord_scan.pdf/png         — chord diagram of SCAN's 14 connections
  4. fc_scan_surface_overlay.dscalar.nii — Workbench overlay (beta per network)
     + fc_scan_surface_overlay_static.pdf — static rendering with network outlines

Usage:
  python fc_figures.py
"""

import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import nibabel as nib

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import DAT_DIR, NETWORKS, ATLAS_DIR

# ── paths ─────────────────────────────────────────────────────────────────────
FC_PATH    = DAT_DIR / 'fc_ses-00A.csv'
RES_PATH   = Path(__file__).parent.parent / 'outputs' / 'tables' / 'fc_lme_threat_baseline.csv'
FIG_DIR    = Path(__file__).parent.parent / 'outputs' / 'figures' / 'fc'
CIFTI_DIR  = Path(__file__).parent.parent / 'outputs' / 'cifti_for_workbench'
ATLAS_PATH = ATLAS_DIR / 'abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii'

# ── network config ─────────────────────────────────────────────────────────────
NET_NAMES = list(NETWORKS)
N         = len(NET_NAMES)
NET_IDX   = {n: i for i, n in enumerate(NET_NAMES)}

# Canonical ReproTM/atlas colors
NET_COLOR = {
    'DMN':   '#FF0000', 'VIS':   '#000099', 'FP':    '#FFFF00',
    'DAN':   '#00FF00', 'VAN':   '#0D85A0', 'SAL':   '#000000',
    'CO':    '#6600CC', 'SMD':   '#66FFFF', 'SML':   '#FF8000',
    'AUD':   '#B266FF', 'Tpole': '#006699', 'MTL':   '#66FF66',
    'PMN':   '#3C3CFB', 'PON':   '#EFEFEF', 'SCAN':  '#8E0067',
}

# Display order for FC matrix: unimodal → attention → salience/SCAN → control → transmodal
DISPLAY_ORDER = ['VIS', 'SMD', 'SML', 'AUD', 'DAN', 'VAN',
                 'SAL', 'SCAN', 'CO', 'FP', 'DMN', 'MTL', 'Tpole', 'PMN', 'PON']
DISP_IDX = [NET_IDX[n] for n in DISPLAY_ORDER]

NET_LABEL = {
    'DMN': 1, 'VIS': 2, 'FP': 3, 'DAN': 5, 'VAN': 7,
    'SAL': 8, 'CO': 9, 'SMD': 10, 'SML': 11, 'AUD': 12,
    'Tpole': 13, 'MTL': 14, 'PMN': 15, 'PON': 16, 'SCAN': 18,
}
N_CORT = 59412
N_FULL = 91282

# ELA group thresholds (consistent with encroachment analysis)
HIGH_THRESH = 1.0   # threat_composite >= +1 SD
LOW_THRESH  = -1.0  # threat_composite <= -1 SD

# Nature figure dimensions (mm → inches)
COL1 = 89  / 25.4   # single column
COL2 = 183 / 25.4   # double column

# ── style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':        7,
    'axes.titlesize':   8,
    'axes.labelsize':   7,
    'xtick.labelsize':  6,
    'ytick.labelsize':  6,
    'axes.linewidth':   0.5,
    'xtick.major.width':0.5,
    'ytick.major.width':0.5,
    'lines.linewidth':  0.75,
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
    'figure.dpi':       300,
})


def log(msg): print(msg, flush=True)


def save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = FIG_DIR / f'{stem}.{ext}'
        fig.savefig(str(p), dpi=300, bbox_inches='tight', facecolor='white')
    log(f'  → {FIG_DIR / stem}.pdf/png')
    plt.close(fig)


# ── helpers ────────────────────────────────────────────────────────────────────

def fc_wide_to_matrix(row, net_names):
    """Reconstruct (N, N) symmetric matrix from a row of fc_ columns."""
    n  = len(net_names)
    fc = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            col = f'fc_{net_names[i]}_{net_names[j]}'
            if col in row.index:
                v = row[col]
                fc[i, j] = v
                fc[j, i] = v
    return fc


def group_mean_fc(df_fc, df_cov, group_col, threshold, above=True):
    """Mean FC matrix for subjects above/below threshold on group_col."""
    mask = df_cov[group_col] >= threshold if above else df_cov[group_col] <= threshold
    subs = df_cov.loc[mask, 'sub_ID']
    sub_fc = df_fc[df_fc['sub_ID'].isin(subs)]

    mats = []
    fc_cols = [c for c in df_fc.columns if c.startswith('fc_')]
    for _, row in sub_fc.iterrows():
        mat = fc_wide_to_matrix(row[fc_cols], NET_NAMES)
        mats.append(mat)

    log(f'    Group N = {len(mats)}')
    return np.nanmean(np.stack(mats), axis=0) if mats else np.full((N, N), np.nan)


def results_to_matrix(df_res, col):
    """Map a results-table column onto a (N, N) symmetric matrix."""
    mat = np.full((N, N), np.nan)
    for _, row in df_res.iterrows():
        i = NET_IDX[row['net1']]
        j = NET_IDX[row['net2']]
        mat[i, j] = row[col]
        mat[j, i] = row[col]
    return mat


def reorder(mat, order):
    """Reorder a square matrix by index list."""
    return mat[np.ix_(order, order)]


# ── Figure 1: beta matrix ──────────────────────────────────────────────────────

def fig_beta_matrix(df_res):
    log('Figure 1: beta matrix')

    beta_mat = reorder(results_to_matrix(df_res, 'beta'), DISP_IDX)
    sig_all  = reorder(results_to_matrix(df_res, 'sig_all').astype(float),  DISP_IDX)
    sig_scan = reorder(results_to_matrix(df_res, 'sig_scan').astype(float), DISP_IDX)

    # color scale: symmetric around 0
    bmax = np.nanpercentile(np.abs(beta_mat), 97)
    bmax = max(bmax, 0.01)
    norm = TwoSlopeNorm(vmin=-bmax, vcenter=0, vmax=bmax)
    cmap = 'RdBu_r'

    fig, ax = plt.subplots(1, 1, figsize=(COL2 * 0.55, COL2 * 0.55))

    im = ax.imshow(beta_mat, cmap=cmap, norm=norm, aspect='equal')

    # SCAN row/column highlight
    scan_disp = DISPLAY_ORDER.index('SCAN')
    rect_row = mpatches.Rectangle((-0.5, scan_disp - 0.5), N, 1,
                                   lw=1.2, edgecolor='#8E0067', facecolor='none')
    rect_col = mpatches.Rectangle((scan_disp - 0.5, -0.5), 1, N,
                                   lw=1.2, edgecolor='#8E0067', facecolor='none')
    ax.add_patch(rect_row)
    ax.add_patch(rect_col)

    # significance markers: all-pairs FDR
    for i in range(N):
        for j in range(i + 1, N):
            if sig_all[i, j] == 1:
                ax.text(j, i, '●', ha='center', va='center',
                        fontsize=5, color='black', alpha=0.7)
                ax.text(i, j, '●', ha='center', va='center',
                        fontsize=5, color='black', alpha=0.7)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(DISPLAY_ORDER, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(DISPLAY_ORDER, fontsize=6)

    # Color the tick labels by network
    for tick, net in zip(ax.get_xticklabels(), DISPLAY_ORDER):
        tick.set_color(NET_COLOR[net] if net != 'SAL' else '#555555')
    for tick, net in zip(ax.get_yticklabels(), DISPLAY_ORDER):
        tick.set_color(NET_COLOR[net] if net != 'SAL' else '#555555')

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('β  (threat composite → FC)', fontsize=7)
    cb.ax.tick_params(labelsize=6)

    ax.set_title('Network FC — effect of early-life adversity (threat)', fontsize=8, pad=6)

    # Legend for significance marker
    ax.text(1.15, 0.5, '● FDR q<0.05\n   (all pairs)',
            transform=ax.transAxes, fontsize=5.5, va='center')

    fig.tight_layout()
    save(fig, 'fc_matrix_beta')


# ── Figure 2: group mean matrices ─────────────────────────────────────────────

def fig_group_means(df_fc, df_cov):
    log('Figure 2: group mean FC matrices')

    log('  High ELA group ...')
    mat_hi  = group_mean_fc(df_fc, df_cov, 'threat_composite', HIGH_THRESH, above=True)
    log('  Low ELA group ...')
    mat_lo  = group_mean_fc(df_fc, df_cov, 'threat_composite', LOW_THRESH, above=False)
    mat_diff = mat_hi - mat_lo

    mat_hi   = reorder(mat_hi,   DISP_IDX)
    mat_lo   = reorder(mat_lo,   DISP_IDX)
    mat_diff = reorder(mat_diff, DISP_IDX)

    # shared scale for hi/lo
    vabs = np.nanpercentile(np.abs(np.concatenate([mat_hi.ravel(), mat_lo.ravel()])), 99)
    vabs = max(vabs, 0.1)

    dabs = np.nanpercentile(np.abs(mat_diff.ravel()), 99)
    dabs = max(dabs, 0.05)

    fig, axes = plt.subplots(1, 3, figsize=(COL2, COL2 * 0.38))
    titles = ['High adversity\n(threat ≥ +1 SD)',
              'Low adversity\n(threat ≤ −1 SD)',
              'Difference\n(high − low)']
    mats   = [mat_hi, mat_lo, mat_diff]
    vnorms = [TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs),
              TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs),
              TwoSlopeNorm(vmin=-dabs, vcenter=0, vmax=dabs)]
    cmaps  = ['RdBu_r', 'RdBu_r', 'PiYG']

    for ax, mat, title, norm, cmap in zip(axes, mats, titles, vnorms, cmaps):
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='equal')

        scan_disp = DISPLAY_ORDER.index('SCAN')
        ax.add_patch(mpatches.Rectangle((-0.5, scan_disp - 0.5), N, 1,
                                         lw=1.0, edgecolor='#8E0067', facecolor='none'))
        ax.add_patch(mpatches.Rectangle((scan_disp - 0.5, -0.5), 1, N,
                                         lw=1.0, edgecolor='#8E0067', facecolor='none'))

        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(DISPLAY_ORDER, rotation=45, ha='right', fontsize=5)
        ax.set_yticklabels(DISPLAY_ORDER, fontsize=5)
        ax.set_title(title, fontsize=7, pad=4)

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=5)

    fig.suptitle('Network-level FC: Fisher-z transformed Pearson r', fontsize=8, y=1.01)
    fig.tight_layout()
    save(fig, 'fc_matrix_group_means')


# ── Figure 3: SCAN chord diagram ───────────────────────────────────────────────

def fig_chord_scan(df_res):
    log('Figure 3: SCAN chord diagram')

    scan_res = df_res[(df_res['net1'] == 'SCAN') | (df_res['net2'] == 'SCAN')].copy()
    scan_res['partner'] = scan_res.apply(
        lambda r: r['net2'] if r['net1'] == 'SCAN' else r['net1'], axis=1
    )
    # partner order: same as DISPLAY_ORDER minus SCAN
    partner_order = [n for n in DISPLAY_ORDER if n != 'SCAN']
    scan_res = scan_res.set_index('partner').reindex(partner_order).reset_index()

    betas = scan_res['beta'].values.astype(float)
    sigs  = scan_res['sig_scan'].fillna(False).values.astype(bool)
    n_p   = len(partner_order)

    fig, ax = plt.subplots(1, 1, figsize=(COL2 * 0.55, COL2 * 0.55),
                           subplot_kw=dict(projection='polar'))

    # Positions: SCAN at top (0), partners evenly distributed
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi * n_p / (n_p + 1),
                         n_p, endpoint=False)
    scan_angle = np.pi / 2 + 2 * np.pi / (n_p + 1)
    all_angles = np.append(angles, scan_angle)
    all_names  = partner_order + ['SCAN']

    R_outer = 1.0
    R_label = 1.15
    R_node  = 0.05

    # Draw network nodes
    for ang, name in zip(all_angles, all_names):
        col = NET_COLOR.get(name, '#888888')
        if name == 'SAL':
            col = '#555555'
        ax.scatter(ang, R_outer, s=60, color=col, zorder=5,
                   edgecolors='white', linewidths=0.5)
        ax.text(ang, R_label, name, ha='center', va='center',
                fontsize=6, color=col if name != 'SAL' else '#333333',
                fontweight='bold' if name == 'SCAN' else 'normal')

    # Draw chords
    bmax = np.nanmax(np.abs(betas))
    bmax = max(bmax, 0.001)

    pos_cmap = plt.cm.Reds
    neg_cmap = plt.cm.Blues_r

    for ang, name, beta, sig in zip(angles, partner_order, betas, sigs):
        if np.isnan(beta):
            continue
        lw    = 0.5 + 3.0 * abs(beta) / bmax
        alpha = 0.85 if sig else 0.35
        color = pos_cmap(0.5 + 0.4 * abs(beta) / bmax) if beta > 0 else \
                neg_cmap(0.5 + 0.4 * abs(beta) / bmax)

        # Bezier chord via a straight line in polar (approximate)
        from matplotlib.patches import FancyArrowPatch
        import matplotlib.patches as mpa
        # Draw a curved line from scan_angle to partner angle through the interior
        theta = np.linspace(scan_angle, ang, 60)
        r_chord = np.full(60, 0.55 + 0.25 * (1 - abs(beta) / bmax))
        # blend from outer to center back to outer
        r_chord = R_outer * np.abs(np.sin(np.linspace(0, np.pi, 60)))
        r_chord = np.clip(r_chord, 0.05, R_outer)
        ax.plot(theta, r_chord, color=color, lw=lw, alpha=alpha, solid_capstyle='round')

        if sig:
            mid_ang = (scan_angle + ang) / 2
            mid_r   = max(r_chord[30], 0.3)
            ax.text(mid_ang, mid_r, '●', ha='center', va='center',
                    fontsize=5, color=color, alpha=0.9)

    ax.set_ylim(0, R_label + 0.1)
    ax.set_rticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('white')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=pos_cmap(0.8), lw=2, label='Positive β (ELA ↑ FC)'),
        Line2D([0], [0], color=neg_cmap(0.2), lw=2, label='Negative β (ELA ↓ FC)'),
        Line2D([0], [0], color='gray', lw=1, linestyle='-', alpha=0.4,
               label='n.s. (FDR q≥0.05)'),
        Line2D([0], [0], color='black', lw=2, label='FDR q<0.05 (SCAN subset)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(-0.1, -0.05),
              fontsize=5.5, frameon=False)

    ax.set_title('SCAN network connectivity\n(effect of early-life adversity)',
                 fontsize=8, pad=12)

    fig.tight_layout()
    save(fig, 'fc_chord_scan')


# ── Figure 4: surface dscalar ─────────────────────────────────────────────────

def fig_surface_dscalar(df_res):
    log('Figure 4: surface dscalar for Workbench')
    CIFTI_DIR.mkdir(parents=True, exist_ok=True)

    out_path = CIFTI_DIR / 'fc_scan_ela_beta_surface.dscalar.nii'

    # Build (N_FULL,) array: each vertex gets the beta for its network's SCAN FC
    scan_betas = {}
    for _, row in df_res.iterrows():
        n1, n2 = row['net1'], row['net2']
        if n1 == 'SCAN':
            scan_betas[n2] = row['beta']
        elif n2 == 'SCAN':
            scan_betas[n1] = row['beta']

    # Load atlas to get BrainModelAxis and vertex labels
    atlas_img = nib.load(str(ATLAS_PATH))
    bm_ax     = atlas_img.header.get_axis(1)
    atlas_data = atlas_img.get_fdata()[0].astype(np.int16)

    # Map label values to network names
    label_to_net = {v: k for k, v in NET_LABEL.items()}

    surface_data = np.zeros(N_FULL, dtype=np.float32)
    for v in range(N_CORT):
        lbl = atlas_data[v]
        net = label_to_net.get(int(lbl), None)
        if net is not None and net in scan_betas:
            surface_data[v] = scan_betas[net]

    scalar_ax = nib.cifti2.ScalarAxis(['ELA_SCAN_FC_beta'])
    header    = nib.Cifti2Header.from_axes((scalar_ax, bm_ax))
    img       = nib.Cifti2Image(surface_data[np.newaxis, :].astype(np.float32), header)
    nib.save(img, str(out_path))
    log(f'  → {out_path.name}')

    # Static figure: bar chart of betas per network (proxy for surface view)
    nets_ordered = DISPLAY_ORDER
    betas = [scan_betas.get(n, np.nan) for n in nets_ordered if n != 'SCAN']
    nets_for_plot = [n for n in nets_ordered if n != 'SCAN']

    sig_scan_dict = {}
    for _, row in df_res.iterrows():
        n1, n2 = row['net1'], row['net2']
        if n1 == 'SCAN':
            sig_scan_dict[n2] = bool(row['sig_scan'])
        elif n2 == 'SCAN':
            sig_scan_dict[n1] = bool(row['sig_scan'])

    fig, ax = plt.subplots(figsize=(COL2, COL1 * 0.9))
    colors = [NET_COLOR.get(n, '#888888') for n in nets_for_plot]
    colors = ['#555555' if n == 'SAL' else c for n, c in zip(nets_for_plot, colors)]
    bars = ax.bar(range(len(nets_for_plot)), betas, color=colors, width=0.7,
                  edgecolor='white', linewidth=0.4)

    # Significance markers
    ymax = np.nanmax(np.abs(betas)) * 1.15
    for i, (b, net) in enumerate(zip(betas, nets_for_plot)):
        if sig_scan_dict.get(net, False) and not np.isnan(b):
            ax.text(i, b + (0.005 if b >= 0 else -0.005),
                    '●', ha='center', va='bottom' if b >= 0 else 'top',
                    fontsize=6, color='black')

    ax.axhline(0, color='black', lw=0.5, ls='-')
    ax.set_xticks(range(len(nets_for_plot)))
    ax.set_xticklabels(nets_for_plot, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('β  (threat composite → FC with SCAN)', fontsize=7)
    ax.set_title('SCAN connectivity with each network — ELA effect size', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save(fig, 'fc_scan_betas_barplot')

    log(f'\nWorkbench notes:')
    log(f'  Load {out_path.name} as a dscalar overlay.')
    log(f'  Palette: blue-red diverging, centered at 0.')
    log(f'  Overlay SCAN_border_atlas_L/R.border for reference.')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    log('=' * 60)
    log('FC FIGURES')
    log('=' * 60)

    if not RES_PATH.exists():
        log(f'ERROR: {RES_PATH} not found. Run fc_analysis.py first.')
        sys.exit(1)
    if not FC_PATH.exists():
        log(f'ERROR: {FC_PATH} not found. Run fc_analysis.py first.')
        sys.exit(1)

    df_res  = pd.read_csv(RES_PATH)
    df_fc   = pd.read_csv(FC_PATH)
    df_base = pd.read_csv(DAT_DIR / 'df_base.csv')
    df_cov  = df_base[['sub_ID', 'threat_composite']].copy()

    # Add site column to df_cov if needed for group labeling
    if 'study_site' in df_base.columns:
        df_cov['study_site'] = df_base['study_site']

    log(f'\nResults: {len(df_res)} pairs')
    log(f'FC data: N={len(df_fc)}')

    fig_beta_matrix(df_res)
    fig_group_means(df_fc, df_cov)
    fig_chord_scan(df_res)
    fig_surface_dscalar(df_res)

    log('\nAll figures saved to: ' + str(FIG_DIR))
    log('Done.')


if __name__ == '__main__':
    main()
