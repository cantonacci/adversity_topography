"""
Phase 0: A priori ELA composite construction.

Three theory-driven composites (McLaughlin & Sheridan threat/deprivation/
unpredictability framework):

  Threat          (4): physical_trauma, family_aggression,
                        family_conflict_youth, family_anger*
  Deprivation     (4): ses_neighborhood, primary_caregiver_support*,
                        secondary_caregiver_support*, caregiver_supervision
  Unpredictability(2): caregiver_psych, caregiver_substance_sep

  * negated before compositing so that high = more adversity.

Steps:
  1. Negate ELA_REVERSE_CODED vars (family_anger, primary/secondary CG support)
  2. Z-score all 10 factors across the full baseline sample
  3. Compute each composite as the mean of its constituent z-scores
  4. Z-score the composite
  5. Save ela_composites.csv  (sub_ID + 3 composites + 10 z-scored factor cols)
  6. Save intercorrelation figure and domain-assignment figure
"""
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import (
    BASE_DIR, DAT_DIR, FIG_DIR, TAB_DIR,
    ELA_COLS, ELA_LABELS_SHORT, ELA_REVERSE_CODED,
    ELA_THREAT_COLS, ELA_DEPRIVATION_COLS, ELA_UNPRED_COLS,
    COMPOSITE_COLS, COMPOSITE_LABELS,
)

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'axes.linewidth':     0.8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

GREY = '#4a4a4a'
BLUE = '#0072B2'
RED  = '#D55E00'
GRN  = '#009E73'

DOMAIN_COLORS = {
    'Threat':           '#C0392B',
    'Deprivation':      '#0072B2',
    'Unpredictability': '#8E44AD',
}

DOMAIN_MAP = (
    [(c, 'Threat')           for c in ELA_THREAT_COLS] +
    [(c, 'Deprivation')      for c in ELA_DEPRIVATION_COLS] +
    [(c, 'Unpredictability') for c in ELA_UNPRED_COLS]
)
FACTOR_DOMAIN = dict(DOMAIN_MAP)

# Display labels (post-negation direction)
FACTOR_LABELS = {
    'ELA_caregiver_psych':              'CG Psychopathology',
    'ELA_ses_neighborhood':             'SES/Neighborhood',
    'ELA_primary_caregiver_support':    'Low Primary CG Support',
    'ELA_secondary_caregiver_support':  'Low Secondary CG Support',
    'ELA_family_conflict_youth':        'Family Conflict',
    'ELA_caregiver_substance_sep':      'CG Substance/Sep',
    'ELA_family_anger':                 'Family Anger (rev)',
    'ELA_family_aggression':            'Family Aggression',
    'ELA_physical_trauma':              'Physical Trauma',
    'ELA_caregiver_supervision':        'Poor CG Supervision',
}


def log(msg=''):
    print(msg, flush=True)


def main():
    log('=' * 70)
    log('PHASE 0 — A Priori ELA Composite Construction')
    log('=' * 70)

    # ── 1. Load baseline data ─────────────────────────────────────────────────
    df = pd.read_csv(DAT_DIR / 'df_base.csv')
    ela_raw = df[['sub_ID'] + ELA_COLS].copy()
    N_before = len(ela_raw)
    ela_raw = ela_raw.dropna(subset=ELA_COLS)
    N = len(ela_raw)
    log(f'\nN subjects with complete ELA: {N}  (dropped {N_before - N} with any NaN)')

    # ── 2. Negate reverse-coded factors (high = more adversity after negation) ─
    ela_rec = ela_raw.copy()
    for col in ELA_REVERSE_CODED:
        ela_rec[col] = -ela_rec[col]
    log(f'Negated: {ELA_REVERSE_CODED}')

    # ── 3. Z-score all 10 factors ─────────────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(ela_rec[ELA_COLS].values)
    X_df = pd.DataFrame(X, columns=ELA_COLS, index=ela_rec.index)

    # ── 4. Build composites ───────────────────────────────────────────────────
    def make_composite(factor_cols):
        raw = X_df[factor_cols].mean(axis=1)
        return (raw - raw.mean()) / raw.std()

    threat_z  = make_composite(ELA_THREAT_COLS)
    depriv_z  = make_composite(ELA_DEPRIVATION_COLS)
    unpred_z  = make_composite(ELA_UNPRED_COLS)

    log('\nComposite descriptive stats (should be mean≈0, std≈1):')
    for name, s in [('threat_composite', threat_z),
                    ('deprivation_composite', depriv_z),
                    ('unpredictability_composite', unpred_z)]:
        log(f'  {name}: mean={s.mean():.4f}  std={s.std():.4f}  '
            f'range=[{s.min():.2f}, {s.max():.2f}]')

    # ── 5. Composite inter-correlations ──────────────────────────────────────
    comp_df = pd.DataFrame({
        'threat_composite':           threat_z.values,
        'deprivation_composite':      depriv_z.values,
        'unpredictability_composite': unpred_z.values,
    }, index=ela_rec.index)

    log('\nComposite inter-correlations:')
    log(comp_df.corr().round(3).to_string())

    # ── 6. Save output CSV ────────────────────────────────────────────────────
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    out = ela_raw[['sub_ID']].copy()
    out = out.join(comp_df)
    # Also save z-scored individual factors (useful for brain figures)
    X_df_named = X_df.add_suffix('_z')
    out = out.join(X_df_named)

    out_path = TAB_DIR / 'ela_composites.csv'
    out.to_csv(out_path, index=False)
    log(f'\nSaved: {out_path}  ({len(out)} rows, {len(out.columns)} cols)')

    # ── 7. Figures ────────────────────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_intercorr(X_df, ELA_COLS)
    _plot_domain_assignment(X_df, comp_df)

    log('\nPhase 0 complete.')


def _plot_intercorr(X_df, ela_cols):
    corr = X_df.corr()
    labels = [FACTOR_LABELS.get(c, c) for c in ela_cols]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='equal')

    n = len(ela_cols)
    ax.set_xticks(range(n));  ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            color = 'white' if abs(v) > 0.5 else GREY
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6.5, color=color,
                    fontweight='bold' if abs(v) > 0.5 else 'normal')

    # Domain color bands on axes
    domain_order = (ELA_THREAT_COLS + ELA_DEPRIVATION_COLS + ELA_UNPRED_COLS)
    ax_order = {c: i for i, c in enumerate(ela_cols)}
    for col, dom in FACTOR_DOMAIN.items():
        idx = ax_order.get(col)
        if idx is None:
            continue
        dc = DOMAIN_COLORS[dom]
        for spine_ax in [ax]:
            spine_ax.add_patch(plt.Rectangle(
                (idx - 0.5, -0.5), 1, n, fill=False,
                edgecolor=dc, lw=2.5, zorder=5, linestyle='-',
                clip_on=False))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson r', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title('ELA Factor Inter-correlations\n(z-scored, adversity direction)',
                 fontsize=11, pad=12)

    for i in range(n + 1):
        ax.axhline(i - 0.5, color='white', lw=0.4)
        ax.axvline(i - 0.5, color='white', lw=0.4)

    plt.tight_layout()
    path = FIG_DIR / 'fig_ela_composite_intercorr.png'
    fig.savefig(path)
    plt.close(fig)
    log(f'  Saved: {path.name}')


def _plot_domain_assignment(X_df, comp_df):
    """
    Two-panel figure:
      Left:  domain assignment (factor × domain matrix, colored by membership)
      Right: composite score distributions
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ── Panel A: domain membership matrix ────────────────────────────────────
    ax = axes[0]
    domains  = ['Threat', 'Deprivation', 'Unpredictability']
    factors  = ELA_THREAT_COLS + ELA_DEPRIVATION_COLS + ELA_UNPRED_COLS
    mat = np.zeros((len(factors), len(domains)))
    for i, f in enumerate(factors):
        dom = FACTOR_DOMAIN.get(f)
        if dom:
            j = domains.index(dom)
            mat[i, j] = 1

    cmap_mem = mcolors.ListedColormap(['#EEEEEE', '#444444'])
    ax.imshow(mat, cmap=cmap_mem, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, fontsize=9, fontweight='bold')
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels([FACTOR_LABELS.get(f, f) for f in factors], fontsize=8.5)

    for i, f in enumerate(factors):
        for j, d in enumerate(domains):
            if mat[i, j]:
                ax.text(j, i, '●', ha='center', va='center',
                        fontsize=14, color=DOMAIN_COLORS[d])

    for x in range(len(domains)):
        ax.axvline(x - 0.5, color='white', lw=1)
    for y in range(len(factors)):
        ax.axhline(y - 0.5, color='white', lw=0.5)

    ax.set_title('ELA Domain Assignment\n(a priori, McLaughlin & Sheridan framework)',
                 fontsize=10, pad=8)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # ── Panel B: composite score distributions ────────────────────────────────
    ax2 = axes[1]
    for i, (col, label) in enumerate(COMPOSITE_LABELS.items()):
        vals = comp_df[col].dropna().values
        color = list(DOMAIN_COLORS.values())[i]
        ax2.hist(vals, bins=60, alpha=0.55, color=color, label=label,
                 density=True, edgecolor='none')

    ax2.set_xlabel('Composite Score (z)', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title('Distribution of ELA Composite Scores\n(baseline analytic sample)',
                  fontsize=10, pad=8)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = FIG_DIR / 'fig_ela_composites.png'
    fig.savefig(path)
    plt.close(fig)
    log(f'  Saved: {path.name}')


if __name__ == '__main__':
    main()
