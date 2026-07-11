"""
Encroachment analysis: statistics and figures.

Requires: outputs/encroachment/encroachment_{tp}.csv (from compute_encroachment.py)

Conceptual framing: the group template defines 'typical' topography, so by construction
low-ELA kids show little encroachment. The meaningful question is WHERE does SCAN
encroach in high-ELA kids — i.e., which networks does it displace? (Lynch et al. 2024)

Analyses:
  1. High-ELA encroachment bar chart (Lynch Fig 2D style) — high-ELA group only,
     sorted by encroachment fraction. One figure per timepoint × split combination.
  2. Zone bar chart (Lynch Fig 2E style) — medial vs. lateral zone of encroachment
     in high-ELA group at baseline, showing that displaced-network profiles differ.
  3. MixedLM regression — continuous threat_composite predictor, for significance stars.
  4. Longitudinal line plot — top networks in high-ELA group across all 4 timepoints.
  5. Whole-sample descriptive bar chart (supplementary) — for reference.

Outputs:
  outputs/tables/encroachment_regression_{tp}.csv
  outputs/figures/encroachment/encroachment_high_ela_{split}_{tp}.{png,pdf}
  outputs/figures/encroachment/encroachment_zone_{split}_baseline.{png,pdf}
  outputs/figures/encroachment/encroachment_longitudinal.{png,pdf}
  outputs/figures/encroachment/encroachment_descriptive_{tp}.{png,pdf}  (supplementary)
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import sem, ttest_rel
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from pathlib import Path

from adtopo.config import cfg
from adtopo.re_models import fit_ols_cluster_table
from adtopo.logging_utils import get_logger
_log = get_logger('encroachment_analysis')

# ── Constants ─────────────────────────────────────────────────────────────────
ENC_DIR = cfg.OUT_DIR / 'encroachment'          # repo outputs/encroachment (was code/outputs/...)
OUT_FIG = cfg.FIG_DIR / 'encroachment'
OUT_TAB = cfg.TAB_DIR

TARGET_NETS = [n for n in cfg.NETWORKS if n != 'SCAN']

ATLAS_NET_COLORS = {
    'DMN':   '#FF0000', 'VIS':   '#000099', 'FP':    '#CCCC00',
    'DAN':   '#00AA00', 'VAN':   '#0D85A0', 'SAL':   '#333333',
    'CO':    '#6600CC', 'SMD':   '#44CCCC', 'SML':   '#FF8000',
    'AUD':   '#B266FF', 'Tpole': '#006699', 'MTL':   '#55CC55',
    'PMN':   '#3C3CFB', 'PON':   '#999999', 'SCAN':  '#8E0067',
}

TP_FILES = {
    'baseline': 'encroachment_baseline.csv',
    'year2':    'encroachment_year2.csv',
    'year4':    'encroachment_year4.csv',
    'year6':    'encroachment_year6.csv',
}

DF_FILES = {
    'baseline': 'df_base.csv',
    'year2':    'df_y2.csv',
    'year4':    'df_y4.csv',
    'year6':    'df_y6.csv',
}

TP_COV_KEY = {
    'baseline': '00A',
    'year2':    '02A',
    'year4':    '04A',
    'year6':    '06A',
}

mpl.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':        8,
    'axes.linewidth':   0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
})


def log(msg=''):
    _log.info(str(msg))


def load_merged(tp):
    """Load encroachment CSV merged with ELA/covariates for a timepoint."""
    enc_path = ENC_DIR / TP_FILES[tp]
    df_path  = cfg.DAT_DIR  / DF_FILES[tp]
    if not enc_path.exists():
        log(f'  WARNING: {enc_path.name} not found'); return None
    if not df_path.exists():
        log(f'  WARNING: {df_path.name} not found'); return None

    enc = pd.read_csv(enc_path)
    df  = pd.read_csv(df_path)

    # Columns needed from df
    cov_key  = TP_COV_KEY[tp]
    cov_info = cfg.TIMEPOINT_COV[cov_key]
    fd_col   = cov_info['fd']
    site_col = cov_info['site']
    age_col  = cov_info['age']

    # Config column names may not match the per-timepoint dataframes; fall back to
    # the consistent names that phase1 uses across all timepoints.
    if fd_col not in df.columns:
        fd_col = next((c for c in ['fd', 'rest_mean_FD'] if c in df.columns), None)
    if site_col not in df.columns:
        site_col = next((c for c in ['study_site', 'site'] if c in df.columns), None)
    if age_col not in df.columns:
        age_col = next((c for c in ['interview_age'] if c in df.columns), None)
    if any(v is None for v in [fd_col, site_col, age_col]):
        log(f'  WARNING: could not find FD/site/age column in {df_path.name} — skipping')
        return None

    keep = ['sub_ID', 'family_id', 'sex_num',
            'threat_composite', 'deprivation_composite', 'unpredictability_composite',
            fd_col, site_col, age_col]
    keep = [c for c in keep if c in df.columns]
    merged = enc.merge(df[keep], on='sub_ID', how='inner')

    # Standardise age to years if needed
    if cov_info.get('age_in_months'):
        merged[age_col] = merged[age_col] / 12.0

    merged['fd']   = merged[fd_col]
    merged['site'] = merged[site_col]
    merged['age']  = merged[age_col]
    merged = merged.dropna(subset=['threat_composite', 'fd', 'site', 'age', 'sex_num'])
    return merged


def frac_cols():
    return [f'encroach_frac_{n}' for n in TARGET_NETS]


# ── Shared plot helper ────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.yaxis.grid(True, linewidth=0.3, color='#dddddd', zorder=0)
    ax.set_axisbelow(True)


def _bar_chart(ax, nets, means, errs, title, ylabel=True):
    """Draw a clean Lynch-style bar chart on ax."""
    colors = [ATLAS_NET_COLORS[n] for n in nets]
    x = np.arange(len(nets))
    ax.bar(x, means, color=colors, edgecolor='none', width=0.65,
           yerr=errs, error_kw={'elinewidth': 0.8, 'capsize': 2.5,
                                'ecolor': '#333333', 'zorder': 4},
           zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(nets, rotation=40, ha='right', fontsize=7.5)
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=4)
    ax.set_xlim(-0.6, len(nets) - 0.4)
    if ylabel:
        ax.set_ylabel('% of network territory displaced', fontsize=8)
    _style_ax(ax)


# ── Analysis 1: Descriptive bar chart (whole sample, supplementary) ───────────

def plot_descriptive(merged, tp='baseline'):
    log(f'\n[Descriptive] {tp}  N={len(merged)}')
    fc    = frac_cols()
    means = merged[fc].mean() * 100
    sems_ = merged[fc].apply(sem) * 100
    nets  = [c.replace('encroach_frac_', '') for c in fc]

    order   = means.argsort()[::-1].values
    nets_s  = [nets[i]        for i in order]
    means_s = np.array([means.iloc[i]  for i in order])
    sems_s  = np.array([sems_.iloc[i]  for i in order])

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    _bar_chart(ax, nets_s, means_s, sems_s,
               title=f'SCAN Encroachment — All Subjects, {tp.capitalize()} (N={len(merged)})')
    fig.tight_layout()

    for fmt in ('png', 'pdf'):
        p = OUT_FIG / f'encroachment_descriptive_{tp}.{fmt}'
        fig.savefig(str(p), dpi=300, bbox_inches='tight')
        log(f'  → {p.name}')
    plt.close(fig)


# ── Analysis 2: High-ELA encroachment bar chart (Lynch Fig 2D style) ──────────

def get_high_group(merged, split='1sd'):
    """Return (high_df, N_label)."""
    tc = merged['threat_composite']
    if split == '1sd':
        return merged[tc >= 1.0], '±1 SD'
    else:
        p90 = float(np.percentile(tc, 90))
        return merged[tc >= p90], 'Top 10%'


def run_regression(merged, tp='baseline'):
    """Canonical reported spec (lib.re_models.fit_ols_cluster_table):
    encroach_frac_NET ~ threat_composite + age + sex + fd + C(site),
    OLS with family-cluster-robust SEs. The 'z' column holds the cluster-robust z."""
    log(f'\n[Regression] {tp}  N={len(merged)}')
    rows = []
    for net in TARGET_NETS:
        outcome = f'encroach_frac_{net}'
        if outcome not in merged.columns:
            continue
        tbl, meta = fit_ols_cluster_table(
            merged, outcome, ['threat_composite'], ['age', 'sex_num', 'fd'],
            site_col='site', family_col='family_id')
        if not meta['converged'] or tbl.empty:
            continue
        r = tbl.iloc[0]
        # Standardized beta (threat is z-scored, so beta_std = beta / SD(outcome));
        # this is the per-SD-of-encroachment scale reported in the manuscript.
        y_sd = merged[outcome].dropna().std()
        beta_std = float(r['beta']) / y_sd if y_sd and y_sd > 0 else np.nan
        rows.append({'network': net, 'timepoint': tp, 'N': meta['n'],
                     'beta': float(r['beta']), 'beta_std': beta_std, 'se': float(r['se']),
                     'z': float(r['z']), 'p': float(r['p'])})

    if not rows:
        return pd.DataFrame()
    results = pd.DataFrame(rows)
    valid_p = results['p'].dropna()
    if len(valid_p) > 1:
        _, q, _, _ = multipletests(valid_p, method='fdr_bh')
        results.loc[results['p'].notna(), 'q_FDR'] = q
    else:
        results['q_FDR'] = np.nan

    n_sig = (results['q_FDR'] < 0.05).sum()
    log(f'  FDR q<0.05: {n_sig}/{len(results)} networks')
    log(results[['network','beta','se','p','q_FDR']].round(5).to_string(index=False))

    out = OUT_TAB / f'encroachment_regression_{tp}.csv'
    results.to_csv(out, index=False)
    log(f'  Saved: {out.name}')
    return results


def plot_high_ela_bar(merged, split='1sd', tp='baseline', reg_results=None):
    """Lynch Fig 2D style: high-ELA group only, sorted by encroachment fraction."""
    high, split_label = get_high_group(merged, split)
    log(f'\n[High-ELA bar] {split_label} | {tp}  N={len(high)}')

    fc    = frac_cols()
    nets  = [c.replace('encroach_frac_', '') for c in fc]
    means = high[fc].mean() * 100
    sems_ = high[fc].apply(sem) * 100

    order   = means.argsort()[::-1].values
    nets_s  = [nets[i]        for i in order]
    means_s = np.array([means.iloc[i]  for i in order])
    sems_s  = np.array([sems_.iloc[i]  for i in order])

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    _bar_chart(ax, nets_s, means_s, sems_s,
               title=f'SCAN Encroachment — High-Adversity Youth '
                     f'({split_label}, N={len(high)}), {tp.capitalize()}')
    fig.tight_layout()

    split_tag = '1sd' if split == '1sd' else 'p10p90'
    for fmt in ('png', 'pdf'):
        p = OUT_FIG / f'encroachment_high_ela_{split_tag}_{tp}.{fmt}'
        fig.savefig(str(p), dpi=300, bbox_inches='tight')
        log(f'  → {p.name}')
    plt.close(fig)

    tbl = pd.DataFrame({'network': nets_s, 'mean_pct': means_s, 'sem_pct': sems_s})
    out = OUT_TAB / f'encroachment_high_ela_{split_tag}_{tp}.csv'
    tbl.to_csv(out, index=False)
    log(f'  Saved: {out.name}')


# ── Analysis 3: Zone bar chart (Lynch Fig 2E style) ───────────────────────────

def _zone_pairwise_stats(high, nets_sorted, zone_suffix, n_compare=5):
    """
    Paired t-tests between adjacent ranked networks within a zone.
    Returns list of (x_left, x_right, q) for significant pairs only.
    """
    top = nets_sorted[:n_compare]
    ps, pairs = [], []
    for i in range(len(top) - 1):
        ca = f'encroach_frac_{top[i]}_{zone_suffix}'
        cb = f'encroach_frac_{top[i+1]}_{zone_suffix}'
        df_pair = high[[ca, cb]].dropna()
        if len(df_pair) < 10:
            ps.append(1.0)
        else:
            _, p = ttest_rel(df_pair[ca], df_pair[cb])
            ps.append(p)
        pairs.append((i, i + 1))
    if not ps:
        return []
    _, qs, _, _ = multipletests(ps, method='fdr_bh')
    return [(i, j, q) for (i, j), q in zip(pairs, qs) if q < 0.05]


def _draw_bracket(ax, x1, x2, y_top, q):
    """Draw a significance bracket between two bar positions."""
    star = '***' if q < 0.001 else ('**' if q < 0.01 else '*')
    h = y_top * 0.04
    ax.plot([x1, x1, x2, x2], [y_top, y_top + h, y_top + h, y_top],
            lw=0.7, color='#333333', clip_on=False)
    ax.text((x1 + x2) / 2, y_top + h * 1.1, star,
            ha='center', va='bottom', fontsize=7.5, color='#333333')


def plot_zone_bar(merged, split='1sd', tp='baseline', reg_results=None):
    """Medial vs. lateral zone encroachment in high-ELA group (Lynch Fig 2E style)."""
    high, split_label = get_high_group(merged, split)
    log(f'\n[Zone bar] {split_label} | {tp}  N={len(high)}')

    med_cols = [f'encroach_frac_{n}_medial'  for n in TARGET_NETS]
    lat_cols = [f'encroach_frac_{n}_lateral' for n in TARGET_NETS]
    if any(c not in merged.columns for c in med_cols[:1] + lat_cols[:1]):
        log('  WARNING: zone columns missing — run compute_encroachment.py first')
        return

    # Sort each zone independently by its own mean (largest first, NaN to end)
    def _sorted_zone(suffix):
        cols  = [f'encroach_frac_{n}_{suffix}' for n in TARGET_NETS]
        means = high[cols].mean() * 100
        errs  = high[cols].apply(sem) * 100
        sorted_means = means.sort_values(ascending=False, na_position='last')
        nets = [c.replace(f'encroach_frac_', '').replace(f'_{suffix}', '')
                for c in sorted_means.index]
        ms   = sorted_means.fillna(0).values
        es   = np.array([errs[f'encroach_frac_{n}_{suffix}'] for n in nets])
        es   = np.where(np.isnan(es), 0, es)
        return nets, ms, es

    nets_med, med_m, med_e = _sorted_zone('medial')
    nets_lat, lat_m, lat_e = _sorted_zone('lateral')

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5), sharey=False)
    specs = [
        ('Medial zone  (|x| < 20 mm)', nets_med, med_m, med_e, axes[0], 'medial'),
        ('Lateral zone  (|x| ≥ 20 mm)', nets_lat, lat_m, lat_e, axes[1], 'lateral'),
    ]

    for title, nets, ms, es, ax, suffix in specs:
        _bar_chart(ax, nets, ms, es, title=title, ylabel=(ax is axes[0]))

        # Pairwise adjacent t-tests on top 5 networks
        sig_pairs = _zone_pairwise_stats(high, nets, suffix, n_compare=5)
        for x1, x2, q in sig_pairs:
            y_top = max(ms[x1] + es[x1], ms[x2] + es[x2])
            _draw_bracket(ax, x1, x2, y_top * 1.08, q)

        log(f'  {suffix}: top network = {nets[0]} ({ms[0]:.2f}%), '
            f'{len(sig_pairs)} significant adjacent pairs')

    axes[0].set_ylabel('% of zone network territory displaced', fontsize=8)
    fig.suptitle(f'SCAN Encroachment by Cortical Zone — High-Adversity Youth '
                 f'({split_label}, N={len(high)}), {tp.capitalize()}',
                 fontsize=8.5, fontweight='bold', y=1.02)
    fig.tight_layout()

    split_tag = '1sd' if split == '1sd' else 'p10p90'
    for fmt in ('png', 'pdf'):
        p = OUT_FIG / f'encroachment_zone_{split_tag}_{tp}.{fmt}'
        fig.savefig(str(p), dpi=300, bbox_inches='tight')
        log(f'  → {p.name}')
    plt.close(fig)

    tbl = pd.DataFrame({
        'network_medial':   nets_med, 'medial_mean_pct': med_m, 'medial_sem_pct': med_e,
        'network_lateral':  nets_lat, 'lateral_mean_pct': lat_m, 'lateral_sem_pct': lat_e,
    })
    out = OUT_TAB / f'encroachment_zone_{split_tag}_{tp}.csv'
    tbl.to_csv(out, index=False)
    log(f'  Saved: {out.name}')


def _lighten(hex_color, amount=0.5):
    """Lighten a hex color by blending toward white."""
    c = mpl.colors.to_rgb(hex_color)
    return mpl.colors.to_hex(tuple(v + (1 - v) * amount for v in c))


# ── Analysis 4: Longitudinal line plot (high-ELA group, top networks) ─────────

def plot_longitudinal(top_nets=None):
    """Line plot of encroachment in high-ELA group across timepoints (top 4 networks)."""
    log('\n[Longitudinal] loading all timepoints...')
    tps       = ['baseline', 'year2', 'year4', 'year6']
    tp_labels = ['Baseline\n(~9y)', 'Year 2\n(~11y)', 'Year 4\n(~13y)', 'Year 6\n(~15y)']

    # Determine top nets from baseline high-ELA group (1SD split)
    if top_nets is None:
        base_merged = load_merged('baseline')
        if base_merged is None:
            log('  baseline data not found — skipping'); return
        high_base, _ = get_high_group(base_merged, '1sd')
        fc    = [f'encroach_frac_{n}' for n in TARGET_NETS]
        means = high_base[fc].mean()
        top_nets = [c.replace('encroach_frac_', '') for c in means.nlargest(4).index]
        log(f'  Top 4 networks (high-ELA baseline): {top_nets}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    splits = [('1sd', '±1 SD', axes[0]), ('p10p90', 'Top 10%', axes[1])]

    for split, split_label, ax in splits:
        valid_labels = []
        for net in top_nets:
            col        = f'encroach_frac_{net}'
            high_vals  = []
            high_errs  = []
            valid_tps  = []
            vlabels    = []

            for tp, tpl in zip(tps, tp_labels):
                merged = load_merged(tp)
                if merged is None or col not in merged.columns:
                    continue
                high, _ = get_high_group(merged, split)
                vals = high[col].dropna() * 100
                high_vals.append(vals.mean())
                high_errs.append(sem(vals))
                valid_tps.append(tp)
                vlabels.append(tpl)

            if not high_vals:
                continue
            valid_labels = vlabels
            x = np.arange(len(valid_tps))
            c = ATLAS_NET_COLORS[net]
            ax.errorbar(x, high_vals, yerr=high_errs, color=c,
                        marker='o', linewidth=1.8, markersize=5,
                        capsize=3, elinewidth=0.8, label=net, zorder=3)

        ax.set_xticks(np.arange(len(valid_labels)))
        ax.set_xticklabels(valid_labels, fontsize=8)
        ax.set_title(split_label, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('SCAN encroachment (% of network territory)', fontsize=8.5)
    axes[0].legend(title='Displaced\nnetwork', fontsize=7.5, title_fontsize=7.5,
                   frameon=False, loc='upper right')
    fig.suptitle('SCAN Encroachment Across Development — High-Adversity Youth\n'
                 '(top 4 displaced networks at baseline; mean ± SEM)',
                 fontsize=9.5, fontweight='bold', y=1.01)
    fig.tight_layout()

    for fmt in ('png', 'pdf'):
        p = OUT_FIG / f'encroachment_longitudinal.{fmt}'
        fig.savefig(str(p), dpi=300, bbox_inches='tight')
        log(f'  → {p.name}')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    log('=' * 60)
    log('ENCROACHMENT ANALYSIS')
    log('=' * 60)

    for tp in ('baseline', 'year2', 'year4', 'year6'):
        log(f'\n{"="*50}')
        log(f'Timepoint: {tp.upper()}')
        log(f'{"="*50}')
        merged = load_merged(tp)
        if merged is None:
            continue

        # Regression (continuous threat predictor — FDR-corrected)
        reg = run_regression(merged, tp=tp)

        for split in ('1sd', 'p10p90'):
            # Primary: high-ELA group only (Lynch Fig 2D style)
            plot_high_ela_bar(merged, split=split, tp=tp)

            # Zone breakdown at baseline (Lynch Fig 2E style)
            if tp == 'baseline':
                plot_zone_bar(merged, split=split, tp=tp)

        # Supplementary: whole-sample descriptive
        plot_descriptive(merged, tp=tp)

    # Longitudinal: top displaced networks in high-ELA group across ages
    plot_longitudinal()

    log('\nDone.')


if __name__ == '__main__':
    main()
