"""
Phase 1: Data preparation and quality checks
ELA and Functional Network Topography in ABCD — all four timepoints.

Age is stored in years in all output DataFrames (ELA interview_age is divided
by 12; yr-4/6 ages from the extended covariates file are already in years).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from adtopo.config import cfg
from adtopo.logging_utils import get_logger
_log = get_logger('build_analysis_dataframes')

np.random.seed(cfg.RANDOM_SEED)

log_lines = []
def log(msg=''):
    _log.info(str(msg))
    log_lines.append(str(msg))

def add_proportions(df, networks, label):
    df = df.copy()
    df['total_area'] = df[networks].sum(axis=1)
    for net in networks:
        df[f'prop_{net}'] = df[net] / df['total_area']
    prop_sums = df[[f'prop_{n}' for n in networks]].sum(axis=1)
    devs = (prop_sums - 1.0).abs()
    log(f'  [{label}] proportion sum: mean={prop_sums.mean():.8f}, max_dev={devs.max():.2e}')
    log(f'  [{label}] total area: mean={df["total_area"].mean():.0f} mm², SD={df["total_area"].std():.0f}')
    return df

def build_analysis_sample(df_topo, df_ela, df_cov, session, label):
    """
    Merge topo + ELA + covariates for a given session, returning a clean DataFrame.

    Age is always output in years under the column 'interview_age'.
    ELA's interview_age (months) is divided by 12. Yr-4/6 ages (already years) are
    renamed from their session-specific column name.
    """
    tcov = cfg.TIMEPOINT_COV[session]
    fd_col, site_col = tcov['fd'], tcov['site']
    age_col, age_in_months = tcov['age'], tcov['age_in_months']

    df = df_topo.merge(df_ela, on='sub_ID', how='inner', validate='one_to_one')
    # sex comes from df_ela; drop from df_cov if present to avoid _x/_y conflict
    df_cov_merge = df_cov.drop(columns=['sex'], errors='ignore')
    df = df.merge(df_cov_merge, on='sub_ID', how='inner', validate='one_to_one')
    log(f'  [{label}] After merge: N={len(df)}')

    before = len(df)
    df = df.dropna(subset=cfg.ELA_COLS)
    log(f'  [{label}] After dropping ELA NAs: N={len(df)} (dropped {before - len(df)})')

    cov_needed = [age_col, 'sex', fd_col, site_col, 'family_id']
    before = len(df)
    df = df.dropna(subset=cov_needed)
    log(f'  [{label}] After dropping covariate NAs: N={len(df)} (dropped {before - len(df)})')

    # Normalise age to years
    if age_col != 'interview_age':
        # ELA already merged in a baseline 'interview_age' (months); replace with
        # the session-specific age from the covariates file.
        df = df.drop(columns=['interview_age'], errors='ignore')
        df = df.rename(columns={age_col: 'interview_age'})
    if age_in_months:
        df['interview_age'] = df['interview_age'] / 12.0

    df['sex_num'] = (df['sex'] == 'M').astype(float)
    # Rename session-specific FD/site cols to generic names so downstream
    # scripts can refer to them without knowing the timepoint.
    #
    # NOTE on site count: study_site contains 22 distinct labels at baseline, but
    # there are 21 actual ABCD data-collection sites (reported as such in the
    # manuscript). The 22nd label is an administrative site assigned to a handful
    # of participants with an irregular site situation (e.g. children who moved
    # between sites); at baseline it holds only ~15 subjects. It is retained as its
    # own fixed-effect level here so those participants are not misattributed to a
    # real site — it does not represent a 21st collection site.
    df = df.rename(columns={fd_col: 'fd', site_col: 'study_site'})
    return df.reset_index(drop=True)

def zscore_ela(df, ela_cols, label):
    df = df.copy()
    for c in ela_cols:
        df[f'{c}_raw'] = df[c]
    for c in ela_cols:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    log(f'  [{label}] Post-z-score check (should be mean≈0, SD≈1):')
    for c in ela_cols:
        log(f'    {c}: mean={df[c].mean():.4f}, SD={df[c].std():.4f}')
    return df


def main():
    for d in [cfg.FIG_DIR, cfg.TAB_DIR, cfg.DAT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1.1 — Load topography data ──────────────────────────────────────────
    log('=' * 70)
    log('STEP 1.1 — Load topography data')
    log('=' * 70)

    topo_raw = {}
    for session, path in cfg.TOPO_FILES.items():
        df = pd.read_csv(path)
        before = len(df)
        df = df.dropna(subset=cfg.NETWORKS).reset_index(drop=True)
        log(f'  {session}: {len(df)} valid subjects (dropped {before - len(df)} NaN rows)')
        topo_raw[session] = df

    df_topo = {
        '00A': add_proportions(topo_raw['00A'], cfg.NETWORKS, 'baseline-00A'),
        '02A': add_proportions(topo_raw['02A'], cfg.NETWORKS, 'year2-02A'),
        '04A': add_proportions(topo_raw['04A'], cfg.NETWORKS, 'year4-04A'),
        '06A': add_proportions(topo_raw['06A'], cfg.NETWORKS, 'year6-06A'),
    }

    # ── Step 1.2 — Load ELA and covariates ───────────────────────────────────────
    log()
    log('=' * 70)
    log('STEP 1.2 — Load ELA factors and covariates')
    log('=' * 70)

    df_ela = pd.read_excel(cfg.ELA_FILE)
    df_cov = pd.read_excel(cfg.COV_FILE, na_values=['NA', 'na', 'N/A', ''])

    for name, df in [('df_ela', df_ela), ('df_cov', df_cov)]:
        log(f'\n  {name}: shape={df.shape}')
        log(f'  Columns: {list(df.columns)}')
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing):
            log(f'  Missing values:\n{missing.to_string()}')

    # ── Step 1.3 — Family structure ───────────────────────────────────────────────
    log()
    log('=' * 70)
    log('STEP 1.3 — Family structure (siblings → random effect, not excluded)')
    log('=' * 70)

    families = df_cov.groupby('family_id')['sub_ID'].apply(list)
    multi_fam = families[families.apply(len) > 1]
    log(f'  Families with >1 subject: {len(multi_fam)}')
    log(f'  Subjects with a sibling in dataset: {sum(len(v) - 1 for v in multi_fam)}')
    log(f'  → Keeping ALL subjects; family_id used as random intercept grouping variable')

    # ── Step 1.4 — Merge samples ─────────────────────────────────────────────────
    log()
    log('=' * 70)
    log('STEP 1.4 — Merge topography + ELA + covariates')
    log('=' * 70)

    df_base = build_analysis_sample(df_topo['00A'], df_ela, df_cov, '00A', 'baseline')
    df_y2   = build_analysis_sample(df_topo['02A'], df_ela, df_cov, '02A', 'year-2')
    df_y4   = build_analysis_sample(df_topo['04A'], df_ela, df_cov, '04A', 'year-4')
    df_y6   = build_analysis_sample(df_topo['06A'], df_ela, df_cov, '06A', 'year-6')

    # Longitudinal sample: subjects present at baseline AND year-2
    long_ids = set(df_base['sub_ID']) & set(df_y2['sub_ID'])
    df_long_base = df_base[df_base['sub_ID'].isin(long_ids)].copy()
    df_long_y2   = df_y2[df_y2['sub_ID'].isin(long_ids)].copy()
    log(f'\n  Longitudinal sample (baseline + year-2): N={len(long_ids)}')

    # ── Step 1.5 — Z-score ELA factors ───────────────────────────────────────────
    log()
    log('=' * 70)
    log('STEP 1.5 — Z-score ELA factor scores')
    log('=' * 70)

    df_base      = zscore_ela(df_base,    cfg.ELA_COLS, 'baseline')
    df_y2        = zscore_ela(df_y2,      cfg.ELA_COLS, 'year-2')
    df_y4        = zscore_ela(df_y4,      cfg.ELA_COLS, 'year-4')
    df_y6        = zscore_ela(df_y6,      cfg.ELA_COLS, 'year-6')
    df_long_base = df_base[df_base['sub_ID'].isin(long_ids)].copy()
    df_long_y2   = df_y2[df_y2['sub_ID'].isin(long_ids)].copy()

    # ── Step 1.6 — Descriptive statistics ────────────────────────────────────────
    log()
    log('=' * 70)
    log('STEP 1.6 — Descriptive statistics')
    log('=' * 70)

    for label, df in [('BASELINE', df_base), ('YEAR-2', df_y2),
                      ('YEAR-4',   df_y4),   ('YEAR-6', df_y6)]:
        log(f'\n  --- {label} (N={len(df)}) ---')
        log(f'  Age (years): mean={df["interview_age"].mean():.2f}, SD={df["interview_age"].std():.2f}')
        sex_pct = df['sex'].value_counts(normalize=True).mul(100).round(1)
        log(f'  Sex: {sex_pct.to_dict()}')
        log(f'  N sites: {df["study_site"].nunique()}')
        log(f'  Mean FD: mean={df["fd"].mean():.4f}, SD={df["fd"].std():.4f}')
        log(f'  Total cortical area: mean={df["total_area"].mean():.0f} mm², SD={df["total_area"].std():.0f}')

    log()
    log('  Network proportion descriptives — BASELINE:')
    prop_cols = [f'prop_{n}' for n in cfg.NETWORKS]
    log(df_base[prop_cols].describe().T.round(5).to_string())

    log()
    log('  ELA factor inter-correlations — BASELINE:')
    ela_corr = df_base[cfg.ELA_COLS].corr()
    log(ela_corr.round(3).to_string())

    # ── Figures ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    short_labels = [c.replace('ELA_', '').replace('_', '\n') for c in cfg.ELA_COLS]
    sns.heatmap(ela_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax,
                xticklabels=short_labels, yticklabels=short_labels)
    ax.set_title('ELA Factor Inter-Correlations (Baseline Sample)', fontsize=13)
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / 'fig_phase1_ela_correlation_matrix.png', dpi=150)
    plt.close()

    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    for i, net in enumerate(cfg.NETWORKS):
        ax = axes.flatten()[i]
        ax.violinplot(df_base[f'prop_{net}'].dropna(), positions=[0])
        ax.set_title(net, fontsize=11)
        ax.set_xticks([])
        ax.set_ylabel('Proportion')
    plt.suptitle('Network Proportion Distributions — Baseline (ABCC-a3)', fontsize=14)
    plt.tight_layout()
    plt.savefig(cfg.FIG_DIR / 'fig_phase1_network_proportions_baseline.png', dpi=150)
    plt.close()

    # ── Save processed datasets ───────────────────────────────────────────────────
    df_base.to_csv(cfg.DAT_DIR / 'df_base.csv', index=False)
    df_y2.to_csv(cfg.DAT_DIR / 'df_y2.csv', index=False)
    df_y4.to_csv(cfg.DAT_DIR / 'df_y4.csv', index=False)
    df_y6.to_csv(cfg.DAT_DIR / 'df_y6.csv', index=False)
    df_long_base.to_csv(cfg.DAT_DIR / 'df_long_base.csv', index=False)
    df_long_y2.to_csv(cfg.DAT_DIR / 'df_long_y2.csv', index=False)

    df_base[['sub_ID'] + prop_cols].describe().T.round(5).to_csv(cfg.TAB_DIR / 'phase1_network_desc_baseline.csv')
    df_y2[['sub_ID']   + prop_cols].describe().T.round(5).to_csv(cfg.TAB_DIR / 'phase1_network_desc_year2.csv')
    df_y4[['sub_ID']   + prop_cols].describe().T.round(5).to_csv(cfg.TAB_DIR / 'phase1_network_desc_year4.csv')
    df_y6[['sub_ID']   + prop_cols].describe().T.round(5).to_csv(cfg.TAB_DIR / 'phase1_network_desc_year6.csv')

    log()
    log(f'Saved: df_base N={len(df_base)}, df_y2 N={len(df_y2)}, '
        f'df_y4 N={len(df_y4)}, df_y6 N={len(df_y6)}, '
        f'df_long_base/y2 N={len(df_long_base)}')

    with open(cfg.DAT_DIR / 'progress_log.txt', 'w') as f:
        f.write('ELA–Topography ABCD Analysis Progress Log\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Template: ABCC-a3-9to16 (new data, 10-min QC threshold)\n')
        f.write(f'PHASE 1 COMPLETE\n')
        f.write(f'  Baseline N: {len(df_base)}\n')
        f.write(f'  Year-2 N:   {len(df_y2)}\n')
        f.write(f'  Year-4 N:   {len(df_y4)}\n')
        f.write(f'  Year-6 N:   {len(df_y6)}\n')
        f.write(f'  Long N:     {len(df_long_base)}\n\n')
        f.write('\n'.join(log_lines))

    log()
    log('Phase 1 complete.')


if __name__ == '__main__':
    main()
