"""
fix_y2_age.py — Correct the Year-2 interview_age in df_y2.csv.

BUG: covariates file had no 2-year age column, so config TIMEPOINT_COV['02A']
used the baseline 'interview_age' as a placeholder. df_y2 age was therefore
baseline age (~9.9) instead of the true Year-2 age (~12.1).

FIX: pull the true Year-2 visit age (in YEARS) from
  data/Covariates_extracted/ABCD_Covariates/ab_g_dyn.tsv
  (column ab_g_dyn__visit_age, session_id == 'ses-02A', key participant_id).
Overwrite df_y2.csv['interview_age'] in place (backup kept once).

Reproducible: the correction is idempotent (the source visit-age is
authoritative, so re-applying it to already-corrected data is a no-op), and
safe to re-run after any phase1 pipeline re-run. The transform is exposed as
pure functions so its accuracy and idempotency are unit-tested in
code/tests/test_fix_y2_age.py.
"""
import shutil
import pandas as pd
from pathlib import Path

from adtopo.config import cfg


def load_y2_true_age(ab_g_dyn_file):
    """Return a DataFrame [sub_ID, age_y2_true] of true ses-02A visit ages (years)."""
    dyn = pd.read_csv(ab_g_dyn_file, sep='\t',
                      usecols=['participant_id', 'session_id', 'ab_g_dyn__visit_age'])
    y2age = dyn[dyn['session_id'] == 'ses-02A'][['participant_id', 'ab_g_dyn__visit_age']]
    y2age = y2age.dropna(subset=['ab_g_dyn__visit_age']).copy()
    return y2age.rename(columns={'participant_id': 'sub_ID',
                                 'ab_g_dyn__visit_age': 'age_y2_true'})


def correct_y2_age(df_y2, y2age):
    """Overwrite df_y2['interview_age'] with the true ses-02A age where available.

    Pure and idempotent: the source age is authoritative, so applying this to
    already-corrected data returns an equal frame; unmatched rows keep their
    prior age. Returns (corrected_df, n_matched, n_unmatched).
    """
    df = df_y2.merge(y2age, on='sub_ID', how='left', validate='many_to_one')
    n_match = int(df['age_y2_true'].notna().sum())
    n_miss  = int(df['age_y2_true'].isna().sum())
    df['interview_age'] = df['age_y2_true'].fillna(df['interview_age'])
    df = df.drop(columns=['age_y2_true'])
    return df, n_match, n_miss


def main():
    y2_path  = cfg.DAT_DIR / 'df_y2.csv'
    bak_path = cfg.DAT_DIR / 'df_y2.csv.bak_preY2agefix'

    print('=' * 64)
    print('fix_y2_age.py — correcting df_y2 interview_age')
    print('=' * 64)

    y2age = load_y2_true_age(cfg.AB_G_DYN_FILE)
    print(f'ses-02A ages available: {len(y2age)}  '
          f'(mean={y2age["age_y2_true"].mean():.2f} years)')

    df = pd.read_csv(y2_path)
    if not bak_path.exists():
        shutil.copy2(y2_path, bak_path)
        print(f'Backup written: {bak_path.name}')
    else:
        print(f'Backup already exists: {bak_path.name} (not overwriting)')

    old_mean = df['interview_age'].mean()
    df, n_match, n_miss = correct_y2_age(df, y2age)
    print(f'df_y2 N={len(df)}: matched={n_match}, unmatched={n_miss}')
    if n_miss:
        print('  WARNING: unmatched subjects keep their old (baseline) age — investigate.')

    df.to_csv(y2_path, index=False)
    print(f'interview_age mean:  {old_mean:.2f}  ->  {df["interview_age"].mean():.2f} years')
    print(f'Saved corrected {y2_path.name}')
    print('Done.')


if __name__ == '__main__':
    main()
