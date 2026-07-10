"""
fix_y2_age.py — Correct the Year-2 interview_age in df_y2.csv.

BUG: covariates file had no 2-year age column, so config TIMEPOINT_COV['02A']
used the baseline 'interview_age' as a placeholder. df_y2 age was therefore
baseline age (~9.9) instead of the true Year-2 age (~12.1).

FIX: pull the true Year-2 visit age (in YEARS) from
  data/Covariates_extracted/ABCD_Covariates/ab_g_dyn.tsv
  (column ab_g_dyn__visit_age, session_id == 'ses-02A', key participant_id).
Overwrite df_y2.csv['interview_age'] in place (backup kept once).

Reproducible: safe to re-run; re-apply after any phase1 pipeline re-run.
"""
import sys, shutil
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import DAT_DIR, AB_G_DYN_FILE

AB_G_DYN = AB_G_DYN_FILE
Y2 = DAT_DIR / 'df_y2.csv'
BAK = DAT_DIR / 'df_y2.csv.bak_preY2agefix'

print('=' * 64)
print('fix_y2_age.py — correcting df_y2 interview_age')
print('=' * 64)

# 1. Year-2 age (years) from ab_g_dyn
dyn = pd.read_csv(AB_G_DYN, sep='\t', usecols=['participant_id', 'session_id',
                                               'ab_g_dyn__visit_age'])
y2age = dyn[dyn['session_id'] == 'ses-02A'][['participant_id', 'ab_g_dyn__visit_age']]
y2age = y2age.dropna(subset=['ab_g_dyn__visit_age']).copy()
y2age = y2age.rename(columns={'participant_id': 'sub_ID',
                              'ab_g_dyn__visit_age': 'age_y2_true'})
print(f'ses-02A ages available: {len(y2age)}  '
      f'(mean={y2age["age_y2_true"].mean():.2f} years)')

# 2. Backup once
df = pd.read_csv(Y2)
if not BAK.exists():
    shutil.copy2(Y2, BAK)
    print(f'Backup written: {BAK.name}')
else:
    print(f'Backup already exists: {BAK.name} (not overwriting)')

old_mean = df['interview_age'].mean()

# 3. Merge + overwrite
df = df.merge(y2age, on='sub_ID', how='left', validate='many_to_one')
n_match = df['age_y2_true'].notna().sum()
n_miss  = df['age_y2_true'].isna().sum()
print(f'df_y2 N={len(df)}: matched={n_match}, unmatched={n_miss}')
if n_miss:
    print('  WARNING: unmatched subjects keep their old (baseline) age — investigate:')
    print('  ', df.loc[df['age_y2_true'].isna(), 'sub_ID'].head().tolist())

df['interview_age'] = df['age_y2_true'].fillna(df['interview_age'])
df = df.drop(columns=['age_y2_true'])
df.to_csv(Y2, index=False)

new_mean = df['interview_age'].mean()
print(f'interview_age mean:  {old_mean:.2f}  ->  {new_mean:.2f} years')
print(f'Saved corrected {Y2.name}')
print('Done.')
