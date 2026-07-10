"""
Build extended covariates file adding year-4 and year-6 columns.

Inputs (from Covariates.zip, extracted to ABCD_TM/data/Covariates_extracted/):
  - mr_y_qc__mot.tsv   : resting-state mean FD per session
  - ab_g_dyn.tsv       : study site + age at visit per session

Output:
  adversity_topography/data/covariates/5_covariates_extended.xlsx

New columns added (year-4 and year-6 only; baseline/yr-2 already in original file):
  rest_mean_FD_4yrFU, study_site_4yrFU, interview_age_4yrFU  (age in years)
  rest_mean_FD_6yrFU, study_site_6yrFU, interview_age_6yrFU
"""
import pandas as pd
from pathlib import Path

from adtopo.config import DATA_DIR, AB_COVARIATES_DIR
from adtopo.logging_utils import get_logger
_log = get_logger('build_covariates')

SRC_DIR    = AB_COVARIATES_DIR
COV_IN     = DATA_DIR / 'covariates' / '5_covariates.xlsx'
COV_OUT    = DATA_DIR / 'covariates' / '5_covariates_extended.xlsx'

SESSION_MAP = {
    '4yrFU': 'ses-04A',
    '6yrFU': 'ses-06A',
}

def log(msg=''):
    _log.info(str(msg))

log('Loading source files...')
mot = pd.read_csv(SRC_DIR / 'mr_y_qc__mot.tsv', sep='\t', na_values=['n/a'],
                  usecols=['participant_id', 'session_id', 'mr_y_qc__mot__rsfmri__mot_mean'])
dyn = pd.read_csv(SRC_DIR / 'ab_g_dyn.tsv', sep='\t', na_values=['n/a'],
                  usecols=['participant_id', 'session_id', 'ab_g_dyn__design_site', 'ab_g_dyn__visit_age'])

log(f'  Motion QC rows: {len(mot):,}')
log(f'  Visit dynamics rows: {len(dyn):,}')

cov = pd.read_excel(COV_IN, na_values=['NA', 'na', 'N/A', ''])
log(f'  Existing covariates: {len(cov):,} subjects, {len(cov.columns)} columns')

# Build one wide-format block per follow-up year, then merge into cov
for suffix, session_id in SESSION_MAP.items():
    log()
    log(f'--- {suffix} ({session_id}) ---')

    mot_sess = mot[mot['session_id'] == session_id].copy()
    dyn_sess = dyn[dyn['session_id'] == session_id].copy()

    mot_sess = mot_sess.rename(columns={
        'participant_id':                      'sub_ID',
        'mr_y_qc__mot__rsfmri__mot_mean':      f'rest_mean_FD_{suffix}',
    }).drop(columns='session_id')

    dyn_sess = dyn_sess.rename(columns={
        'participant_id':          'sub_ID',
        'ab_g_dyn__design_site':   f'study_site_{suffix}',
        'ab_g_dyn__visit_age':     f'interview_age_{suffix}',
    }).drop(columns='session_id')

    sess_df = mot_sess.merge(dyn_sess, on='sub_ID', how='outer')

    log(f'  Rows in merged session block: {len(sess_df):,}')
    for col in [f'rest_mean_FD_{suffix}', f'study_site_{suffix}', f'interview_age_{suffix}']:
        n_nan = sess_df[col].isna().sum()
        log(f'  {col}: {len(sess_df) - n_nan:,} non-NaN, {n_nan:,} NaN')

    cov = cov.merge(sess_df, on='sub_ID', how='left')
    log(f'  Subjects gaining FD data: {cov[f"rest_mean_FD_{suffix}"].notna().sum():,} / {len(cov):,}')

log()
log('Final column list:')
for col in cov.columns:
    log(f'  {col}')

log()
log(f'Writing to {COV_OUT}')
cov.to_excel(COV_OUT, index=False)
log('Done.')
