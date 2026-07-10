"""
Cross-sectional mediation of cognition at EACH outcome timepoint.

threat (baseline) -> SCAN (baseline) -> cognition (wave W)
  cognition = age-corrected NIH composite measured at wave W
  (nc_y_nihtb__comp__{cryst,fluid}__agecorr_score in each wave dataframe)

Tested: crystallized @ y2, y4, y6 ; fluid @ y6 (only wave with fluid data).
Same spec as the production cognition mediation: NO baseline-cognition covariate;
covariates = age@W, sex, site@W (path b) and age_base/sex/fd_base/site_base (path a).
5000-family cluster bootstrap. Confirms whether crystallized is the only
significant cross-sectional cognitive mediation and at which waves.
"""
import sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from pathlib import Path

sys.path.insert(0, str(next(a for a in Path(__file__).resolve().parents if (a/'config.py').exists())))
from config import DAT_DIR, TAB_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)
N_BOOT = 5000
PRED, MED = 'threat_composite', 'prop_SCAN'
CRYST = 'nc_y_nihtb__comp__cryst__agecorr_score'
FLUID = 'nc_y_nihtb__comp__fluid__agecorr_score'

# (wave label, wave df file, measure col, measure name)
TESTS = [
    ('y2', 'df_y2.csv', CRYST, 'Crystallized'),
    ('y4', 'df_y4.csv', CRYST, 'Crystallized'),
    ('y6', 'df_y6.csv', CRYST, 'Crystallized'),
    ('y2', 'df_y2.csv', FLUID, 'Fluid'),
    ('y4', 'df_y4.csv', FLUID, 'Fluid'),
    ('y6', 'df_y6.csv', FLUID, 'Fluid'),
]

def log(m=''): print(m, flush=True)

df_base = pd.read_csv(DAT_DIR / 'df_base.csv')
fb = 'fd' if 'fd' in df_base.columns else 'rest_mean_FD'
sb = 'study_site' if 'study_site' in df_base.columns else 'study_site_baseline'
db = df_base[['sub_ID','family_id','interview_age','sex_num',fb,sb,PRED,MED]].copy()
db = db.rename(columns={'interview_age':'age_base', fb:'fd_base', sb:'site_base'})

def design(frame, cols, site_col):
    parts=[frame[c].values.astype(float) for c in cols]
    parts.append(pd.get_dummies(frame[site_col], prefix='s', drop_first=True, dtype=float).values)
    return np.column_stack([np.ones(len(frame))]+parts)

def path_a(frame):
    cols=[PRED,'age_base','sex_num','fd_base']
    t=frame[[MED]+cols+['site_base','family_id']].dropna()
    X=design(t,cols,'site_base')
    try:
        r=MixedLM(t[MED].values,X,groups=t['family_id'].values,
                  exog_re=np.ones((len(t),1))).fit(reml=True,method='lbfgs',maxiter=200)
        if r.converged: return r.params[1], r.pvalues[1]
    except Exception: pass
    r=sm.OLS(t[MED].values,X).fit(); return r.params[1], r.pvalues[1]

def boot(seed, fg, Xa, ya, Xb, yb, midx):
    rng=np.random.default_rng(seed)
    rows=np.concatenate([fg[i] for i in rng.integers(0,len(fg),size=len(fg))])
    try:
        ba=np.linalg.lstsq(Xa[rows],ya[rows],rcond=None)[0][1]
        bb=np.linalg.lstsq(Xb[rows],yb[rows],rcond=None)[0][midx]
    except Exception: return np.nan
    return ba*bb

rows=[]
for wave, f, meas, mname in TESTS:
    dw = pd.read_csv(DAT_DIR / f)
    if meas not in dw.columns:
        log(f'  {mname:<12} @ {wave}: measure column missing — skip'); continue
    keep = dw[['sub_ID','interview_age','study_site', meas]].copy()
    keep = keep.rename(columns={'interview_age':'age_w','study_site':'site_w', meas:'outcome'})
    keep = keep.dropna(subset=['outcome'])
    if len(keep) < 80:
        rows.append(dict(measure=mname, wave=wave, n=len(keep), note='insufficient data (no fluid at this wave)',
                         beta_a=np.nan, beta_b=np.nan, indirect=np.nan,
                         ci_lo=np.nan, ci_hi=np.nan, boot_p=np.nan, ci_excl_0=False))
        log(f'  {mname:<12} @ {wave}: only N={len(keep)} with data — skip mediation'); continue
    df = db.merge(keep, on='sub_ID', how='inner').dropna(subset=[PRED, MED, 'outcome'])
    cb = [PRED, MED, 'age_w', 'sex_num']
    t = df.dropna(subset=cb+['outcome','site_w','site_base','age_base','fd_base','family_id']).reset_index(drop=True)
    n = len(t)
    ba, pa = path_a(t)
    Xb = design(t, cb, 'site_w'); yb = t['outcome'].values.astype(float)
    rb = sm.OLS(yb, Xb).fit(cov_type='HC3'); midx = 1 + cb.index(MED)
    bb, pb = rb.params[midx], rb.pvalues[midx]
    ind = ba*bb
    ca=[PRED,'age_base','sex_num','fd_base']
    Xa=design(t,ca,'site_base'); ya=t[MED].values.astype(float)
    fams=t['family_id'].unique()
    inv=t['family_id'].map({fm:i for i,fm in enumerate(fams)}).values
    fg=[np.where(inv==i)[0] for i in range(len(fams))]
    bs=np.array([boot(i,fg,Xa,ya,Xb,yb,midx) for i in range(N_BOOT)])
    bs=bs[~np.isnan(bs)]
    lo,hi=np.percentile(bs,[2.5,97.5])
    prop=np.mean(bs<0) if ind>0 else np.mean(bs>0)
    bp=2*min(prop,1-prop)
    rows.append(dict(measure=mname, wave=wave, n=n, note='',
                     beta_a=round(ba,5), p_a=round(pa,5), beta_b=round(bb,4), p_b=round(pb,4),
                     indirect=round(ind,5), ci_lo=round(lo,5), ci_hi=round(hi,5),
                     boot_p=round(bp,5), ci_excl_0=bool(lo*hi>0)))
    log(f'  {mname:<12} @ {wave}: N={n}  b_b={bb:+.2f}  indirect={ind:+.4f}  '
        f'CI=[{lo:+.4f},{hi:+.4f}]  boot_p={bp:.4f}  {"*" if lo*hi>0 else ""}')

out=pd.DataFrame(rows)
out.to_csv(TAB_DIR/'phase6_cognition_timepoints.csv',index=False)
log(f'\nSaved {TAB_DIR/"phase6_cognition_timepoints.csv"}')
log('\n'+'='*60+'\nSUMMARY (cross-sectional SCAN mediation of cognition)\n'+'='*60)
for _,r in out.iterrows():
    if r.get('note'):
        log(f'  {r.measure:<12} @ {r.wave}: {r.note}'); continue
    flag='SIG (CI excl 0)' if r.ci_excl_0 else 'n.s.'
    log(f'  {r.measure:<12} @ {r.wave}:  N={r.n}  indirect={r.indirect:+.4f}  '
        f'CI=[{r.ci_lo:+.4f},{r.ci_hi:+.4f}]  boot_p={r.boot_p:.4f}  -> {flag}')
log('\nDone.')
