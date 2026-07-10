#!/usr/bin/env python3
"""
Analysis F — structure-function correspondence between SCAN encroachment and SCAN FC.

Question: Across the 14 non-SCAN networks, do the networks SCAN encroaches into
(territorially) coincide with the networks whose FC with SCAN is reshaped by threat?

We compute, per network, two "structural" quantities and one "functional" quantity:
  x1 = mean SCAN-encroachment fraction (whole-sample, baseline)         [anatomical overlap]
  x2 = threat -> encroachment standardized beta (OLS, family-clustered SE) [adversity-driven encroachment]
  y  = threat -> SCAN-FC beta (from fc_lme_threat_baseline.csv)            [adversity-driven FC change]

Then correlate y vs x1 and y vs x2 (Pearson + Spearman), with a permutation null
(shuffle the network pairing, 10,000 iters) for inference. n=14 networks.

Outputs:
  outputs/tables/F_encroach_fc_correspondence.csv   (per-network vectors)
  outputs/tables/F_encroach_fc_correspondence_summary.txt
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENC  = ROOT / 'outputs/encroachment/encroachment_baseline.csv'
DFB  = ROOT / 'outputs/data_processed/df_base.csv'
FC   = ROOT / 'outputs/tables/fc_lme_threat_baseline.csv'
OUT_CSV = ROOT / 'outputs/tables/F_encroach_fc_correspondence.csv'
OUT_TXT = ROOT / 'outputs/tables/F_encroach_fc_correspondence_summary.txt'

NETS = ['DMN','VIS','FP','DAN','VAN','SAL','CO','SMD','SML','AUD','Tpole','MTL','PMN','PON']
RNG  = np.random.default_rng(20260620)
N_PERM = 10000

def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

# ---- load & merge -----------------------------------------------------------
enc = pd.read_csv(ENC)
df  = pd.read_csv(DFB)
keep = ['sub_ID','family_id','sex_num','threat_composite','fd','study_site','interview_age']
m = enc.merge(df[keep], on='sub_ID', how='inner')
m['age']  = m['interview_age'] / 12.0
m['site'] = m['study_site'].astype(str)
m = m.dropna(subset=['threat_composite','fd','site','age','sex_num'])
print(f'Merged N = {len(m)}')

# ---- x1: whole-sample mean encroachment fraction (in %) ---------------------
x1 = {n: m[f'encroach_frac_{n}'].mean() * 100 for n in NETS}

# ---- x2: threat -> encroachment standardized beta, family-clustered SE ------
m['threat_z'] = zscore(m['threat_composite'])
x2, x2_p = {}, {}
for n in NETS:
    col = f'encroach_frac_{n}'
    tmp = m[[col,'threat_z','age','sex_num','fd','site','family_id']].dropna().copy()
    tmp['y_z'] = zscore(tmp[col])
    res = smf.ols('y_z ~ threat_z + age + sex_num + fd + C(site)', data=tmp).fit(
        cov_type='cluster', cov_kwds={'groups': tmp['family_id']})
    x2[n]   = res.params['threat_z']
    x2_p[n] = res.pvalues['threat_z']

# ---- y: threat -> SCAN-FC beta ---------------------------------------------
fc = pd.read_csv(FC)
fc_scan = fc[(fc.net1 == 'SCAN') | (fc.net2 == 'SCAN')].copy()
fc_scan['other'] = np.where(fc_scan.net1 == 'SCAN', fc_scan.net2, fc_scan.net1)
yb = dict(zip(fc_scan['other'], fc_scan['beta']))
y  = {n: yb[n] for n in NETS}

# ---- assemble ---------------------------------------------------------------
tab = pd.DataFrame({
    'network': NETS,
    'encroach_pct': [x1[n] for n in NETS],
    'threat_encroach_beta': [x2[n] for n in NETS],
    'threat_encroach_p':    [x2_p[n] for n in NETS],
    'threat_fc_beta':       [y[n] for n in NETS],
}).sort_values('encroach_pct', ascending=False).reset_index(drop=True)
tab.to_csv(OUT_CSV, index=False)

# ---- correspondence stats with permutation null -----------------------------
def perm_corr(a, b, method='pearson'):
    a = np.asarray(a); b = np.asarray(b)
    f = (lambda u,v: stats.pearsonr(u,v)[0]) if method=='pearson' else (lambda u,v: stats.spearmanr(u,v)[0])
    obs = f(a,b)
    null = np.array([f(a, RNG.permutation(b)) for _ in range(N_PERM)])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
    return obs, p

lines = []
lines.append('Analysis F — SCAN encroachment x SCAN-FC correspondence (n=14 networks, baseline)')
lines.append(f'Merged N = {len(m)} subjects')
lines.append('')
lines.append('Per-network table (sorted by encroachment %):')
lines.append(tab.round(4).to_string(index=False))
lines.append('')
for xname, xvec in [('encroach_pct (anatomical overlap)', tab['encroach_pct'].values),
                    ('threat->encroach beta (adversity-driven)', tab['threat_encroach_beta'].values)]:
    yv = tab['threat_fc_beta'].values
    rp, pp = perm_corr(xvec, yv, 'pearson')
    rs, ps = perm_corr(xvec, yv, 'spearman')
    lines.append(f'y = threat->FC beta   vs   x = {xname}')
    lines.append(f'   Pearson  r = {rp:+.3f}   perm-p = {pp:.4f}')
    lines.append(f'   Spearman r = {rs:+.3f}   perm-p = {ps:.4f}')
    lines.append('')

txt = '\n'.join(lines)
OUT_TXT.write_text(txt)
print(txt)
