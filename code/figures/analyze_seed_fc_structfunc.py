"""
Vertexwise structure–function correspondence (the payoff of the SCAN-seed FC analysis).

Compares, vertex by vertex across cortex:
  adversity-driven ANATOMY  = SCAN_density_threat_baseline_diff.dscalar (high−low SCAN prob)
  adversity-driven FUNCTION = scan_seed_fc_groupmean.dscalar map-3 (high−low SCAN-seed FC)
with an Alexander-Bloch spin test (reuses Analysis D), and tests whether the FC change
localises to the sensorimotor pole of the principal gradient (as the encroachment does, Fig 2c).

Outputs:
  outputs/tables/seed_fc_structfunc_summary.txt
  outputs/figures/nn/fig3/fig3d_structfunc_vertexwise.{pdf,png}
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('NEUROMAPS_DATA', str(ROOT / 'data/neuromaps_cache'))
import figsrc as F
fs = F.fs; fs.set_style()
from adtopo.logging_utils import get_logger
_log = get_logger('analyze_seed_fc_structfunc')

from neuromaps.datasets import fetch_annotation
from neuromaps.nulls import alexander_bloch
from neuromaps.stats import compare_images

NVERT = 32492
ENC  = ROOT / 'outputs/cifti_for_workbench/SCAN_density_threat_baseline_diff.dscalar.nii'
FC   = ROOT / 'outputs/cifti_for_workbench/scan_seed_fc_groupmean.dscalar.nii'
OUTT = ROOT / 'outputs/tables/seed_fc_structfunc_summary.txt'
OUT  = str(F.FIG_OUT / 'fig3')
L = []
def log(s=''): _log.info(str(s)); L.append(s)


def cifti_to_full(img, mapidx=0):
    data = np.asarray(img.get_fdata())[mapidx]
    ax = img.header.get_axis(1)
    full = {'CIFTI_STRUCTURE_CORTEX_LEFT': np.full(NVERT, np.nan),
            'CIFTI_STRUCTURE_CORTEX_RIGHT': np.full(NVERT, np.nan)}
    for name, slc, bm in ax.iter_structures():
        if name in full:
            full[name][bm.vertex] = data[slc]
    return np.concatenate([full['CIFTI_STRUCTURE_CORTEX_LEFT'],
                           full['CIFTI_STRUCTURE_CORTEX_RIGHT']])


def main():
    F.Path(OUT).mkdir(parents=True, exist_ok=True)
    log('Vertexwise structure–function correspondence (SCAN-seed FC vs encroachment)')
    enc = cifti_to_full(nib.load(str(ENC)), 0)                 # high−low SCAN density
    fcd = cifti_to_full(nib.load(str(FC)), 2)                  # high−low SCAN-seed FC (map 3)

    grad = fetch_annotation(source='margulies2016', desc='fcgradient01', space='fsLR', den='32k')
    gL, gR = grad
    g_full = np.concatenate([nib.load(str(gL)).darrays[0].data,
                             nib.load(str(gR)).darrays[0].data]).astype(float)
    g_full[g_full == 0] = np.nan

    valid = np.isfinite(enc) & np.isfinite(fcd)
    log(f'valid cortical vertices (both maps): {valid.sum()}')
    ev, fv = enc[valid], fcd[valid]
    r_p, _ = stats.pearsonr(ev, fv)
    r_s, _ = stats.spearmanr(ev, fv)
    log(f'  Pearson  r(encroach-diff, FC-diff) = {r_p:+.3f}')
    log(f'  Spearman r = {r_s:+.3f}')

    # spin test (Alexander-Bloch); spin the encroachment map, correlate vs FC-diff
    log('  spin test (Alexander-Bloch, 1000 rotations)...')
    enc_sp = enc.copy(); enc_sp[~valid] = np.nan
    fcd_sp = fcd.copy(); fcd_sp[~valid] = np.nan
    nulls = alexander_bloch(enc_sp, atlas='fsLR', density='32k', n_perm=1000, seed=1234)
    r_spin, p_spin = compare_images(enc_sp, fcd_sp, nulls=nulls, metric='pearsonr')
    log(f'  spin-tested r = {r_spin:+.3f}, p_spin = {p_spin:.4g}')

    # gradient localisation of the FC change (top-decile FC-gain vertices)
    gv_valid = np.isfinite(g_full) & valid
    gz = (g_full - np.nanmean(g_full)) / np.nanstd(g_full)
    hot = np.zeros_like(valid)
    thr = np.nanpercentile(fcd[gv_valid], 90)
    hot[gv_valid] = fcd[gv_valid] >= thr
    g_pct = stats.rankdata(g_full[gv_valid]) / gv_valid.sum() * 100
    hot_in_valid = fcd[gv_valid] >= thr
    log('')
    log('FC-gain hotspot (top-decile FC increase) on the principal gradient:')
    log(f'  whole cortex pct          : {np.mean(g_pct):.1f}')
    log(f'  FC-gain hotspot pct       : {np.mean(g_pct[hot_in_valid]):.1f}')
    log(f'  FC-gain hotspot mean G1(z): {np.nanmean(gz[hot]):.3f}')

    OUTT.write_text('\n'.join(L))
    log(f'wrote {OUTT}')

    # ── figure: vertexwise FC-diff vs encroachment-diff ──────────────────────────
    x, y = ev, fv
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    # equal-width bins across the trimmed range (handles the zero-inflated x cleanly)
    edges = np.linspace(xlo, xhi, 13)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ib = np.clip(np.digitize(x, edges[1:-1]), 0, len(ctr) - 1)
    by = np.array([y[ib == b].mean() if (ib == b).sum() else np.nan for b in range(len(ctr))])
    be = np.array([y[ib == b].std(ddof=1) / np.sqrt((ib == b).sum())
                   if (ib == b).sum() > 1 else np.nan for b in range(len(ctr))])

    fig, ax = fs.figure('single', 60)
    ax.axhline(0, color='#d6d6d6', lw=0.3, zorder=0)
    ax.axvline(0, color='#d6d6d6', lw=0.3, zorder=0)
    hb = ax.hexbin(x, y, gridsize=45, bins='log', cmap='Greys', linewidths=0,
                   extent=(xlo, xhi, np.percentile(y, 0.5), np.percentile(y, 99.5)),
                   zorder=1, mincnt=1)
    ax.plot(ctr, by, color=F.SCAN_COLOR, lw=1.2, zorder=3)
    ax.errorbar(ctr, by, yerr=be, fmt='o', ms=3, color=F.SCAN_COLOR, ecolor=F.SCAN_COLOR,
                elinewidth=0.7, capsize=1.5, markeredgecolor='white',
                markeredgewidth=0.3, zorder=4)
    ax.text(0.04, 0.96, f'spin r = {r_spin:.2f}\nspin p = {p_spin:.3f}\n{int(valid.sum()/1000)}k vertices',
            transform=ax.transAxes, va='top', ha='left', fontsize=6)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(np.percentile(y, 0.5), np.percentile(y, 99.5))
    ax.set_xlabel('Adversity → SCAN expansion\n(high − low SCAN prob., per vertex)')
    ax.set_ylabel('Adversity → SCAN coupling\n(high − low Fisher-z)')
    fs.despine(ax)
    fs.panel_label(ax, 'd')
    fs.save(fig, 'fig3d_structfunc_vertexwise', outdir=OUT)
    log('wrote fig3d_structfunc_vertexwise')


if __name__ == '__main__':
    main()
