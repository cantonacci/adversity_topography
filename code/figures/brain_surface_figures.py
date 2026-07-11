"""
Brain surface figures — fsLR-32k native rendering via nilearn.

Renders directly on local fsLR-32k very_inflated surfaces (no resampling).
Background: per-vertex curvature pre-clipped to the p5–p95 range and
normalized to [-1, 1], giving a smooth sulcal-depth gradient rather than
the raw high-frequency T1w curvature texture or a hard binary step.

Produces:
  brain_surface/fig_brain_network_labels.{pdf,png}   — network parcellation
  brain_surface/old_style/FigureBrain_Overall.{pdf,png}
  brain_surface/old_style/FigureBrain_{Threat,Deprivation,Unpredictability}.{pdf,png}
"""

import os, tempfile, warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from nilearn import plotting
from pathlib import Path

from adtopo.config import cfg
from adtopo.logging_utils import get_logger
_log = get_logger('brain_surface_figures')

mpl.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':       9,
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
})

# ── Directories ────────────────────────────────────────────────────────────────
OUT_DIR    = cfg.FIG_DIR / 'brain_surface' / 'old_style'
PARCEL_DIR = cfg.FIG_DIR / 'brain_surface'
SURF_DIR   = cfg.FIG_DIR / 'brain_surface' / 'inflated_surfaces'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARCEL_DIR.mkdir(parents=True, exist_ok=True)

SURF_L = SURF_DIR / 'hemi-L_very_inflated.surf.gii'
SURF_R = SURF_DIR / 'hemi-R_very_inflated.surf.gii'
CURV_L = SURF_DIR / 'hemi-L_curvature.shape.gii'
CURV_R = SURF_DIR / 'hemi-R_curvature.shape.gii'

# ── Network labels and colors ─────────────────────────────────────────────────
NET_MAP = {
    'DMN':1, 'VIS':2, 'FP':3, 'DAN':5, 'VAN':7,
    'SAL':8, 'CO':9, 'SMD':10, 'SML':11, 'AUD':12,
    'Tpole':13, 'MTL':14, 'PMN':15, 'PON':16, 'SCAN':18,
}
ATLAS_NET_COLORS = {
    'DMN':'#FF0000', 'VIS':'#000099', 'FP':'#FFFF00',
    'DAN':'#00FF00', 'VAN':'#0D85A0', 'SAL':'#000000',
    'CO':'#6600CC',  'SMD':'#66FFFF', 'SML':'#FF8000',
    'AUD':'#B266FF', 'Tpole':'#006699', 'MTL':'#66FF66',
    'PMN':'#3C3CFB', 'PON':'#EFEFEF', 'SCAN':'#8E0067',
}
PARCEL_NET_ORDER = ['DMN','VIS','FP','DAN','VAN','SAL','CO','SMD',
                    'SML','AUD','Tpole','MTL','PMN','PON','SCAN']
N_NETS = len(PARCEL_NET_ORDER)

# ── Colormaps ──────────────────────────────────────────────────────────────────
CMAP_PARCEL  = ListedColormap(
    [ATLAS_NET_COLORS[n] for n in PARCEL_NET_ORDER], name='parcel', N=N_NETS)
CMAP_OVERALL = LinearSegmentedColormap.from_list(
    'overall', ['#FFF3CC', '#FF8800', '#990000'], N=256)
CMAP_THREAT  = LinearSegmentedColormap.from_list(
    'threat',  ['#2E74B5', '#FFFFFF', '#C0392B'], N=256)
CMAP_DEPR    = LinearSegmentedColormap.from_list(
    'depr',    ['#C0392B', '#FFFFFF', '#0072B2'], N=256)
CMAP_UNPRED  = LinearSegmentedColormap.from_list(
    'unpred',  ['#1A7845', '#FFFFFF', '#8E44AD'], N=256)

VIEW_LABELS = ['L lateral', 'L medial', 'R medial', 'R lateral']


def log(msg=''):
    _log.info(str(msg))


def _load_smooth_bg(curv_path):
    """
    Load curvature and clip to the p5–p95 range, then normalize to [-1, 1].
    This removes extreme values and fine-grained T1w texture, giving a smooth
    sulcal-depth gradient suitable for use as a nilearn background.
    """
    curv = nib.load(str(curv_path)).darrays[0].data.astype(np.float32)
    lo, hi = np.percentile(curv, 5), np.percentile(curv, 95)
    curv_c = np.clip(curv, lo, hi)
    # Normalize to [-1, 1]
    span = (hi - lo) / 2.0
    mid  = (hi + lo) / 2.0
    return (curv_c - mid) / span


# Pre-load backgrounds once at module level
_BG_L = _load_smooth_bg(CURV_L) if CURV_L.exists() else None
_BG_R = _load_smooth_bg(CURV_R) if CURV_R.exists() else None


# ── Atlas loading ──────────────────────────────────────────────────────────────

def load_atlas():
    atlas_path = cfg.ATLAS_DIR / 'abcd_template_matching_v2_combined_clusters_thresh0.50.dlabel.nii'
    log(f'Loading atlas: {atlas_path.name}')
    img   = nib.load(str(atlas_path))
    data  = img.get_fdata()[0]
    bm_ax = img.header.get_axis(1)

    atlas_L = np.full(32492, np.nan)
    atlas_R = np.full(32492, np.nan)
    for name, slc, _ in bm_ax.iter_structures():
        verts = bm_ax.vertex[slc]
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            atlas_L[verts] = data[slc]
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            atlas_R[verts] = data[slc]
    atlas_L[atlas_L == 0] = np.nan
    atlas_R[atlas_R == 0] = np.nan
    log(f'  L: {(~np.isnan(atlas_L)).sum():,} | R: {(~np.isnan(atlas_R)).sum():,} labeled verts')
    return atlas_L, atlas_R


def build_parcel_map(atlas):
    out = np.full(len(atlas), np.nan)
    for idx, net in enumerate(PARCEL_NET_ORDER):
        out[atlas == NET_MAP[net]] = float(idx + 1)
    return out


def build_score_map(atlas, scores_dict):
    out = np.full(len(atlas), np.nan)
    for net, score in scores_dict.items():
        if net in NET_MAP and score is not None:
            try:
                v = float(score)
                if not np.isnan(v):
                    out[atlas == NET_MAP[net]] = v
            except (TypeError, ValueError):
                pass
    return out


# ── Rendering helpers ──────────────────────────────────────────────────────────

def tight_crop(img, pad=14):
    bg   = img[0, 0]
    mask = np.any(np.abs(img - bg) > 0.03, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    r0 = max(0, rows[0] - pad);  r1 = min(img.shape[0], rows[-1] + pad)
    c0 = max(0, cols[0] - pad);  c1 = min(img.shape[1], cols[-1] + pad)
    return img[r0:r1, c0:c1]


def resize_height(img, h):
    from PIL import Image as PILImage
    pil = PILImage.fromarray((img * 255).astype(np.uint8))
    w   = int(pil.width * h / pil.height)
    return np.array(pil.resize((w, h), PILImage.LANCZOS)) / 255.0


def render_four_views(data_L, data_R, cmap, vmin, vmax, dpi=200):
    """Render L-lat, L-med, R-med, R-lat on local fsLR-32k very_inflated surfaces."""
    views_spec = [
        (data_L, SURF_L, _BG_L, 'left',  'lateral'),
        (data_L, SURF_L, _BG_L, 'left',  'medial'),
        (data_R, SURF_R, _BG_R, 'right', 'medial'),
        (data_R, SURF_R, _BG_R, 'right', 'lateral'),
    ]
    imgs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (data, surf, bg, hemi, view) in enumerate(views_spec):
            tmp = os.path.join(tmpdir, f'{i}.png')
            try:
                sfig = plotting.plot_surf_stat_map(
                    surf_mesh  = str(surf),
                    stat_map   = data,
                    bg_map     = bg,       # pre-normalized array, smooth gradient
                    hemi       = hemi,
                    view       = view,
                    cmap       = cmap,
                    vmin       = vmin,
                    vmax       = vmax,
                    colorbar   = False,
                    bg_on_data = True,
                    title      = None,
                )
                sfig.savefig(tmp, dpi=dpi, bbox_inches='tight',
                             facecolor='white', edgecolor='none')
                plt.close(sfig)
            except Exception as e:
                log(f'    WARNING view {i} ({hemi} {view}): {e}')
                from PIL import Image as PILImage
                PILImage.fromarray(np.ones((300, 300, 3), np.uint8) * 220).save(tmp)
            imgs.append(tight_crop(mpimg.imread(tmp)))
    return imgs


def assemble_figure(imgs, title, cbar_label, cmap, vmin, vmax,
                    outname, out_dir, discrete=False):
    log(f'  Assembling {outname}...')
    target_h = max(im.shape[0] for im in imgs)
    imgs_r   = [resize_height(im, target_h) for im in imgs]
    canvas   = np.concatenate(imgs_r, axis=1)

    fig_h = 3.4
    fig_w = fig_h * canvas.shape[1] / canvas.shape[0] + 0.9
    fig, (ax_brain, ax_cb) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={'width_ratios': [canvas.shape[1], 22], 'wspace': 0.01})

    ax_brain.imshow(canvas, aspect='equal', interpolation='lanczos')
    ax_brain.axis('off')

    x_cursor = 0
    for im, lbl in zip(imgs_r, VIEW_LABELS):
        ax_brain.text((x_cursor + im.shape[1] / 2), canvas.shape[0] + 5, lbl,
                      ha='center', va='top', fontsize=6, color='#444444',
                      transform=ax_brain.transData)
        x_cursor += im.shape[1]

    if discrete:
        bounds = np.arange(0.5, N_NETS + 1.5, 1.0)
        norm   = BoundaryNorm(bounds, N_NETS)
        sm     = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb, orientation='vertical',
                          ticks=np.arange(1, N_NETS + 1))
        cb.ax.set_yticklabels(PARCEL_NET_ORDER, fontsize=5.5)
        cb.ax.set_ylabel('Network', fontsize=7, labelpad=4)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm   = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb, orientation='vertical')
        cb.set_label(cbar_label, fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.4)

    fig.suptitle(title, fontsize=9.5, fontweight='bold', y=1.01)
    for fmt in ('pdf', 'png'):
        fig.savefig(str(out_dir / f'{outname}.{fmt}'),
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        log(f'    → {outname}.{fmt}')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log('=' * 70)
    log('BRAIN SURFACE FIGURES — fsLR-32k (nilearn, smooth curvature background)')
    log('=' * 70)

    missing = [p for p in (SURF_L, SURF_R, CURV_L, CURV_R) if not p.exists()]
    if missing:
        log('ERROR: missing files:')
        for p in missing: log(f'  {p}')
        return
    log(f'Surfaces + curvature OK  (bg arrays pre-loaded)')

    atlas_L, atlas_R = load_atlas()

    # ── [1] Network parcellation ──────────────────────────────────────────────
    log('\n[1] Network parcellation...')
    parcel_L = build_parcel_map(atlas_L)
    parcel_R = build_parcel_map(atlas_R)
    imgs = render_four_views(parcel_L, parcel_R,
                             cmap=CMAP_PARCEL,
                             vmin=0.5, vmax=N_NETS + 0.5, dpi=200)
    assemble_figure(imgs,
                    title='Functional Network Parcellation (ABCC Template)',
                    cbar_label='Network',
                    cmap=CMAP_PARCEL, vmin=0.5, vmax=N_NETS + 0.5,
                    outname='fig_brain_network_labels',
                    out_dir=PARCEL_DIR,
                    discrete=True)

    # ── Load phase3 data ──────────────────────────────────────────────────────
    log('\nLoading phase3 tables...')
    bsi_path = cfg.TAB_DIR / 'phase3_composites_brain_surface_inputs.csv'
    dr2_path = cfg.TAB_DIR / 'phase3_individual_delta_r2_baseline.csv'
    if not bsi_path.exists():
        log(f'ERROR: {bsi_path} not found'); return

    bsi      = pd.read_csv(bsi_path)
    bsi_base = bsi[bsi['timepoint'] == 'baseline'].set_index('network')

    overall_scores = (pd.read_csv(dr2_path)
                        .pipe(lambda d: d[d['timepoint']=='baseline']
                              if 'timepoint' in d.columns else d)
                        .set_index('network')['delta_R2'].to_dict()
                     ) if dr2_path.exists() else bsi_base['delta_R2'].to_dict()

    threat_scores = bsi_base['threat_composite_beta'].to_dict()
    depriv_scores = bsi_base['deprivation_composite_beta'].to_dict()
    unpred_scores = bsi_base['unpredictability_composite_beta'].to_dict()

    overall_L = build_score_map(atlas_L, overall_scores)
    overall_R = build_score_map(atlas_R, overall_scores)
    threat_L  = build_score_map(atlas_L, threat_scores)
    threat_R  = build_score_map(atlas_R, threat_scores)
    depriv_L  = build_score_map(atlas_L, depriv_scores)
    depriv_R  = build_score_map(atlas_R, depriv_scores)
    unpred_L  = build_score_map(atlas_L, unpred_scores)
    unpred_R  = build_score_map(atlas_R, unpred_scores)

    all_beta  = np.array([v for d in (threat_scores, depriv_scores, unpred_scores)
                          for v in d.values()], dtype=float)
    beta_vmax = float(np.nanmax(np.abs(all_beta)))
    dr2_vmax  = float(np.nanmax(np.array(list(overall_scores.values()), dtype=float)))

    # ── [2] Overall ΔR² ──────────────────────────────────────────────────────
    log('\n[2] Overall ΔR² map...')
    imgs = render_four_views(overall_L, overall_R,
                             cmap=CMAP_OVERALL, vmin=0, vmax=dr2_vmax, dpi=200)
    assemble_figure(imgs,
                    title='Overall ELA Sensitivity — ΔR² (10-factor model, Baseline)',
                    cbar_label='ΔR²', cmap=CMAP_OVERALL, vmin=0, vmax=dr2_vmax,
                    outname='FigureBrain_Overall', out_dir=OUT_DIR)

    # ── [3–5] Composite beta maps ─────────────────────────────────────────────
    for name, dL, dR, cmap, outname in [
        ('Threat',           threat_L,  threat_R,  CMAP_THREAT,  'FigureBrain_Threat'),
        ('Deprivation',      depriv_L,  depriv_R,  CMAP_DEPR,    'FigureBrain_Deprivation'),
        ('Unpredictability', unpred_L,  unpred_R,  CMAP_UNPRED,  'FigureBrain_Unpredictability'),
    ]:
        log(f'\n[{name}] beta map...')
        imgs = render_four_views(dL, dR, cmap=cmap,
                                 vmin=-beta_vmax, vmax=beta_vmax, dpi=200)
        assemble_figure(imgs,
                        title=f'{name} — Standardized β (simultaneous composite model, Baseline)',
                        cbar_label=f'{name} β',
                        cmap=cmap, vmin=-beta_vmax, vmax=beta_vmax,
                        outname=outname, out_dir=OUT_DIR)

    log(f'\nDone.')
    log(f'  Parcellation → {PARCEL_DIR}/')
    log(f'  Statistical  → {OUT_DIR}/')


if __name__ == '__main__':
    main()
