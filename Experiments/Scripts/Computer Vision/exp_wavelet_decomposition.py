"""
exp_wavelet_decomposition.py
============================
2D Discrete Wavelet Transform decomposition of the Elon Musk image.
Reproduces Figure 3.36 from Szeliski (2021) using PyWavelets (pywt).

HYPOTHESES:
  H1 - Single-level DWT (db4) produces 4 subbands:
       LL (approx), LH (horizontal edges), HL (vertical edges), HH (diagonal).
  H2 - Perfect reconstruction: idwt2 recovers original with max error < 1e-10.
  H3 - Multi-level DWT: LL subband halves exactly at each level.
  H4 - Energy compaction: LL subband holds >85% of total image energy.

LIBRARIES:
  pywt      - PyWavelets (dwt2/idwt2/wavedec2)
  numpy     - numerical computation
  matplotlib - visualisation
  PIL       - image loading

ASSUMPTIONS:
  - Elon Musk image at Data/images/elon_musk.jpg
  - Resized to 256x256 (power of 2 -> exact half-size subbands)
  - Wavelet: 'db4'  |  Mode: 'periodization' (exact critically-sampled sizes)
  - N_LEVELS = 4

REPRODUCIBILITY: fully deterministic.

Reference: Szeliski, Computer Vision, 2nd ed., Figure 3.36, p.160
Author: Ocean Library Experiment
Date:   2026-06-23
"""

import sys
import numpy as np
import pywt
import matplotlib
matplotlib.use('Agg')    # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
WAVELET   = 'db4'
N_LEVELS  = 4
RECON_TOL = 1e-10
IMG_SIZE  = 256
MODE      = 'periodization'   # exact power-of-2 subband sizes

WORKSPACE = Path(r'C:\Ocean Library\Matlab code practice')
IMG_PATH  = WORKSPACE / 'Data' / 'images' / 'elon_musk.jpg'
OUT_PATH  = WORKSPACE / 'Experiments' / 'Scripts' / 'Computer Vision' / \
            'exp_wavelet_decomposition_result.png'

# =============================================================================
# HELPERS  (defined first so they are available throughout)
# =============================================================================
def _p(flag: bool) -> str:
    return 'PASS' if flag else 'FAIL'


def normalise(X: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1] for display."""
    mn, mx = float(X.min()), float(X.max())
    return (X - mn) / (mx - mn + 1e-12)


def show_detail(X: np.ndarray) -> np.ndarray:
    """Absolute value + normalise — standard display for detail subbands."""
    return normalise(np.abs(X))


def build_mosaic(coeffs, full_size: int, N: int) -> np.ndarray:
    """
    Build a standard multi-level wavelet coefficient magnitude mosaic.

    Standard quadrant tiling (for N=3 on full_size=256):
        +---------+---------+-----------------+
        | LL_3    | LH_3    | LH_2            |
        +---------+---------+                 |
        | HL_3    | HH_3    |                 |
        +---------+---------+-----------------+
        | HL_2              | HH_2            |
    Build a standard multi-level wavelet coefficient mosaic (display-only).

    With mode='periodization' every subband is exactly full_size / 2^lv,
    so no resizing needed — subbands slot directly into power-of-2 quadrants.

    IMPORTANT: each subband is normalised independently to [0,1] so that
    detail coefficients (which hold <1% of energy) are still visible.

    pywt.wavedec2 ordering:
        coeffs[0]     = coarsest LL
        coeffs[1]     = (LH, HL, HH) at level N  (coarsest detail)
        ...
        coeffs[N]     = (LH, HL, HH) at level 1  (finest detail)
    """
    mosaic = np.zeros((full_size, full_size), dtype=np.float64)

    def norm(X):
        """Normalise a single subband to [0, 1]."""
        a = np.abs(X)
        mx = a.max()
        return a / (mx + 1e-12)

    # Coarsest LL — top-left
    slot_n = full_size >> N
    mosaic[0:slot_n, 0:slot_n] = norm(coeffs[0])

    # Detail subbands — coarsest (lv=N) to finest (lv=1)
    for lv in range(N, 0, -1):
        slot = full_size >> lv                 # exact subband edge length
        dH, dV, dD = coeffs[N + 1 - lv]       # list[1]=coarsest, list[N]=finest
        mosaic[0:slot,     slot:2*slot] = norm(dH)   # LH: top-right
        mosaic[slot:2*slot, 0:slot]     = norm(dV)   # HL: bottom-left
        mosaic[slot:2*slot, slot:2*slot] = norm(dD)  # HH: bottom-right

    return mosaic


# =============================================================================
# 2. LOAD IMAGE
# =============================================================================
print('=== EXPERIMENT: 2D Wavelet Decomposition (Szeliski Fig 3.36) ===\n')

if not IMG_PATH.is_file():
    sys.exit(f'ERROR: Image not found at {IMG_PATH}')

img_pil  = Image.open(IMG_PATH).convert('L')            # grayscale
img_pil  = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
I        = np.array(img_pil, dtype=np.float64) / 255.0  # [0, 1]

print(f'--- Image: Elon Musk, resized to {IMG_SIZE}x{IMG_SIZE} grayscale')
print(f'    Intensity range: [{I.min():.4f}, {I.max():.4f}]\n')

# =============================================================================
# 3. H1 + H2 — Single-Level DWT and Perfect Reconstruction
# =============================================================================
print(f'[H1/H2] Single-level 2D DWT  (wavelet={WAVELET}, mode={MODE})')

# pywt.dwt2 returns (cA, (cH, cV, cD))
#   cA = LL approximation         (low-low)
#   cH = LH horizontal detail    (highpass rows,  lowpass  cols -> horizontal edges)
#   cV = HL vertical detail      (lowpass  rows,  highpass cols -> vertical edges)
#   cD = HH diagonal detail      (highpass rows,  highpass cols -> diagonal / mixed)
cA, (cH, cV, cD) = pywt.dwt2(I, WAVELET, mode=MODE)

E_total = float(np.sum(I ** 2))
E_LL    = float(np.sum(cA ** 2))
E_LH    = float(np.sum(cH ** 2))
E_HL    = float(np.sum(cV ** 2))
E_HH    = float(np.sum(cD ** 2))

print(f'  Input size   : {I.shape}')
print(f'  Subband size : {cA.shape}')
print(f'\n  Subband energies (single level):')
print(f'    LL  approx      : {E_LL:.4f}  ({E_LL / E_total * 100:.1f}%)')
print(f'    LH  horiz edges : {E_LH:.4f}  ({E_LH / E_total * 100:.1f}%)')
print(f'    HL  vert  edges : {E_HL:.4f}  ({E_HL / E_total * 100:.1f}%)')
print(f'    HH  diagonal    : {E_HH:.4f}  ({E_HH / E_total * 100:.1f}%)')

h1_pass = all(x.shape == cA.shape for x in [cH, cV, cD])
print(f'\n  H1 result: {_p(h1_pass)}')

# Perfect reconstruction
I_recon   = pywt.idwt2((cA, (cH, cV, cD)), WAVELET, mode=MODE)
err_recon = float(np.max(np.abs(I_recon - I)))
h2_pass   = err_recon < RECON_TOL
print(f'  Reconstruction max|error| : {err_recon:.2e}  (tol={RECON_TOL:.0e})')
print(f'  H2 result: {_p(h2_pass)}')

# =============================================================================
# 4. H3 — Multi-Level: LL halves each level
# =============================================================================
print(f'\n[H3] Multi-level DWT ({N_LEVELS} levels, {WAVELET})')

# wavedec2 returns list: [LL_N, (cH_N,cV_N,cD_N), ..., (cH_1,cV_1,cD_1)]
#   index 0   = coarsest LL
#   index 1   = coarsest detail (level N)
#   index N   = finest   detail (level 1)
coeffs = pywt.wavedec2(I, WAVELET, level=N_LEVELS, mode=MODE)

h3_pass = True
prev_sz = IMG_SIZE
print(f'  Coarsest LL : {coeffs[0].shape}')
for lv in range(1, N_LEVELS + 1):
    dH, dV, dD = coeffs[lv]          # lv=1 -> coarsest, lv=N -> finest
    lv_sz       = dH.shape[0]
    # Map list index to pyramid level (index 1 = level N in spatial frequency)
    pyramid_lv  = N_LEVELS + 1 - lv
    expected_sz = IMG_SIZE >> pyramid_lv
    ok          = lv_sz == expected_sz
    print(f'  Pyramid Level {pyramid_lv} detail : {dH.shape}  '
          f'(expected {expected_sz}x{expected_sz})  {_p(ok)}')
    h3_pass = h3_pass and ok
print(f'  H3 result: {_p(h3_pass)}')

# =============================================================================
# 5. H4 — Energy Compaction
# =============================================================================
print(f'\n[H4] Energy compaction')

frac_ll = E_LL / E_total
h4_pass = frac_ll > 0.85
print(f'  Total image energy : {E_total:.4f}')
print(f'  LL energy fraction : {frac_ll * 100:.1f}%  (threshold: >85%)')
print(f'  H4 result: {_p(h4_pass)}')

# =============================================================================
# 6. VISUALISATION
# =============================================================================
BG   = '#0d1117'
DARK = '#161b22'
TXT  = '#e6edf3'
ACC1 = '#3399ff'

plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 14), facecolor=BG)
fig.suptitle(
    'Wavelet Decomposition — Elon Musk  [Szeliski 2021, Fig 3.36]\n'
    f'Wavelet: {WAVELET}   Levels: {N_LEVELS}   Mode: {MODE}',
    color=TXT, fontsize=14, fontweight='bold', y=0.99)

# Top block: 2 rows x 5 cols for subbands (rows 1-2 of the figure)
# Bottom block: 1 row for mosaic, spans full width
gs_top = gridspec.GridSpec(2, 5, figure=fig,
                           hspace=0.38, wspace=0.08,
                           left=0.03, right=0.97,
                           top=0.90, bottom=0.42)

gs_bot = gridspec.GridSpec(1, 2, figure=fig,
                           wspace=0.08,
                           left=0.03, right=0.97,
                           top=0.38, bottom=0.04)


def _show(ax, img, title, accent=False):
    ax.imshow(img, cmap='gray', aspect='equal',
              interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(title, color=ACC1 if accent else TXT,
                 fontsize=8.5, pad=4, fontweight='bold')
    ax.axis('off')
    ax.set_facecolor(DARK)


# --- Row 1: Single-level subbands (Fig 3.36a) ---
row1 = [
    (I,               'Original\n256x256',         True),
    (normalise(cA),   'LL  (Approx)',               False),
    (show_detail(cH), 'LH  Horizontal Edges',       False),
    (show_detail(cV), 'HL  Vertical Edges',          False),
    (show_detail(cD), 'HH  Diagonal',               False),
]
for k, (img, ttl, acc) in enumerate(row1):
    _show(fig.add_subplot(gs_top[0, k]), img, ttl, accent=acc)

# --- Row 2: LH at each pyramid level + coarsest LL ---
for idx in range(N_LEVELS):
    pyramid_lv  = N_LEVELS - idx          # 4, 3, 2, 1
    list_idx    = N_LEVELS + 1 - pyramid_lv  # 1, 2, 3, 4
    dH          = coeffs[list_idx][0]
    _show(fig.add_subplot(gs_top[1, idx]), show_detail(dH),
          f'LH Level {pyramid_lv}\n{dH.shape[0]}x{dH.shape[1]}')

cA_deep = coeffs[0]
_show(fig.add_subplot(gs_top[1, 4]), normalise(cA_deep),
      f'LL coarsest (Lv{N_LEVELS})\n{cA_deep.shape[0]}x{cA_deep.shape[1]}')

# --- Bottom left: mosaic (Fig 3.36b) ---
ax_m   = fig.add_subplot(gs_bot[0, 0])
mosaic = build_mosaic(coeffs, IMG_SIZE, N_LEVELS)
ax_m.imshow(mosaic, cmap='gray', aspect='equal',
            interpolation='nearest', vmin=0, vmax=1)
ax_m.set_title(
    f'Multi-Level Coefficient Mosaic — {N_LEVELS} levels, {WAVELET}\n'
    f'[cf. Szeliski Fig 3.36b]  (each subband normalised independently)',
    color=TXT, fontsize=10, fontweight='bold', pad=6)
ax_m.axis('off')
ax_m.set_facecolor(DARK)
ax_m.text(0.01, 0.02,
    f'H1:{_p(h1_pass)}  H2:{_p(h2_pass)}  H3:{_p(h3_pass)}  H4:{_p(h4_pass)}'
    f'   Recon err: {err_recon:.1e}   LL energy: {frac_ll*100:.1f}%',
    transform=ax_m.transAxes, color=ACC1, fontsize=8.5, va='bottom',
    family='monospace',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117', alpha=0.85))

# --- Bottom right: reconstructed image ---
ax_r = fig.add_subplot(gs_bot[0, 1])
ax_r.imshow(normalise(I_recon), cmap='gray', aspect='equal',
            interpolation='nearest', vmin=0, vmax=1)
ax_r.set_title(
    f'Reconstructed Image (idwt2)\nmax|error| = {err_recon:.1e}',
    color=TXT, fontsize=10, fontweight='bold', pad=6)
ax_r.axis('off')
ax_r.set_facecolor(DARK)



plt.savefig(OUT_PATH, dpi=150, facecolor=BG, bbox_inches='tight')
print(f'\n--- Figure saved: {OUT_PATH}')

# =============================================================================
# 7. SUMMARY
# =============================================================================
print('\n=== EXPERIMENT SUMMARY ===')
print(f'  Image    : Elon Musk {IMG_SIZE}x{IMG_SIZE}')
print(f'  Wavelet  : {WAVELET}')
print(f'  Levels   : {N_LEVELS}')
print(f'\n  Hypothesis Results:')
print(f'    H1 - 4-subband single-level DWT : {_p(h1_pass)}')
print(f'    H2 - Perfect reconstruction     : {_p(h2_pass)}  (err={err_recon:.2e})')
print(f'    H3 - LL halves each level        : {_p(h3_pass)}')
print(f'    H4 - Energy compaction >85%     : {_p(h4_pass)}  ({frac_ll*100:.1f}% in LL)')
all_pass = h1_pass and h2_pass and h3_pass and h4_pass
print(f'\n  Overall: {"ALL HYPOTHESES PASSED" if all_pass else "ONE OR MORE HYPOTHESES FAILED"}')
print('==========================')
