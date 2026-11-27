"""
exp_pca_digits.py
=================
PCA Handwritten Digits Experiment -- Production Python Edition

Replicates ESL Section 14.5.1 (Figures 14.22-14.24):
Principal Components applied to handwritten digits.

Uses PCA (via SVD) to perform dimension reduction and compression on 16x16
grayscale images of the digit "3", and visualizes the principal components.

DATASET:        USPS Handwritten Digits (fetch_openml)
PREPROCESSING:  Mean-centering
MODEL:          Principal Component Analysis (PCA)

Ref: Hastie et al. (2009), ESL Section 14.5.1, Figures 14.22-14.24.

Usage:
    python exp_pca_digits.py
"""

# %% ============================================================
#  IMPORTS & CONFIGURATION
# ============================================================
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "pca_digits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
RANDOM_SEED = 42

# Plotly theme (keeping consistent with our library style)
PLOTLY_TEMPLATE = "plotly_dark"
BG_COLOR        = "#1a1a2e"
PAPER_COLOR     = "#16213e"
TEXT_COLOR      = "#e0e0e0"

print("=" * 65)
print("  PCA HANDWRITTEN DIGITS EXPERIMENT -- Python Production")
print("=" * 65)


# %% ============================================================
#  PHASE 1: DATA ACQUISITION
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 1: DATA ACQUISITION")
print(f"{'=' * 65}")

t0 = time.time()
print("  Fetching USPS dataset from OpenML...")
# parser='liac-arff' avoids pandas dependency warnings
usps = fetch_openml('usps', version=2, as_frame=False, parser='liac-arff')

# Filter for the digit "3"
mask_threes = (usps.target == '3')
X_raw = usps.data[mask_threes]

# OpenML USPS data is flat (256,), values usually in [-1, 1] or [0, 2]
# We'll normalize to [0, 1] for easier plotting
X_raw = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())

N, p = X_raw.shape
# The images are 16x16
IMG_SHAPE = (16, 16)

t_data = time.time() - t0
print(f"  Dataset loaded in {t_data:.1f}s")
print(f"  Total '3's found: {N}")
print(f"  Features:         {p} ({IMG_SHAPE[0]}x{IMG_SHAPE[1]} pixels)")


# %% ============================================================
#  PHASE 2: FEATURE ENGINEERING
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 2: FEATURE ENGINEERING")
print(f"{'=' * 65}")

# Mean centering
X_mean = np.mean(X_raw, axis=0)
X_centered = X_raw - X_mean

print("  Mean image computed and subtracted from all samples.")


# %% ============================================================
#  PHASE 3: TRAIN / TEST SPLIT (N/A)
# ============================================================


# %% ============================================================
#  PHASE 4 & 5: MODEL CONFIGURATION & TRAINING
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 4 & 5: MODEL TRAINING")
print(f"{'=' * 65}")

# Fit PCA on real data
pca = PCA(n_components=p, random_state=RANDOM_SEED)
pca.fit(X_centered)
scores = pca.transform(X_centered)  # u_ij

# For ESL Fig 14.24: We need a randomized baseline
# Randomly scramble each column of X_centered independently
rng = np.random.RandomState(RANDOM_SEED)
X_rand = np.zeros_like(X_centered)
for j in range(p):
    X_rand[:, j] = rng.permutation(X_centered[:, j])

pca_rand = PCA(n_components=p, random_state=RANDOM_SEED)
pca_rand.fit(X_rand)

print("  PCA fitted on real data and randomized baseline.")


# %% ============================================================
#  PHASE 6: EVALUATION
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 6: EVALUATION")
print(f"{'=' * 65}")

var_real = pca.explained_variance_ratio_
cum_var_real = np.cumsum(var_real)

# ESL states ~50 components account for 90%, 12 account for 63%
idx_90 = np.argmax(cum_var_real >= 0.90) + 1
idx_63 = np.argmax(cum_var_real >= 0.63) + 1

print(f"  Variance explained by 12 components:  {cum_var_real[11]*100:.1f}% (ESL ~63%)")
print(f"  Variance explained by 50 components:  {cum_var_real[49]*100:.1f}% (ESL ~90%)")
print(f"  Components for 90% variance:          {idx_90}")


# %% ============================================================
#  PHASE 7: VISUALIZATION
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 7: VISUALIZATION")
print(f"{'=' * 65}")

def stitch_images(images, rows, cols, shape=(16,16), invert=True):
    """Helper to stitch multiple small images into a large grid image."""
    h, w = shape
    stitched = np.zeros((rows * h, cols * w))
    for idx, img in enumerate(images[:rows*cols]):
        r, c = divmod(idx, cols)
        stitched[r*h:(r+1)*h, c*w:(c+1)*w] = img.reshape(shape)
    if invert:
        # Invert colors so digits are dark on light background (like ESL)
        stitched = 1.0 - stitched
    return stitched


# -----------------------------------------------------------------------
# Figure 1: Sample of 130 Handwritten 3's (ESL Fig 14.22)
# -----------------------------------------------------------------------
print("  Creating Figure 1: Sample of 130 Digits...")
# We use a 10x13 grid to show 130 digits
sample_img = stitch_images(X_raw[:130], rows=10, cols=13)

fig1 = go.Figure(data=go.Heatmap(
    z=np.flipud(sample_img),  # Flipud because plotly y-axis goes up from bottom
    colorscale="Greys", showscale=False
))
# Draw green grid lines to separate the digits
h, w = IMG_SHAPE
for r in range(11):
    fig1.add_hline(y=r*h, line=dict(color="green", width=1))
for c in range(14):
    fig1.add_vline(x=c*w, line=dict(color="green", width=1))

fig1.update_layout(
    title="Sample of 130 Handwritten 3's (ESL Fig 14.22)",
    template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    width=800, height=600,
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="y"),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
)

fig1_path = os.path.join(OUTPUT_DIR, "fig1_digits_sample.html")
fig1.write_html(fig1_path)
print(f"    Saved: {fig1_path}")


# -----------------------------------------------------------------------
# Figure 2 & 3 Setup: PC1 vs PC2 and Representative Images
# -----------------------------------------------------------------------
print("  Creating Figure 2 & 3: PC Quantiles and Grid Images...")
pc1 = scores[:, 0]
pc2 = scores[:, 1]

# 5%, 25%, 50%, 75%, 95% quantiles
q_vals = [5, 25, 50, 75, 95]
q_pc1 = np.percentile(pc1, q_vals)
q_pc2 = np.percentile(pc2, q_vals)

# Find closest images to the 25 intersections
grid_images = []
highlight_x = []
highlight_y = []

# To match the grid layout visually, we sort rows top to bottom (highest PC2 to lowest PC2)
for qy in reversed(q_pc2):
    for qx in q_pc1:
        # Distance from point to this intersection
        dists = (pc1 - qx)**2 + (pc2 - qy)**2
        closest_idx = np.argmin(dists)
        highlight_x.append(pc1[closest_idx])
        highlight_y.append(pc2[closest_idx])
        grid_images.append(X_raw[closest_idx])


# -----------------------------------------------------------------------
# Figure 2: PC1 vs PC2 Scatter Plot (ESL Fig 14.23 Left)
# -----------------------------------------------------------------------
fig2 = go.Figure()

# Background points
fig2.add_trace(go.Scatter(
    x=pc1, y=pc2, mode='markers',
    marker=dict(size=3, color='#1abc9c'),
    name="All Digits"
))

# Highlighted points closest to grid
fig2.add_trace(go.Scatter(
    x=highlight_x, y=highlight_y, mode='markers',
    marker=dict(size=8, color='rgba(0,0,0,0)', line=dict(color='red', width=2)),
    name="Grid Representatives"
))

# Draw quantile grid lines
for qx in q_pc1:
    fig2.add_vline(x=qx, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"))
for qy in q_pc2:
    fig2.add_hline(y=qy, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"))
fig2.add_vline(x=0, line=dict(color="rgba(255,255,255,0.8)", width=1))
fig2.add_hline(y=0, line=dict(color="rgba(255,255,255,0.8)", width=1))

fig2.update_layout(
    title="First Two Principal Components (ESL Fig 14.23 Left)",
    template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="First Principal Component",
    yaxis_title="Second Principal Component",
    width=600, height=600,
    showlegend=False
)

fig2_path = os.path.join(OUTPUT_DIR, "fig2_pc1_vs_pc2.html")
fig2.write_html(fig2_path)
print(f"    Saved: {fig2_path}")


# -----------------------------------------------------------------------
# Figure 3: Grid of Representative Images (ESL Fig 14.23 Right)
# -----------------------------------------------------------------------
grid_stitched = stitch_images(grid_images, rows=5, cols=5)

fig3 = go.Figure(data=go.Heatmap(
    z=np.flipud(grid_stitched),
    colorscale="Greys", showscale=False
))
# Red grid lines
for r in range(6):
    fig3.add_hline(y=r*h, line=dict(color="rgba(255,0,0,0.5)", width=1))
for c in range(6):
    fig3.add_vline(x=c*w, line=dict(color="rgba(255,0,0,0.5)", width=1))

fig3.update_layout(
    title="Grid Images (ESL Fig 14.23 Right)",
    template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    width=500, height=500,
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="y"),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
)

fig3_path = os.path.join(OUTPUT_DIR, "fig3_quantile_grid_images.html")
fig3.write_html(fig3_path)
print(f"    Saved: {fig3_path}")


# -----------------------------------------------------------------------
# Figure 4: PC Directions as Images (Equation 14.55)
# -----------------------------------------------------------------------
print("  Creating Figure 4: PC Directions...")
# PC components are in pca.components_ (p, p). We want v1, v2
v1 = pca.components_[0]
v2 = pca.components_[1]

# To make them visible as images, we scale them
v1_scaled = (v1 - v1.min()) / (v1.max() - v1.min())
v2_scaled = (v2 - v2.min()) / (v2.max() - v2.min())

# We will just plot mean, v1, and v2 side by side
directions_img = stitch_images([X_mean, v1_scaled, v2_scaled], rows=1, cols=3)

fig4 = go.Figure(data=go.Heatmap(
    z=np.flipud(directions_img),
    colorscale="Greys", showscale=False
))
# Red boxes around them
for r in range(2):
    fig4.add_hline(y=r*h, line=dict(color="red", width=1))
for c in range(4):
    fig4.add_vline(x=c*w, line=dict(color="red", width=1))

fig4.update_layout(
    title="PC Directions: Mean, V₁, V₂ (Eq 14.55)",
    template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    width=500, height=250,
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="y"),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
)

fig4_path = os.path.join(OUTPUT_DIR, "fig4_pc_directions.html")
fig4.write_html(fig4_path)
print(f"    Saved: {fig4_path}")


# -----------------------------------------------------------------------
# Figure 5: Scree Plot (ESL Fig 14.24)
# -----------------------------------------------------------------------
print("  Creating Figure 5: Scree Plot...")
# Singular values are pca.singular_values_
sv_real = pca.singular_values_
sv_rand = pca_rand.singular_values_

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=np.arange(1, p+1), y=sv_real,
    mode='markers', marker=dict(size=4, color="#3498db"),
    name="Real Trace"
))
fig5.add_trace(go.Scatter(
    x=np.arange(1, p+1), y=sv_rand,
    mode='markers', marker=dict(size=4, color="#e67e22"),
    name="Randomized Trace"
))

fig5.update_layout(
    title="Singular Values (ESL Fig 14.24)",
    template=PLOTLY_TEMPLATE, paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="Dimension", yaxis_title="Singular Values",
    width=700, height=500,
    legend=dict(x=0.6, y=0.8, bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1)
)

fig5_path = os.path.join(OUTPUT_DIR, "fig5_scree_plot.html")
fig5.write_html(fig5_path)
print(f"    Saved: {fig5_path}")

print(f"\n{'=' * 65}")
print("  PIPELINE COMPLETE")
print(f"{'=' * 65}\n")
