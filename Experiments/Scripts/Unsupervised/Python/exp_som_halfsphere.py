"""
exp_som_halfsphere.py
=====================
SOM Half-Sphere Experiment — Production Python Edition

Replicates ESL Section 14.4 (Figures 14.15–14.18).
Uses MiniSom for training, scikit-learn for K-Means baseline,
and Plotly for interactive 3D visualization.

DATASET:        Synthetic, N=90, p=3 (surface of half unit sphere)
                Three clusters: Red (0,1,0), Green (0,0,1), Blue (1,0,0)
                Red sigma=0.1 (tight), Green/Blue sigma=0.5 (wide)
MODEL:          Self-Organizing Map (Kohonen, 1982)
                5x5 grid = 25 prototypes
                Gaussian neighborhood kernel
ASSUMPTIONS:    - Points generated from isotropic Gaussian then projected
                  onto unit sphere and clipped to z >= 0 (half-sphere).
                - Sigma values from ESL Exercise 14.5.
                - SOM hyperparameters match ESL Section 14.4 exactly:
                  R=2, 40 passes, linear decay of alpha and r.
REPRODUCIBILITY: np.random.seed(42)

Ref: Hastie et al. (2009), ESL Section 14.4, Figures 14.15–14.18.

Usage:
    python exp_som_halfsphere.py
"""

# %% ============================================================
#  IMPORTS & CONFIGURATION
# ============================================================
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory — same folder as this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ESL Section 14.4 hyperparameters
GRID_SIZE      = (5, 5)       # Q1 x Q2
INIT_SIGMA     = 2.0          # Initial neighborhood radius
LEARNING_RATE  = 0.5          # Initial learning rate
NUM_PASSES     = 40           # Full passes through the dataset
N_PER_CLASS    = 30           # Points per cluster
K_CLASSES      = 3            # Number of clusters
RANDOM_SEED    = 42

# Plotly theme
PLOTLY_TEMPLATE = "plotly_dark"
BG_COLOR        = "#1a1a2e"
PAPER_COLOR     = "#16213e"
TEXT_COLOR      = "#e0e0e0"
GRID_LINE_COLOR = "rgba(150,150,150,0.85)"
NODE_COLOR      = "rgb(140, 50, 210)"
PALETTE         = ["#e74c3c", "#2ecc71", "#3498db"]   # Red, Green, Blue clusters

print("=" * 65)
print("  SOM HALF-SPHERE EXPERIMENT -- Python Production")
print(f"  Grid:          {GRID_SIZE[0]}x{GRID_SIZE[1]} = {GRID_SIZE[0]*GRID_SIZE[1]} prototypes")
print(f"  Passes:        {NUM_PASSES}")
print(f"  Init sigma:    {INIT_SIGMA}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Seed:          {RANDOM_SEED}")
print("=" * 65)


# %% ============================================================
#  1. DATA GENERATION
#  Generate 90 data points on the surface of a half-sphere (z >= 0)
#  Three clusters centered near (0,1,0), (0,0,1), (1,0,0)
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 1: DATA GENERATION")
print(f"{'=' * 65}")

np.random.seed(RANDOM_SEED)

centers = np.array([
    [0, 1, 0],    # Red cluster center
    [0, 0, 1],    # Green cluster center
    [1, 0, 0],    # Blue cluster center
])
sigmas = np.array([0.1, 0.5, 0.5])   # Red tight, Green/Blue wide

X = np.zeros((N_PER_CLASS * K_CLASSES, 3)) #90 x 3
Y = np.zeros(N_PER_CLASS * K_CLASSES, dtype=int) # 90 x 1

for k in range(K_CLASSES):
    start = k * N_PER_CLASS
    end   = (k + 1) * N_PER_CLASS

    # Sample from N(center, sigma^2 * I_3)
    raw = centers[k] + sigmas[k] * np.random.randn(N_PER_CLASS, 3)

    # Project onto unit sphere: x <- x / ||x||
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    raw = raw / norms

    # Half-sphere constraint: z >= 0 (flip z if negative)
    raw[raw[:, 2] < 0, 2] *= -1

    X[start:end] = raw
    Y[start:end] = k

N, p = X.shape
class_names = ["Red (tight)", "Green (wide)", "Blue (wide)"]

print(f"  Generated {N} points in {K_CLASSES} classes on a half-sphere.")
for k in range(K_CLASSES):
    print(f"    {class_names[k]}: center={tuple(centers[k])}, sigma={sigmas[k]}")


# %% ============================================================
#  2. FEATURE ENGINEERING
#  None required — raw 3D coordinates on the half-sphere surface.
# ============================================================

# %% ============================================================
#  3. TRAIN / TEST SPLIT
#  N/A — unsupervised learning experiment, full dataset for SOM.
# ============================================================

# %% ============================================================
#  4. MODEL CONFIGURATION
#  MiniSom with linear decay to match our MATLAB implementation.
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 4: MODEL CONFIGURATION")
print(f"{'=' * 65}")

som = MiniSom(
    x=GRID_SIZE[0],
    y=GRID_SIZE[1],
    input_len=p,
    sigma=INIT_SIGMA,
    learning_rate=LEARNING_RATE,
    decay_function="linear_decay_to_zero",
    sigma_decay_function="linear_decay_to_one",
    neighborhood_function="gaussian",
    topology="rectangular",
    random_seed=RANDOM_SEED,
)

# Initialize with random samples from data
som.random_weights_init(X)

print(f"  MiniSom initialized: {GRID_SIZE[0]}x{GRID_SIZE[1]} grid, p={p}")
print(f"  Decay:     linear (lr -> 0, sigma -> 1)")
print(f"  Topology:  rectangular")
print(f"  Distance:  euclidean")


# %% ============================================================
#  5. TRAINING
#  Online mode with per-pass history tracking.
#  MiniSom's train_random picks random samples per iteration.
#  We call it N iterations at a time to snapshot weights per pass.
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 5: TRAINING")
print(f"{'=' * 65}")

t_train = time.time()

total_iterations = NUM_PASSES * N

# --- History tracking ---
# Snapshot weights at initial state + after each pass
history = [som.get_weights().copy()]
error_history = []

# Initial reconstruction error
def compute_recon_error(som_obj, data):
    """Sum of squared distances from each point to its BMU."""
    err = 0.0
    for xi in data:
        w = som_obj.winner(xi) # Find BMU
        err += np.sum((xi - som_obj.get_weights()[w]) ** 2)
    return err

error_history.append(compute_recon_error(som, X))

# --- Manual training loop with global decay counter ---
# MiniSom's train_random() resets its internal decay schedule on each call,
# so calling it 40 times with N iterations each would restart lr/sigma decay
# per pass instead of decaying smoothly over all 3600 iterations.
# Instead, we drive the loop manually using som.update(x, win, t, max_iter)
# with a global iteration counter — identical to our MATLAB implementation.
rng_train = np.random.RandomState(RANDOM_SEED)
iter_count = 0

for pass_idx in range(NUM_PASSES):
    perm = rng_train.permutation(N)    # shuffle order each pass
    for s in range(N):
        xi = X[perm[s]]
        winner = som.winner(xi)
        som.update(xi, winner, iter_count, total_iterations)
        iter_count += 1

    history.append(som.get_weights().copy())
    error_history.append(compute_recon_error(som, X))

    if (pass_idx + 1) % 10 == 0 or pass_idx == 0:
        mqe_now = som.quantization_error(X)
        print(f"  [SOM-online] Pass {pass_idx+1:3d}/{NUM_PASSES} | MQE={mqe_now:.4f}")

t_train = time.time() - t_train
print(f"\n  Training time: {t_train:.2f}s ({total_iterations} iterations)")

# Initial weights for before/after comparison
initial_weights = history[0]


# %% ============================================================
#  6. EVALUATION
#  Compute all metrics from scratch — never hard-code expected values.
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 6: EVALUATION")
print(f"{'=' * 65}")

# MQE: average Euclidean distance to BMU
mqe = som.quantization_error(X)

# Topographic Error: fraction where 1st & 2nd BMU are not adjacent
te = som.topographic_error(X)

# U-Matrix: built into MiniSom
U = som.distance_map()

print(f"  MQE (Mean Quantization Error) : {mqe:.4f}")
print(f"  Topographic Error             : {te:.4f} ({te*100:.1f}%)")


# %% ============================================================
#  7. VISUALIZATION
#  5 interactive Plotly figures, saved as HTML.
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 7: VISUALIZATION")
print(f"{'=' * 65}")


# -----------------------------------------------------------------------
# Helper: generate half-sphere mesh for 3D backgrounds
# -----------------------------------------------------------------------
def make_halfsphere_mesh(n=40):
    """Generate a translucent half-sphere surface for Plotly."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi / 2, n)   # only upper hemisphere
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    return sx, sy, sz


# -----------------------------------------------------------------------
# Figure 1: 3D Half-Sphere Data (ESL Figure 14.15)
# -----------------------------------------------------------------------
print("  Creating Figure 1: 3D Half-Sphere Data...")

fig1 = go.Figure()

# Translucent half-sphere surface
sx, sy, sz = make_halfsphere_mesh()
fig1.add_trace(go.Surface(
    x=sx, y=sy, z=sz,
    opacity=0.12,
    colorscale=[[0, "rgba(230,210,80,0.3)"], [1, "rgba(230,210,80,0.3)"]],
    showscale=False,
    hoverinfo="skip",
    name="Half-sphere",
))

# Data points by class
for k in range(K_CLASSES):
    mask = Y == k
    fig1.add_trace(go.Scatter3d(
        x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
        mode="markers",
        marker=dict(size=5, color=PALETTE[k], opacity=0.85,
                    line=dict(width=0.5, color="rgba(0,0,0,0.4)")),
        name=class_names[k],
    ))

# Cluster centers
fig1.add_trace(go.Scatter3d(
    x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
    mode="markers",
    marker=dict(size=8, color="gold", symbol="diamond",
                line=dict(width=1.5, color="black")),
    name="True centers",
))

fig1.update_layout(
    title="Half-Sphere Data — 3 Clusters (ESL Fig 14.15)",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    scene=dict(
        xaxis_title="X₁", yaxis_title="X₂", zaxis_title="X₃",
        camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
    ),
    width=900, height=700,
    margin=dict(l=20, r=20, t=60, b=20),
)

fig1_path = os.path.join(OUTPUT_DIR, "fig1_halfsphere_data.html")
fig1.write_html(fig1_path)
print(f"    Saved: {fig1_path}")


# -----------------------------------------------------------------------
# Figure 2: SOM Grid — Initial vs Final (ESL Figure 14.16)
# -----------------------------------------------------------------------
print("  Creating Figure 2: SOM Grid — Initial vs Final...")

def assign_labels(weights, data):
    """Assign each data point to its BMU given a weight matrix."""
    labels = []
    for xi in data:
        dists = np.sum((weights.reshape(-1, weights.shape[-1]) - xi) ** 2, axis=1) #axes=1, sum horizontal across cols
        labels.append(np.argmin(dists))
    return np.array(labels)

def create_som_grid_subplot(weights, data, class_labels, grid_size, title_text):
    """Create a jittered SOM grid plot as a list of Plotly traces."""
    Q1, Q2 = grid_size
    M = Q1 * Q2
    flat_w = weights.reshape(M, -1) #-1 for dynamic sizing
    labels = assign_labels(weights, data)

    traces = []
    rng = np.random.RandomState(123)

    # Draw circles for each prototype
    theta = np.linspace(0, 2 * np.pi, 40)
    radius = 0.35
    for j in range(M):
        gy, gx = divmod(j, Q1)
        gx_pos, gy_pos = gy + 1, gx + 1  # 1-indexed
        cx = radius * np.cos(theta) + gx_pos
        cy = radius * np.sin(theta) + gy_pos
        traces.append(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color="rgba(150,150,150,0.5)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Jitter data points inside their prototype circle
    for k in range(K_CLASSES):
        mask_k = class_labels == k
        pts_in_class = np.where(mask_k)[0]
        jx, jy = [], []
        for idx in pts_in_class:
            bmu = labels[idx]
            gy, gx = divmod(bmu, Q1)
            gx_pos, gy_pos = gy + 1, gx + 1
            r = radius * 0.8 * np.sqrt(rng.rand())
            a = rng.rand() * 2 * np.pi
            jx.append(gx_pos + r * np.cos(a))
            jy.append(gy_pos + r * np.sin(a))

        traces.append(go.Scatter(
            x=jx, y=jy, mode="markers",
            marker=dict(size=6, color=PALETTE[k], opacity=0.85),
            name=class_names[k],
            showlegend=False,
        ))

    return traces

fig2 = make_subplots(rows=1, cols=2,
                     subplot_titles=["Initial Configuration", "Final Configuration"],
                     horizontal_spacing=0.08)

# Initial
init_traces = create_som_grid_subplot(initial_weights, X, Y, GRID_SIZE, "Initial")
for tr in init_traces:
    fig2.add_trace(tr, row=1, col=1)

# Final
final_traces = create_som_grid_subplot(som.get_weights(), X, Y, GRID_SIZE, "Final")
for tr in final_traces:
    fig2.add_trace(tr, row=1, col=2)

fig2.update_layout(
    title="Self-Organizing Map — 5×5 Grid (ESL Fig 14.16)",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    width=1100, height=550,
    margin=dict(l=40, r=40, t=80, b=40),
)
for col in [1, 2]:
    fig2.update_xaxes(range=[0.2, GRID_SIZE[1] + 0.8], row=1, col=col,
                      showgrid=False, zeroline=False)
    fig2.update_yaxes(range=[0.2, GRID_SIZE[0] + 0.8], row=1, col=col,
                      scaleanchor=f"x{col}", showgrid=False, zeroline=False)

fig2_path = os.path.join(OUTPUT_DIR, "fig2_som_grid.html")
fig2.write_html(fig2_path)
print(f"    Saved: {fig2_path}")


# -----------------------------------------------------------------------
# Figure 3: U-Matrix Heatmap
# -----------------------------------------------------------------------
print("  Creating Figure 3: U-Matrix...")

fig3 = go.Figure(data=go.Heatmap(
    z=U,
    colorscale="Hot",
    colorbar=dict(title="Mean Neighbor<br>Distance", title_font=dict(color=TEXT_COLOR),
                  tickfont=dict(color=TEXT_COLOR)),
    hovertemplate="Row %{y}, Col %{x}<br>Distance: %{z:.4f}<extra></extra>",
))

fig3.update_layout(
    title="U-Matrix — Cluster Boundaries",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="Grid Column",
    yaxis_title="Grid Row",
    width=650, height=580,
    margin=dict(l=60, r=20, t=60, b=60),
    yaxis=dict(scaleanchor="x"),
)

fig3_path = os.path.join(OUTPUT_DIR, "fig3_umatrix.html")
fig3.write_html(fig3_path)
print(f"    Saved: {fig3_path}")


# -----------------------------------------------------------------------
# Figure 4: Animated 3D Wiremesh (ESL Figure 14.17)
#   Plotly frames-based animation with play/pause slider.
# -----------------------------------------------------------------------
print("  Creating Figure 4: Animated 3D Wiremesh...")

Q1, Q2 = GRID_SIZE
M = Q1 * Q2

# Build grid edge list: horizontal + vertical adjacency
edges = []
for i in range(Q1):
    for j in range(Q2):
        idx = j * Q1 + i    # linear index matching ndgrid layout
        if j < Q2 - 1:      # horizontal neighbor
            edges.append((idx, (j + 1) * Q1 + i))
        if i < Q1 - 1:      # vertical neighbor
            edges.append((idx, j * Q1 + (i + 1)))


def build_wiremesh_traces(W_snapshot, show_data=True):
    """Build Plotly traces for one wiremesh frame."""
    W = W_snapshot.reshape(M, -1)
    traces = []

    # Edge lines (drawn as one trace with None separators for performance)
    edge_x, edge_y, edge_z = [], [], []
    for i1, i2 in edges:
        edge_x += [W[i1, 0], W[i2, 0], None]
        edge_y += [W[i1, 1], W[i2, 1], None]
        edge_z += [W[i1, 2], W[i2, 2], None]

    traces.append(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color=GRID_LINE_COLOR, width=3),
        hoverinfo="skip",
        showlegend=False,
        name="edges",
    ))

    # Node markers
    traces.append(go.Scatter3d(
        x=W[:, 0], y=W[:, 1], z=W[:, 2],
        mode="markers",
        marker=dict(size=5, color=NODE_COLOR, opacity=0.9,
                    line=dict(width=1, color="rgba(80,20,140,0.7)")),
        hovertemplate="Node %{pointNumber}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
        showlegend=False,
        name="nodes",
    ))

    return traces


# Background: data points + half-sphere (static, not in frames)
fig4 = go.Figure()

sx, sy, sz = make_halfsphere_mesh()
fig4.add_trace(go.Surface(
    x=sx, y=sy, z=sz,
    opacity=0.08,
    colorscale=[[0, "rgba(230,210,80,0.2)"], [1, "rgba(230,210,80,0.2)"]],
    showscale=False, hoverinfo="skip",
))

for k in range(K_CLASSES):
    mask = Y == k
    fig4.add_trace(go.Scatter3d(
        x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
        mode="markers",
        marker=dict(size=3, color=PALETTE[k], opacity=0.5),
        showlegend=False, hoverinfo="skip",
    ))

# Initial wiremesh (trace indices 4 and 5: edges + nodes)
init_traces = build_wiremesh_traces(history[0])
for tr in init_traces:
    fig4.add_trace(tr)

# Build animation frames
n_bg_traces = 1 + K_CLASSES   # surface + 3 scatter
frames = []
for frame_idx, W_snap in enumerate(history):
    frame_traces = build_wiremesh_traces(W_snap)
    frames.append(go.Frame(
        data=frame_traces,
        name=str(frame_idx),
        traces=[n_bg_traces, n_bg_traces + 1],   # update only edges + nodes
    ))

fig4.frames = frames

# Slider steps
slider_steps = []
for i in range(len(history)):
    label = "Initial" if i == 0 else f"Pass {i}"
    slider_steps.append(dict(
        args=[[str(i)], dict(frame=dict(duration=150, redraw=True), mode="immediate")],
        label=label,
        method="animate",
    ))

fig4.update_layout(
    title="SOM Wiremesh — Pass 0 (Initial) — ESL Fig 14.17",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    scene=dict(
        xaxis_title="X₁", yaxis_title="X₂", zaxis_title="X₃",
        camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        xaxis=dict(range=[-1.3, 1.3]),
        yaxis=dict(range=[-1.3, 1.3]),
        zaxis=dict(range=[-0.3, 1.5]),
    ),
    width=950, height=750,
    margin=dict(l=20, r=20, t=60, b=100),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.1, y=-0.02,
        xanchor="right", yanchor="top",
        buttons=[
            dict(label="▶ Play",
                 method="animate",
                 args=[None, dict(frame=dict(duration=150, redraw=True),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="⏸ Pause",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate")]),
        ],
    )],
    sliders=[dict(
        active=0,
        steps=slider_steps,
        x=0.15, len=0.75,
        xanchor="left",
        y=-0.02,
        currentvalue=dict(prefix="Pass: ", font=dict(size=13, color=TEXT_COLOR)),
        font=dict(color=TEXT_COLOR),
        tickcolor=TEXT_COLOR,
    )],
)

fig4_path = os.path.join(OUTPUT_DIR, "fig4_wiremesh_animation.html")
fig4.write_html(fig4_path)
print(f"    Saved: {fig4_path}")


# -----------------------------------------------------------------------
# Figure 5: Reconstruction Error vs Iteration (ESL Figure 14.18)
#   Sum of squared distances to BMU per pass + K-Means baseline.
# -----------------------------------------------------------------------
print("  Creating Figure 5: Reconstruction Error...")

# K-Means baseline: K = 25 (same as number of prototypes)
km = KMeans(n_clusters=M, random_state=RANDOM_SEED, n_init=10).fit(X)
km_error = km.inertia_   # sum of squared distances to cluster center

iter_axis = np.arange(len(error_history)) * N

fig5 = go.Figure()

# SOM reconstruction error curve
fig5.add_trace(go.Scatter(
    x=iter_axis, y=error_history,
    mode="lines+markers",
    marker=dict(size=6, color="#2ecc71"),
    line=dict(width=2, color="#2ecc71"),
    name="SOM Reconstruction Error",
))

# K-Means baseline horizontal line
fig5.add_trace(go.Scatter(
    x=[iter_axis[0], iter_axis[-1]],
    y=[km_error, km_error],
    mode="lines",
    line=dict(width=2.5, color="#e74c3c", dash="dash"),
    name=f"K-Means baseline (K={M})",
))

fig5.update_layout(
    title="Reconstruction Error vs Iteration (ESL Fig 14.18)",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="Iteration",
    yaxis_title="Reconstruction Error (SSE)",
    width=800, height=520,
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(x=0.55, y=0.95, bgcolor="rgba(0,0,0,0.5)"),
)

fig5_path = os.path.join(OUTPUT_DIR, "fig5_reconstruction_error.html")
fig5.write_html(fig5_path)
print(f"    Saved: {fig5_path}")


# %% ============================================================
#  8. SUMMARY
# ============================================================
print(f"\n{'=' * 65}")
print("  EXPERIMENT SUMMARY")
print(f"{'=' * 65}")
print(f"  Dataset       : Half-Sphere (N={N}, p={p}, K={K_CLASSES} classes)")
print(f"  Model         : Self-Organizing Map (Kohonen, 1982)")
print(f"  Grid          : {GRID_SIZE[0]} x {GRID_SIZE[1]} = {M} prototypes")
print(f"  Training      : {NUM_PASSES} passes x {N} samples = {total_iterations} iterations")
print(f"  Init sigma    : {INIT_SIGMA}")
print(f"  Learning rate : {LEARNING_RATE}")
print(f"  MQE           : {mqe:.4f}")
print(f"  Topographic Err: {te:.4f} ({te*100:.1f}%)")
print(f"  Recon Error   : {error_history[-1]:.4f} (SOM), {km_error:.4f} (K-Means)")
print(f"  Training time : {t_train:.2f}s")
print(f"{'=' * 65}")

# Comparison table
print(f"\n{'=' * 65}")
print("  MATLAB vs PYTHON COMPARISON")
print(f"{'=' * 65}")
print(f"  {'Aspect':<25} | {'MATLAB (custom)':<22} | {'Python (production)':<22}")
print(f"  {'-' * 75}")
print(f"  {'SOM Engine':<25} | {'OceanStats.SOM':<22} | {'MiniSom (PyPI)':<22}")
print(f"  {'K-Means':<25} | {'OceanStats.KMeans':<22} | {'sklearn.KMeans':<22}")
print(f"  {'Visualization':<25} | {'OceanViz (static)':<22} | {'Plotly (interactive)':<22}")
print(f"  {'Animation':<25} | {'drawnow + pause':<22} | {'Plotly frames+slider':<22}")
print(f"  {'U-Matrix':<25} | {'Manual loop':<22} | {'som.distance_map()':<22}")
print(f"  {'Topographic Error':<25} | {'Manual loop':<22} | {'som.topographic_error':<22}")
print(f"  {'Lines of code':<25} | {'~785 (class+exp)':<22} | {'~420 (exp only)':<22}")
print(f"  {'Interactivity':<25} | {'None':<22} | {'Rotate/zoom/scrub':<22}")
print(f"  {'Dependencies':<25} | {'None (all custom)':<22} | {'minisom,plotly,sklearn':<22}")
print(f"{'=' * 65}\n")
