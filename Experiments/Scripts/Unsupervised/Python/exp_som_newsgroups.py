"""
exp_som_newsgroups.py
=====================
SOM Document Organization & Retrieval -- Production Python Edition

Replicates ESL Section 14.4, Figure 14.19: WEBSOM applied to a newsgroup corpus.
Uses scikit-learn for TF-IDF + SVD, MiniSom for training, and Plotly for
interactive heatmap visualization with auto-generated topic labels.

DATASET:        20 Newsgroups (sklearn built-in), N=18846, 20 categories
PREPROCESSING:  TF-IDF (10K terms) -> TruncatedSVD (50 components)
MODEL:          Self-Organizing Map (Kohonen, 1982)
                25x25 grid = 625 prototypes
                Gaussian neighborhood kernel
ASSUMPTIONS:    - Headers/footers/quotes removed to prevent metadata leakage.
                - SVD reduces 10K sparse features to 50 dense latent topics.
                - Grid size follows 5*sqrt(N) heuristic.
REPRODUCIBILITY: random_seed=42

Ref: Hastie et al. (2009), ESL Section 14.4, Figure 14.19.
     Kohonen et al. (2000), WEBSOM project.

Usage:
    python exp_som_newsgroups.py
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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hyperparameters
GRID_SIZE       = (25, 25)
INIT_SIGMA      = 10.0
LEARNING_RATE   = 0.5
NUM_PASSES      = 10
SVD_COMPONENTS  = 50
MAX_FEATURES    = 10000
RANDOM_SEED     = 42

# Plotly theme
PLOTLY_TEMPLATE = "plotly_dark"
BG_COLOR        = "#1a1a2e"
PAPER_COLOR     = "#16213e"
TEXT_COLOR      = "#e0e0e0"

# Super-group color mapping
SUPERGROUP_MAP = {
    'comp.graphics': 'Computers', 'comp.os.ms-windows.misc': 'Computers',
    'comp.sys.ibm.pc.hardware': 'Computers', 'comp.sys.mac.hardware': 'Computers',
    'comp.windows.x': 'Computers',
    'sci.crypt': 'Science', 'sci.electronics': 'Science',
    'sci.med': 'Science', 'sci.space': 'Science',
    'rec.autos': 'Recreation', 'rec.motorcycles': 'Recreation',
    'rec.sport.baseball': 'Recreation', 'rec.sport.hockey': 'Recreation',
    'talk.politics.guns': 'Politics', 'talk.politics.mideast': 'Politics',
    'talk.politics.misc': 'Politics',
    'alt.atheism': 'Religion', 'soc.religion.christian': 'Religion',
    'talk.religion.misc': 'Religion',
    'misc.forsale': 'Misc',
}
SUPERGROUP_COLORS = {
    'Computers': '#3498db', 'Science': '#2ecc71', 'Recreation': '#e67e22',
    'Politics': '#e74c3c', 'Religion': '#9b59b6', 'Misc': '#95a5a6',
}

print("=" * 65)
print("  SOM DOCUMENT ORGANIZATION -- Python Production")
print(f"  Grid:          {GRID_SIZE[0]}x{GRID_SIZE[1]} = {GRID_SIZE[0]*GRID_SIZE[1]} prototypes")
print(f"  SVD dims:      {SVD_COMPONENTS}")
print(f"  TF-IDF terms:  {MAX_FEATURES}")
print(f"  Passes:        {NUM_PASSES}")
print(f"  Seed:          {RANDOM_SEED}")
print("=" * 65)


# %% ============================================================
#  PHASE 1: DATA ACQUISITION
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 1: DATA ACQUISITION")
print(f"{'=' * 65}")

t1 = time.time()

dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
docs         = dataset.data
true_labels  = dataset.target
target_names = dataset.target_names

N_docs = len(docs)
avg_len = np.mean([len(d.split()) for d in docs])

t1 = time.time() - t1

print(f"  Documents:    {N_docs:,}")
print(f"  Categories:   {len(target_names)}")
print(f"  Avg doc len:  {avg_len:.0f} words")
print(f"  Load time:    {t1:.1f}s")

# Category distribution
print(f"\n  {'Category':<35} | {'Count':>6} | {'Super-group':<12}")
print(f"  {'-' * 60}")
for i, name in enumerate(target_names):
    count = np.sum(true_labels == i)
    sg = SUPERGROUP_MAP.get(name, '?')
    print(f"  {name:<35} | {count:>6} | {sg:<12}")


# %% ============================================================
#  PHASE 2: FEATURE ENGINEERING -- TF-IDF + SVD
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 2: FEATURE ENGINEERING (TF-IDF + SVD)")
print(f"{'=' * 65}")

t2 = time.time()

# Step 1: TF-IDF vectorization
tfidf = TfidfVectorizer(
    max_features=MAX_FEATURES,
    stop_words='english',
    max_df=0.5,
    min_df=5,
)
X_tfidf = tfidf.fit_transform(docs)
feature_names = tfidf.get_feature_names_out()

print(f"  TF-IDF matrix: {X_tfidf.shape[0]:,} x {X_tfidf.shape[1]:,}")
print(f"  Sparse density: {X_tfidf.nnz / (X_tfidf.shape[0]*X_tfidf.shape[1])*100:.2f}%")

# Step 2: SVD dimensionality reduction (as recommended by ESL)
svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_SEED)
X = svd.fit_transform(X_tfidf)

explained = svd.explained_variance_ratio_.sum() * 100

t2 = time.time() - t2

print(f"  SVD output:    {X.shape[0]:,} x {X.shape[1]}")
print(f"  Explained var: {explained:.1f}%")
print(f"  Phase 2 time:  {t2:.1f}s")

N, p = X.shape


# %% ============================================================
#  PHASE 3: TRAIN / TEST SPLIT
#  N/A -- unsupervised learning experiment, full dataset for SOM.
# ============================================================

# %% ============================================================
#  PHASE 4: MODEL CONFIGURATION
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 4: MODEL CONFIGURATION")
print(f"{'=' * 65}")

Q1, Q2 = GRID_SIZE
M = Q1 * Q2

som = MiniSom(
    x=Q1, y=Q2, input_len=p,
    sigma=INIT_SIGMA,
    learning_rate=LEARNING_RATE,
    decay_function="linear_decay_to_zero",
    sigma_decay_function="linear_decay_to_one",
    neighborhood_function="gaussian",
    topology="rectangular",
    random_seed=RANDOM_SEED,
)

som.random_weights_init(X)

print(f"  MiniSom: {Q1}x{Q2} = {M} prototypes, p={p}")
print(f"  Sigma: {INIT_SIGMA}, LR: {LEARNING_RATE}")
print(f"  Decay: linear (lr -> 0, sigma -> 1)")


# %% ============================================================
#  PHASE 5: TRAINING
#  Manual loop with global decay counter.
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 5: TRAINING")
print(f"{'=' * 65}")

t5 = time.time()

total_iterations = NUM_PASSES * N
rng_train = np.random.RandomState(RANDOM_SEED)
iter_count = 0
mqe_history = []

# Record initial MQE
mqe_history.append(som.quantization_error(X))
print(f"  [SOM] Initial MQE: {mqe_history[-1]:.4f}")

for pass_idx in range(NUM_PASSES):
    t_pass = time.time()
    perm = rng_train.permutation(N)
    for s in range(N):
        xi = X[perm[s]]
        winner = som.winner(xi)
        som.update(xi, winner, iter_count, total_iterations)
        iter_count += 1

    mqe_now = som.quantization_error(X)
    mqe_history.append(mqe_now)
    t_pass = time.time() - t_pass
    print(f"  [SOM] Pass {pass_idx+1:2d}/{NUM_PASSES} | MQE={mqe_now:.4f} | {t_pass:.1f}s")

t5 = time.time() - t5
print(f"\n  Training time: {t5:.1f}s ({total_iterations:,} iterations)")


# %% ============================================================
#  PHASE 6: EVALUATION
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 6: EVALUATION")
print(f"{'=' * 65}")

# Basic SOM metrics
mqe = som.quantization_error(X)
te  = som.topographic_error(X)
U   = som.distance_map()

print(f"  MQE (Mean Quantization Error) : {mqe:.4f}")
print(f"  Topographic Error             : {te:.4f} ({te*100:.1f}%)")

# --- Build per-node data structures ---
# For each prototype: which documents are assigned to it?
node_docs  = {}   # (i,j) -> list of doc indices
node_labels = {}  # (i,j) -> list of true labels

for idx in range(N):
    xi = X[idx]
    w = som.winner(xi)
    if w not in node_docs:
        node_docs[w] = []
        node_labels[w] = []
    node_docs[w].append(idx)
    node_labels[w].append(true_labels[idx])

# Document density map
density_map = np.zeros((Q1, Q2))
for (i, j), doc_list in node_docs.items():
    density_map[i, j] = len(doc_list)

populated = sum(1 for v in node_docs.values() if len(v) > 0)
print(f"  Populated nodes: {populated}/{M} ({populated/M*100:.0f}%)")

# Category purity: for each node, fraction belonging to the majority class
purities = []
for (i, j), labels in node_labels.items():
    if len(labels) > 0:
        counts = np.bincount(labels, minlength=len(target_names))
        purities.append(counts.max() / len(labels))

avg_purity = np.mean(purities) if purities else 0
print(f"  Avg category purity: {avg_purity:.2f} ({avg_purity*100:.0f}%)")
print(f"  Random baseline:     {1/len(target_names):.2f} ({100/len(target_names):.0f}%)")

# --- Dominant category map ---
dominant_cat = np.full((Q1, Q2), -1, dtype=int)
for (i, j), labels in node_labels.items():
    if len(labels) > 0:
        counts = np.bincount(labels, minlength=len(target_names))
        dominant_cat[i, j] = np.argmax(counts)

# --- Top term per node (for WEBSOM-style labeling) ---
# Reconstruct approximate TF-IDF from SVD weights, find top term
W_som = som.get_weights().reshape(M, p)   # (625, 50)
V = svd.components_                        # (50, 10000)
W_tfidf_approx = W_som @ V                # (625, 10000) - back to term space

node_top_terms = np.empty((Q1, Q2), dtype=object)
for i in range(Q1):
    for j in range(Q2):
        lin_idx = i * Q2 + j
        top_term_idx = np.argmax(W_tfidf_approx[lin_idx])
        node_top_terms[i, j] = feature_names[top_term_idx]

print(f"  Top-term labeling: computed for all {M} nodes")


# %% ============================================================
#  PHASE 7: VISUALIZATION
# ============================================================
print(f"\n{'=' * 65}")
print("  PHASE 7: VISUALIZATION")
print(f"{'=' * 65}")


# -----------------------------------------------------------------------
# Figure 1: Category Distribution (Bar Chart)
# -----------------------------------------------------------------------
print("  Creating Figure 1: Category Distribution...")

cat_counts = [np.sum(true_labels == i) for i in range(len(target_names))]
cat_colors = [SUPERGROUP_COLORS[SUPERGROUP_MAP[n]] for n in target_names]

fig1 = go.Figure(data=go.Bar(
    x=list(target_names),
    y=cat_counts,
    marker_color=cat_colors,
    hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
))
fig1.update_layout(
    title="20 Newsgroups -- Category Distribution",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="Newsgroup", yaxis_title="Document Count",
    xaxis_tickangle=-45,
    width=1000, height=550,
    margin=dict(l=60, r=30, t=60, b=150),
)

fig1_path = os.path.join(OUTPUT_DIR, "fig1_category_distribution.html")
fig1.write_html(fig1_path)
print(f"    Saved: {fig1_path}")


# -----------------------------------------------------------------------
# Figure 2: Document Density Heatmap with Topic Labels (ESL Fig 14.19)
# -----------------------------------------------------------------------
print("  Creating Figure 2: Density Heatmap with Topic Labels...")

# Build annotation text: top term for populated nodes only
annotations = []
for i in range(Q1):
    for j in range(Q2):
        if density_map[i, j] > 5:   # Only label nodes with >5 docs
            annotations.append(dict(
                x=j, y=i,
                text=str(node_top_terms[i, j]),
                showarrow=False,
                font=dict(size=7, color="white"),
            ))

fig2 = go.Figure(data=go.Heatmap(
    z=density_map,
    colorscale="Hot",
    colorbar=dict(title="Documents", title_font=dict(color=TEXT_COLOR),
                  tickfont=dict(color=TEXT_COLOR)),
    hovertemplate="Node (%{y}, %{x})<br>Documents: %{z}<extra></extra>",
))
fig2.update_layout(
    title="Document Density Heatmap with Topic Labels (ESL Fig 14.19)",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="Grid Column", yaxis_title="Grid Row",
    annotations=annotations,
    width=800, height=750,
    margin=dict(l=60, r=20, t=60, b=60),
    yaxis=dict(scaleanchor="x"),
)

fig2_path = os.path.join(OUTPUT_DIR, "fig2_density_heatmap.html")
fig2.write_html(fig2_path)
print(f"    Saved: {fig2_path}")


# -----------------------------------------------------------------------
# Figure 3: Topic Map (Color-coded by Dominant Category)
# -----------------------------------------------------------------------
print("  Creating Figure 3: Topic Map...")

# Map dominant category index to super-group color
supergroup_idx_map = {}
for i, name in enumerate(target_names):
    supergroup_idx_map[i] = SUPERGROUP_MAP[name]

# Build a numeric super-group grid for coloring
sg_names_ordered = ['Computers', 'Science', 'Recreation', 'Politics', 'Religion', 'Misc']
sg_to_num = {s: idx for idx, s in enumerate(sg_names_ordered)}

topic_color_grid = np.full((Q1, Q2), -1.0)
hover_text = np.empty((Q1, Q2), dtype=object)

for i in range(Q1):
    for j in range(Q2):
        cat_idx = dominant_cat[i, j]
        if cat_idx >= 0:
            sg = supergroup_idx_map[cat_idx]
            topic_color_grid[i, j] = sg_to_num[sg]
            n_docs_here = int(density_map[i, j])
            hover_text[i, j] = (f"Node ({i},{j})<br>"
                                f"Dominant: {target_names[cat_idx]}<br>"
                                f"Super-group: {sg}<br>"
                                f"Docs: {n_docs_here}<br>"
                                f"Top term: {node_top_terms[i,j]}")
        else:
            topic_color_grid[i, j] = float('nan')
            hover_text[i, j] = f"Node ({i},{j})<br>Empty"

# Custom discrete colorscale for 6 super-groups
n_sg = len(sg_names_ordered)
discrete_cs = []
for idx, sg in enumerate(sg_names_ordered):
    lo = idx / n_sg
    hi = (idx + 1) / n_sg
    c = SUPERGROUP_COLORS[sg]
    discrete_cs.append([lo, c])
    discrete_cs.append([hi, c])

fig3 = go.Figure(data=go.Heatmap(
    z=topic_color_grid,
    text=hover_text,
    hoverinfo="text",
    colorscale=discrete_cs,
    zmin=-0.5, zmax=n_sg - 0.5,
    colorbar=dict(
        title="Super-group", title_font=dict(color=TEXT_COLOR),
        tickfont=dict(color=TEXT_COLOR),
        tickvals=list(range(n_sg)),
        ticktext=sg_names_ordered,
    ),
    showscale=True,
))
fig3.update_layout(
    title="Topic Map -- Dominant Category per Node",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="Grid Column", yaxis_title="Grid Row",
    width=800, height=750,
    margin=dict(l=60, r=20, t=60, b=60),
    yaxis=dict(scaleanchor="x"),
)

fig3_path = os.path.join(OUTPUT_DIR, "fig3_topic_map.html")
fig3.write_html(fig3_path)
print(f"    Saved: {fig3_path}")


# -----------------------------------------------------------------------
# Figure 4: U-Matrix Heatmap
# -----------------------------------------------------------------------
print("  Creating Figure 4: U-Matrix...")

fig4 = go.Figure(data=go.Heatmap(
    z=U,
    colorscale="Hot",
    colorbar=dict(title="Mean Neighbor<br>Distance", title_font=dict(color=TEXT_COLOR),
                  tickfont=dict(color=TEXT_COLOR)),
    hovertemplate="Node (%{y}, %{x})<br>Distance: %{z:.4f}<extra></extra>",
))
fig4.update_layout(
    title="U-Matrix -- Cluster Boundaries",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="Grid Column", yaxis_title="Grid Row",
    width=700, height=650,
    margin=dict(l=60, r=20, t=60, b=60),
    yaxis=dict(scaleanchor="x"),
)

fig4_path = os.path.join(OUTPUT_DIR, "fig4_umatrix.html")
fig4.write_html(fig4_path)
print(f"    Saved: {fig4_path}")


# -----------------------------------------------------------------------
# Figure 5: MQE Convergence Curve
# -----------------------------------------------------------------------
print("  Creating Figure 5: MQE Convergence...")

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=list(range(len(mqe_history))),
    y=mqe_history,
    mode="lines+markers",
    marker=dict(size=8, color="#2ecc71"),
    line=dict(width=2.5, color="#2ecc71"),
    name="MQE per pass",
    hovertemplate="Pass %{x}<br>MQE: %{y:.4f}<extra></extra>",
))
fig5.update_layout(
    title="MQE Convergence -- Training Progress",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR, plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="Pass (0 = initial)", yaxis_title="Mean Quantization Error",
    width=750, height=480,
    margin=dict(l=60, r=30, t=60, b=60),
)

fig5_path = os.path.join(OUTPUT_DIR, "fig5_mqe_convergence.html")
fig5.write_html(fig5_path)
print(f"    Saved: {fig5_path}")


# %% ============================================================
#  PHASE 8: SUMMARY
# ============================================================
print(f"\n{'=' * 65}")
print("  EXPERIMENT SUMMARY")
print(f"{'=' * 65}")
print(f"  Dataset       : 20 Newsgroups (N={N:,}, categories={len(target_names)})")
print(f"  Features      : TF-IDF ({MAX_FEATURES:,}) -> SVD ({SVD_COMPONENTS})")
print(f"  Model         : SOM {Q1}x{Q2} = {M} prototypes")
print(f"  Training      : {NUM_PASSES} passes x {N:,} = {total_iterations:,} iterations")
print(f"  MQE           : {mqe:.4f}")
print(f"  Topographic Err: {te:.4f} ({te*100:.1f}%)")
print(f"  Category purity: {avg_purity:.2f} ({avg_purity*100:.0f}%)")
print(f"  Populated nodes: {populated}/{M}")
print(f"  Training time : {t5:.1f}s")
print(f"{'=' * 65}")

# Top-5 most coherent regions
print(f"\n  TOP-5 PUREST NODES:")
node_purity_list = []
for (i, j), labels in node_labels.items():
    if len(labels) >= 10:
        counts = np.bincount(labels, minlength=len(target_names))
        best = np.argmax(counts)
        purity = counts[best] / len(labels)
        node_purity_list.append((i, j, target_names[best], purity, len(labels)))

node_purity_list.sort(key=lambda x: -x[3])
for rank, (i, j, cat, pur, ndocs) in enumerate(node_purity_list[:5], 1):
    print(f"  {rank}. Node ({i:2d},{j:2d}): {cat:<35} purity={pur:.0%} ({ndocs} docs)")

print(f"\n{'=' * 65}")
print("  PIPELINE COMPLETE")
print(f"{'=' * 65}\n")
