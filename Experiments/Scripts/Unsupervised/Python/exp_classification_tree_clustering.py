"""
exp_classification_tree_clustering.py
======================================
ESL Exercise 14.4 -- Classification Tree Clustering on Demographic Data

METHOD (ESL Section 14.4 / Ex. 14.4):
  1. Load the San Francisco Bay Area demographic data (Table 14.1).
  2. Generate a reference (null) sample by randomly permuting values
     within each feature column independently -- this destroys inter-
     feature structure while preserving each marginal distribution.
  3. Assign labels: real data = Class 1, reference data = Class 0.
  4. Fit a CART classification tree (sklearn DecisionTreeClassifier).
  5. Describe terminal nodes with highest estimated Class 1 probability.
  6. Compare to the PRIM association-rule results cited in the book,
     and to K-means clustering (K=3) on the same data.

DATASET:   SF Bay Area Demographics (Impact Resources Inc., 1987)
           N=8993, 14 features (Income + 13 demographics)
           Missing values: median imputation per feature.

NOTES:
  - All features are treated as ordinal/numeric integers; category
    semantics are documented in comments for human interpretation.
  - We do NOT encode expected outcomes; the tree discovers structure.
  - RANDOM_SEED ensures reproducibility.
  - Feature ordering in the data file follows data.info.txt:
        Col 0 = Annual Income (1-9)
        Col 1 = Sex (1-2)
        Col 2 = Marital Status (1-5)
        Col 3 = Age (1-7)
        Col 4 = Education (1-6)
        Col 5 = Occupation (1-9)
        Col 6 = Years in Bay Area (1-5)
        Col 7 = Dual Incomes (1-3)
        Col 8 = Number in Household (1-9)
        Col 9 = Number of Children (0-9)
        Col 10 = Householder Status (1-3)
        Col 11 = Type of Home (1-5)
        Col 12 = Ethnic Classification (1-8)
        Col 13 = Language in Home (1-3)

Ref: Hastie, Tibshirani, Friedman (2009), ESL, Section 14.2 & Ex. 14.4
"""

# =============================================================================
#  IMPORTS & CONFIGURATION
# =============================================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = r"c:\Ocean Library\Matlab code practice\Data\Demographics\data.txt"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "tree_clustering")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Hyperparameters ------------------------------------------------------------
RANDOM_SEED      = 42
MAX_DEPTH        = 6      # Depth-limited CART tree (prevents overfit; readable)
MIN_SAMPLES_LEAF = 50     # Minimum leaf size for interpretability
TOP_N_LEAVES     = 5      # Number of high-probability leaves to report
K_MEANS_K        = 3      # K for K-means comparison

# -- Plotly theme (library standard) -------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"
BG_COLOR        = "#1a1a2e"
PAPER_COLOR     = "#16213e"
TEXT_COLOR      = "#e0e0e0"
ACCENT1         = "#3498db"   # blue
ACCENT2         = "#e74c3c"   # red
ACCENT3         = "#2ecc71"   # green
ACCENT4         = "#f39c12"   # orange

# -- Feature metadata (ESL Table 14.1) ----------------------------------------
FEATURE_NAMES = [
    "Income",             # Col 0  – Ordinal  1-9
    "Sex",                # Col 1  – Categorical 1-2
    "Marital Status",     # Col 2  – Categorical 1-5
    "Age",                # Col 3  – Ordinal  1-7
    "Education",          # Col 4  – Ordinal  1-6
    "Occupation",         # Col 5  – Categorical 1-9
    "Years in Bay Area",  # Col 6  – Ordinal  1-5
    "Dual Incomes",       # Col 7  – Categorical 1-3
    "Number in Household",# Col 8  – Ordinal  1-9
    "Number of Children", # Col 9  – Ordinal  0-9
    "Householder Status", # Col 10 – Categorical 1-3
    "Type of Home",       # Col 11 – Categorical 1-5
    "Ethnic Classification",# Col 12 – Categorical 1-8
    "Language in Home",   # Col 13 – Categorical 1-3
]

# Human-readable value maps for terminal-node reporting
VALUE_MAPS = {
    "Income": {1:"<$10k",2:"$10-15k",3:"$15-20k",4:"$20-25k",
               5:"$25-30k",6:"$30-40k",7:"$40-50k",8:"$50-75k",9:">$75k"},
    "Sex": {1:"Male", 2:"Female"},
    "Marital Status": {1:"Married",2:"Living together",3:"Divorced/Sep",
                       4:"Widowed",5:"Single"},
    "Age": {1:"14-17",2:"18-24",3:"25-34",4:"35-44",
            5:"45-54",6:"55-64",7:"65+"},
    "Education": {1:"Grade≤8",2:"Grades 9-11",3:"HS Grad",
                  4:"Some College",5:"College Grad",6:"Grad Study"},
    "Occupation": {1:"Prof/Mgr",2:"Sales",3:"Factory/Driver",4:"Clerical",
                   5:"Homemaker",6:"Student",7:"Military",8:"Retired",9:"Unemployed"},
    "Years in Bay Area": {1:"<1yr",2:"1-3yr",3:"4-6yr",4:"7-10yr",5:">10yr"},
    "Dual Incomes": {1:"Not Married",2:"Yes",3:"No"},
    "Householder Status": {1:"Own",2:"Rent",3:"With Parents"},
    "Type of Home": {1:"House",2:"Condo",3:"Apartment",4:"Mobile",5:"Other"},
    "Language in Home": {1:"English",2:"Spanish",3:"Other"},
}

print("=" * 70)
print("  ESL Ex 14.4 — Classification Tree Clustering (Demographic Data)")
print("=" * 70)


# =============================================================================
#  PHASE 1: DATA ACQUISITION & PREPROCESSING
# =============================================================================
print(f"\n{'-'*70}")
print("  PHASE 1: DATA ACQUISITION & PREPROCESSING")
print(f"{'-'*70}")

# -- Assumption: columns correspond to FEATURE_NAMES as listed in data.info.txt
df_raw = pd.read_csv(DATA_PATH, sep=r'\s+', header=None,
                     na_values=['NA'], names=FEATURE_NAMES)

N_raw, p = df_raw.shape
print(f"  Raw shape:           {N_raw} x {p}")
print(f"  Missing values:\n{df_raw.isna().sum().to_string()}")

# -- Median imputation (each feature independently) ---------------------------
# Assumption: missing values are MCAR/MAR; median preserves ordinal structure.
df = df_raw.copy()
for col in df.columns:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# Convert to integer (ordinal codes)
df = df.round().astype(int)

N, p = df.shape
print(f"\n  After imputation:    {N} x {p}  (no missing values)")

X = df.values   # numpy array  (N, 14)


# =============================================================================
#  PHASE 2: GENERATE NULL REFERENCE SAMPLE
# =============================================================================
print(f"\n{'-'*70}")
print("  PHASE 2: GENERATE NULL REFERENCE SAMPLE")
print(f"{'-'*70}")

# -- Method: randomly permute each feature column independently.
#    This preserves the marginal distribution of each feature but
#    destroys all cross-feature correlations (the signal we want to find).
rng = np.random.RandomState(RANDOM_SEED)

X_ref = X.copy()
for j in range(p):
    X_ref[:, j] = rng.permutation(X_ref[:, j])

print(f"  Reference sample:    {X_ref.shape[0]} x {X_ref.shape[1]}")
print("  Permuted independently per column — marginals preserved.")
print("  Cross-feature structure destroyed — serves as null model.")

# -- Stack real (class=1) and reference (class=0) -----------------------------
X_combined = np.vstack([X, X_ref])
y_combined  = np.hstack([np.ones(N), np.zeros(N)]).astype(int)

print(f"\n  Combined dataset:    {X_combined.shape[0]} x {X_combined.shape[1]}")
print(f"  Class 1 (real):      {y_combined.sum()}")
print(f"  Class 0 (reference): {(1-y_combined).sum()}")


# =============================================================================
#  PHASE 3: CLASSIFICATION TREE
# =============================================================================
print(f"\n{'-'*70}")
print("  PHASE 3: CLASSIFICATION TREE FITTING")
print(f"{'-'*70}")

# -- CART tree: Gini impurity, limited depth for interpretability.
#    In a balanced problem (N class-1 == N class-0), a terminal node
#    with P(class=1) >> 0.5 identifies a dense real-data region.
tree = DecisionTreeClassifier(
    max_depth        = MAX_DEPTH,
    min_samples_leaf = MIN_SAMPLES_LEAF,
    random_state     = RANDOM_SEED,
    criterion        = "gini",
)
tree.fit(X_combined, y_combined)

n_leaves = tree.get_n_leaves()
print(f"  Tree depth:          {tree.get_depth()}")
print(f"  Number of leaves:    {n_leaves}")
print(f"  Training accuracy:   {tree.score(X_combined, y_combined):.4f}")


# =============================================================================
#  PHASE 4: TERMINAL NODE ANALYSIS
# =============================================================================
print(f"\n{'-'*70}")
print("  PHASE 4: TERMINAL NODE ANALYSIS")
print(f"{'-'*70}")

# -- Map each REAL data point to its leaf -------------------------------------
leaf_ids_real = tree.apply(X)          # leaf index for each real sample
leaf_ids_all  = tree.apply(X_combined) # leaf index for combined data

# -- For each leaf: compute Class-1 probability and support -------------------
unique_leaves = np.unique(leaf_ids_all)
leaf_stats = []

for lid in unique_leaves:
    mask_all  = (leaf_ids_all == lid)
    mask_real = (leaf_ids_real == lid)

    n_total = mask_all.sum()
    n_real  = mask_real.sum()         # real data points in leaf
    n_ref   = n_total - n_real        # reference points in leaf

    p_class1 = n_real / n_total if n_total > 0 else 0.0
    support  = n_real / N             # fraction of real data in leaf

    leaf_stats.append({
        "leaf_id"  : lid,
        "n_total"  : n_total,
        "n_real"   : n_real,
        "n_ref"    : n_ref,
        "p_class1" : p_class1,
        "support"  : support,
    })

leaf_df = pd.DataFrame(leaf_stats).sort_values("p_class1", ascending=False)

# -- Extract decision path for each leaf --------------------------------------
feature  = tree.tree_.feature
threshold = tree.tree_.threshold

def get_decision_path(tree_obj, leaf_id, feature_names):
    """Return a list of (feature_name, direction, threshold) for a leaf node."""
    node_indicator = tree_obj.decision_path(
        X_combined[leaf_ids_all == leaf_id][:1]
    ) #decision_path return a aparse matrix 1 x total nodes
    path_nodes = node_indicator.indices.tolist()
    rules = []
    for i, node in enumerate(path_nodes[:-1]): #All node except the last
        feat_idx = tree_obj.tree_.feature[node]
        thr      = tree_obj.tree_.threshold[node]
        # determine direction taken (left = <=, right = >)
        next_node = path_nodes[i + 1]
        direction = "<=" if next_node == tree_obj.tree_.children_left[node] else ">"
        rules.append({
            "feature"   : feature_names[feat_idx],
            "direction" : direction,
            "threshold" : thr,
        })
    return rules

print(f"\n  Top {TOP_N_LEAVES} terminal nodes by P(Class=1):\n")

top_leaves = leaf_df.head(TOP_N_LEAVES)
leaf_rules = {}  # store for later use in plots

for rank, (_, row) in enumerate(top_leaves.iterrows(), start=1): #iterrows yield (index, Series) pairs, we discard index.
    lid     = int(row["leaf_id"]) #Get leaf id from current
    rules   = get_decision_path(tree, lid, FEATURE_NAMES)
    leaf_rules[lid] = rules

    print(f"  [{rank}] Leaf {lid}")
    print(f"      P(Class=1)  = {row['p_class1']:.4f}")
    print(f"      Support     = {row['support']*100:.2f}%  ({int(row['n_real'])} real samples)")
    print(f"      n_total     = {int(row['n_total'])}  (real={int(row['n_real'])}, ref={int(row['n_ref'])})")
    print("      Decision rules:")
    for r in rules:
        thr_val = r["threshold"]
        # Round threshold to nearest integer for ordinal readability
        thr_int = int(round(thr_val))
        feat = r["feature"]
        dirn = r["direction"]
        # Attempt human-readable value
        vmap = VALUE_MAPS.get(feat, {})
        # For ordinal <=, the category is thr_int; for > the next level
        readable = vmap.get(thr_int, str(thr_int))
        print(f"        {feat} {dirn} {thr_val:.2f}  [≈ {readable}]")
    print()


# =============================================================================
#  PHASE 5: K-MEANS CLUSTERING (COMPARISON)
# =============================================================================
print(f"\n{'-'*70}")
print(f"  PHASE 5: K-MEANS CLUSTERING (K={K_MEANS_K})")
print(f"{'-'*70}")

# -- K-means on real data only (not the reference sample)
kmeans = KMeans(n_clusters=K_MEANS_K, random_state=RANDOM_SEED, n_init=20)
km_labels = kmeans.fit_predict(X)

print(f"  Cluster sizes: {np.bincount(km_labels).tolist()}")
print(f"  Within-cluster sum of squares (inertia): {kmeans.inertia_:.2f}")

# -- Cluster profiles: mean feature value per cluster -------------------------
df_km = df.copy()
df_km["Cluster"] = km_labels

cluster_means = df_km.groupby("Cluster")[FEATURE_NAMES].mean()
print("\n  Cluster mean profiles:")
print(cluster_means.to_string())


# =============================================================================
#  PHASE 6: VISUALIZATION
# =============================================================================
print(f"\n{'-'*70}")
print("  PHASE 6: VISUALIZATION")
print(f"{'-'*70}")

COLORS = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, "#9b59b6", "#1abc9c"]

# -----------------------------------------------------------------------------
# Fig 1: Leaf P(Class=1) vs Support scatter — all leaves
# -----------------------------------------------------------------------------
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=leaf_df["support"] * 100,
    y=leaf_df["p_class1"],
    mode="markers",
    marker=dict(
        size=np.clip(leaf_df["n_real"] / 30, 5, 30),
        color=leaf_df["p_class1"],
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="P(Class=1)", tickfont=dict(color=TEXT_COLOR)),
        line=dict(color="white", width=0.5),
    ),
    text=[f"Leaf {lid}<br>P(C=1)={p:.3f}<br>Support={s*100:.1f}%"
          for lid, p, s in zip(leaf_df["leaf_id"], leaf_df["p_class1"], leaf_df["support"])],
    hovertemplate="%{text}<extra></extra>",
    name="Leaf",
))

# Highlight top leaves
fig1.add_trace(go.Scatter(
    x=top_leaves["support"] * 100,
    y=top_leaves["p_class1"],
    mode="markers+text",
    marker=dict(size=18, color="rgba(0,0,0,0)",
                line=dict(color=ACCENT2, width=2.5)),
    text=[f"#{r+1}" for r in range(TOP_N_LEAVES)],
    textposition="top center",
    textfont=dict(color=ACCENT2, size=11),
    name=f"Top {TOP_N_LEAVES} Leaves",
))

# P=0.5 reference line
fig1.add_hline(y=0.5, line=dict(color="rgba(255,255,255,0.4)", dash="dash"),
               annotation_text="Chance (P=0.5)", annotation_font_color=TEXT_COLOR)

fig1.update_layout(
    title="Fig 1 — Classification Tree Leaves: P(Class=1) vs Support<br>"
          "<sup>Bubble size ∝ number of real samples in leaf</sup>",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="Support (% of real data in leaf)",
    yaxis_title="Estimated P(Class=1)",
    width=800, height=550,
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="white", borderwidth=1),
)
f1_path = os.path.join(OUTPUT_DIR, "fig1_leaf_prob_vs_support.html")
fig1.write_html(f1_path)
print(f"  Saved: {f1_path}")


# -----------------------------------------------------------------------------
# Fig 2: Top-leaf feature profiles (radar / bar subplot)
# -----------------------------------------------------------------------------
# For each top leaf, compute mean feature value of real data in that leaf
# normalise features to [0,1] for comparability

X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9)

fig2 = make_subplots(
    rows=1, cols=TOP_N_LEAVES,
    subplot_titles=[f"Leaf #{r+1}<br>P(C=1)={row['p_class1']:.3f}"
                    for r, (_, row) in enumerate(top_leaves.iterrows())],
    shared_yaxes=True,
)

for rank, (_, row) in enumerate(top_leaves.iterrows(), start=1):
    lid  = int(row["leaf_id"])
    mask = (leaf_ids_real == lid)
    if mask.sum() > 0:
        means = X_norm[mask].mean(axis=0)
    else:
        means = np.zeros(p)

    # Global mean (all real data)
    global_means = X_norm.mean(axis=0)
    deviation    = means - global_means   # positive = above average

    fig2.add_trace(go.Bar(
        x=FEATURE_NAMES,
        y=deviation,
        marker_color=[ACCENT2 if v > 0 else ACCENT1 for v in deviation],
        name=f"Leaf #{rank}",
        showlegend=False,
    ), row=1, col=rank)

fig2.update_xaxes(tickangle=60, tickfont=dict(size=8))
fig2.update_yaxes(title_text="Deviation from global mean", row=1, col=1)
fig2.update_layout(
    title="Fig 2 — Feature Profiles of Top-K Leaves (deviation from global mean)",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    height=550, width=200 * TOP_N_LEAVES + 100,
)
f2_path = os.path.join(OUTPUT_DIR, "fig2_top_leaf_profiles.html")
fig2.write_html(f2_path)
print(f"  Saved: {f2_path}")


# -----------------------------------------------------------------------------
# Fig 3: K-means cluster heat map (cluster × feature mean)
# -----------------------------------------------------------------------------
cluster_means_norm = (cluster_means - cluster_means.min()) / \
                     (cluster_means.max() - cluster_means.min() + 1e-9)

fig3 = go.Figure(data=go.Heatmap(
    z=cluster_means_norm.values,
    x=FEATURE_NAMES,
    y=[f"Cluster {k}" for k in range(K_MEANS_K)],
    colorscale="Plasma",
    colorbar=dict(title="Normalised Mean", tickfont=dict(color=TEXT_COLOR)),
    text=np.round(cluster_means.values, 2),
    texttemplate="%{text}",
    textfont=dict(size=9, color="white"),
))

fig3.update_layout(
    title=f"Fig 3 — K-Means (K={K_MEANS_K}) Cluster Feature Profiles",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis=dict(tickangle=40),
    height=350, width=900,
)
f3_path = os.path.join(OUTPUT_DIR, "fig3_kmeans_profiles.html")
fig3.write_html(f3_path)
print(f"  Saved: {f3_path}")


# -----------------------------------------------------------------------------
# Fig 4: Distribution of P(Class=1) across all leaves (histogram)
# -----------------------------------------------------------------------------
fig4 = go.Figure(data=go.Histogram(
    x=leaf_df["p_class1"],
    nbinsx=20,
    marker_color=ACCENT1,
    opacity=0.8,
))
fig4.add_vline(x=0.5, line=dict(color=ACCENT2, dash="dash", width=2),
               annotation_text="Chance", annotation_font_color=TEXT_COLOR)

fig4.update_layout(
    title="Fig 4 — Distribution of P(Class=1) Across All Terminal Nodes",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="P(Class=1) in Leaf",
    yaxis_title="Count of Leaves",
    height=420, width=700,
)
f4_path = os.path.join(OUTPUT_DIR, "fig4_leaf_prob_distribution.html")
fig4.write_html(f4_path)
print(f"  Saved: {f4_path}")


# -----------------------------------------------------------------------------
# Fig 5: K-means vs Tree leaf membership comparison
#   Show the proportion of each K-means cluster in each top leaf
# -----------------------------------------------------------------------------
fig5 = go.Figure()

leaf_labels_top = []
for rank, (_, row) in enumerate(top_leaves.iterrows(), start=1):
    lid  = int(row["leaf_id"])
    mask = (leaf_ids_real == lid)
    km_in_leaf = km_labels[mask]
    counts = np.bincount(km_in_leaf, minlength=K_MEANS_K)
    prop   = counts / counts.sum() if counts.sum() > 0 else counts
    leaf_labels_top.append(f"Leaf #{rank}")

    for k in range(K_MEANS_K):
        if rank == 1:
            fig5.add_trace(go.Bar(
                name=f"K-means Cluster {k}",
                x=[f"Leaf #{rank}"],
                y=[prop[k]],
                marker_color=COLORS[k],
            ))
        else:
            fig5.data[k].x = fig5.data[k].x + (f"Leaf #{rank}",)
            fig5.data[k].y = fig5.data[k].y + (prop[k],)

fig5.update_layout(
    barmode="stack",
    title="Fig 5 — K-Means Cluster Composition in Top Tree Leaves",
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PAPER_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR),
    xaxis_title="Terminal Node",
    yaxis_title="Proportion of K-means Cluster",
    height=450, width=700,
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="white", borderwidth=1),
)
f5_path = os.path.join(OUTPUT_DIR, "fig5_kmeans_vs_tree_leaves.html")
fig5.write_html(f5_path)
print(f"  Saved: {f5_path}")


# =============================================================================
#  PHASE 7: SUMMARY REPORT (text)
# =============================================================================
print(f"\n{'='*70}")
print("  PHASE 7: SUMMARY REPORT")
print(f"{'='*70}")

print("""
METHODOLOGY RECAP
-----------------
  • Real data: N=8993 SF Bay Area demographic survey records (14 features).
  • Reference: same size, each feature permuted independently (null model).
  • Tree trained to separate real (class=1) from reference (class=0).
  • Terminal nodes with P(class=1) >> 0.5 identify dense real-data clusters.

COMPARISON FRAMEWORK
--------------------
  METHOD          | What it finds
  ----------------┼--------------------------------------------------------
  Classification  | Dense regions in the joint feature space vs. a null
  Tree (Ex 14.4)  | distribution.  Each leaf = a hyper-rectangle rule.
  ----------------┼--------------------------------------------------------
  PRIM (Table 14.1| Patient Rule Induction: finds high-density "bumps"
  Association     | in feature space via bump hunting.  More flexible than
  Rules)          | axis-aligned rectangles; iteratively refines support.
  ----------------┼--------------------------------------------------------
  K-means (K=3)  | Partitions all real data into K Voronoi regions.
                  | Finds spherical clusters; does NOT use a null reference.
""")

print("TOP TERMINAL NODES (Classification Tree):")
print(f"{'-'*70}")
for rank, (_, row) in enumerate(top_leaves.iterrows(), start=1):
    lid = int(row["leaf_id"])
    print(f"\n  [{rank}] Leaf {lid}  |  P(Class=1) = {row['p_class1']:.4f}  "
          f"|  Support = {row['support']*100:.2f}%  "
          f"|  n_real = {int(row['n_real'])}")
    for r in leaf_rules[lid]:
        thr_int = int(round(r["threshold"]))
        feat    = r["feature"]
        vmap    = VALUE_MAPS.get(feat, {})
        readable = vmap.get(thr_int, str(thr_int))
        print(f"       {feat} {r['direction']} {r['threshold']:.2f}  (≈ {readable})")

print(f"\n{'='*70}")
print("  PIPELINE COMPLETE — outputs saved to:", OUTPUT_DIR)
print(f"{'='*70}\n")
