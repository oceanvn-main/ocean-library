"""
Fetches a real public dataset (20 Newsgroups, via sklearn's network fetcher) and computes
TF-IDF, cosine similarity, and BM25 using the production Python libraries a workplace would
actually reach for: scikit-learn (TF-IDF, cosine similarity) and rank_bm25 (classic Okapi
BM25), plus a direct numpy reproduction of the Elasticsearch/Lucene non-negative-idf BM25
variant -- the specific formula OceanStats.Data.Weighting.BM25Weighting implements (its own
docstring names this variant explicitly; rank_bm25's classic Robertson formula is a
different, well-known BM25 variant and is expected to diverge -- included for context, not
as the ground truth for our implementation).

Output: nlp_validation_data.mat, loaded by compare_in_matlab.m to run the exact same counts
through OceanStats/OceanNLP and diff the results.
"""
# %% Setup & Imports
import numpy as np
from scipy.io import savemat
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

RNG = np.random.RandomState(0)

# %%
# ---------------------------------------------------------------------------
# 1. Real dataset from the internet (sklearn downloads this over the network
#    on first call -- this is not a bundled toy dataset).
# ---------------------------------------------------------------------------
CATEGORIES = ["sci.space", "rec.autos", "comp.graphics", "talk.religion.misc"]
DOCS_PER_CATEGORY = 15

print("Fetching 20 Newsgroups (network fetch)...")
raw = fetch_20newsgroups(
    subset="train",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    shuffle=True,
    random_state=0,
)

# Take a small, balanced subset per category so the corpus stays fast to
# round-trip through MATLAB and easy to sanity-check by eye.
docs, labels = [], []
for cat_idx, cat_name in enumerate(CATEGORIES):
    idx = [i for i, t in enumerate(raw.target) if raw.target_names[t] == cat_name]
    idx = idx[:DOCS_PER_CATEGORY]
    for i in idx:
        text = raw.data[i].strip()
        if len(text) > 20:  # drop near-empty documents
            docs.append(text)
            labels.append(cat_name)

print(f"Corpus: {len(docs)} documents across {len(set(labels))} categories "
      f"({CATEGORIES})")

# %%
# ---------------------------------------------------------------------------
# 2. Production tokenization + raw counts (scikit-learn CountVectorizer).
#    This matrix is what gets handed to MATLAB -- comparisons are on the
#    WEIGHTING math, not on tokenization, so both sides start from identical
#    counts.
# ---------------------------------------------------------------------------
count_vec = CountVectorizer(lowercase=True, stop_words="english", max_features=300)
X_counts = count_vec.fit_transform(docs)  # sparse (N x V) int64
vocab = count_vec.get_feature_names_out().tolist()
N, V = X_counts.shape
print(f"Vocabulary size: {V}, count matrix shape: {X_counts.shape}, "
      f"nnz: {X_counts.nnz} ({100*X_counts.nnz/(N*V):.2f}% dense)")

X_counts_d = X_counts.toarray().astype(np.float64)

# %%
# ---------------------------------------------------------------------------
# 3. Production TF-IDF (scikit-learn TfidfTransformer -- smooth_idf=True is
#    the exact formula OceanStats.Data.Weighting.TfidfWeighting documents
#    itself as matching: idf_t = ln((1+N)/(1+df_t)) + 1).
# ---------------------------------------------------------------------------
tfidf_transformer = TfidfTransformer(norm="l2", smooth_idf=True, sublinear_tf=False)
X_tfidf_sklearn = tfidf_transformer.fit_transform(X_counts)
sklearn_idf = tfidf_transformer.idf_  # (V,) -- ground truth idf vector

# %%
# ---------------------------------------------------------------------------
# 4. Production cosine similarity on the TF-IDF vectors (scikit-learn).
# ---------------------------------------------------------------------------
sklearn_cosine = cosine_similarity(X_tfidf_sklearn)  # dense (N x N)

# %%
# ---------------------------------------------------------------------------
# 5. Stratified Cross-Validation Loop to Find Optimal (k1, b)
# ---------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import os
import sys

cv_cache_file = 'results/cv_results.npz'
force_cv = '--force-cv' in sys.argv

os.makedirs('results', exist_ok=True)

if os.path.exists(cv_cache_file) and not force_cv:
    print(f"\n--- Loading cached CV results from {cv_cache_file} (use --force-cv to rerun) ---")
    with np.load(cv_cache_file) as data:
        k1_vals = data['k1_vals']
        b_vals = data['b_vals']
        grid_scores = data['grid_scores']
        best_k1 = data['best_k1']
        best_b = data['best_b']
        best_prec = data['best_prec']
    # Extract scalars from 0-d arrays
    best_k1 = float(best_k1)
    best_b = float(best_b)
    best_prec = float(best_prec)
    print(f"Loaded Optimal Parameters: k1 = {best_k1:.2f}, b = {best_b:.2f} | Best Validation Precision@5: {best_prec:.4f}")
else:
    print("\n--- Running 5-Fold Cross-Validation for BM25 Hyperparameters (k1, b) ---")

    k1_vals = np.linspace(0.2, 3.0, 15)
    b_vals = np.linspace(0.0, 1.0, 11)
    grid_scores = np.zeros((len(k1_vals), len(b_vals)))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels_arr = np.array(labels)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_counts_d, labels_arr)):
        X_train = X_counts_d[train_idx]
        X_val = X_counts_d[val_idx]
        y_train = labels_arr[train_idx]
        y_val = labels_arr[val_idx]
        
        N_tr = X_train.shape[0]
        df_tr = (X_train > 0).sum(axis=0)
        bm25_idf_tr = np.log((N_tr - df_tr + 0.5) / (df_tr + 0.5) + 1.0)
        doc_len_tr = X_train.sum(axis=1)
        avgdl_tr = doc_len_tr.mean()
        
        for i, k1 in enumerate(k1_vals):
            for j, b in enumerate(b_vals):
                denom_tr = k1 * (1 - b + b * doc_len_tr / avgdl_tr)
                W_tr = bm25_idf_tr[None, :] * (X_train * (k1 + 1) / (X_train + denom_tr[:, None]))
                W_tr[X_train == 0] = 0.0
                
                # Score validation docs against train set: (N_val x N_train)
                val_scores = X_val @ W_tr.T
                
                # Top-5 Precision: fraction of top-5 retrieved train docs matching val label
                top5_indices = np.argsort(-val_scores, axis=1)[:, :5]
                top5_labels = y_train[top5_indices]
                matches = (top5_labels == y_val[:, None]).mean()
                
                grid_scores[i, j] += matches

    grid_scores /= skf.n_splits

    best_i, best_j = np.unravel_index(np.argmax(grid_scores), grid_scores.shape)
    best_k1 = k1_vals[best_i]
    best_b = b_vals[best_j]
    best_prec = grid_scores[best_i, best_j]

    print(f"Optimal Parameters found: k1 = {best_k1:.2f}, b = {best_b:.2f} | Best Validation Precision@5: {best_prec:.4f}")

    # Save results to cache
    np.savez(cv_cache_file, k1_vals=k1_vals, b_vals=b_vals, grid_scores=grid_scores, 
             best_k1=best_k1, best_b=best_b, best_prec=best_prec)
    print(f"Saved CV results to '{cv_cache_file}'")

    # Plotting heatmap
    plt.figure(figsize=(9, 7))
    im = plt.imshow(grid_scores.T, origin='lower', aspect='auto',
                    extent=[k1_vals[0], k1_vals[-1], b_vals[0], b_vals[-1]],
                    cmap='viridis')
    plt.colorbar(im, label='Mean Precision@5 Score')
    plt.plot(best_k1, best_b, 'r*', markersize=15, label=f'Optimal (k1={best_k1:.2f}, b={best_b:.2f})')
    plt.xlabel('k1 (Term Frequency Saturation)')
    plt.ylabel('b (Document Length Normalization)')
    plt.title('BM25 Hyperparameter Tuning (5-Fold Cross-Validation on 20 Newsgroups)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('results/bm25_cv_grid_search.png', dpi=150)
    plt.show()
    print("Saved cross-validation heatmap plot to 'results/bm25_cv_grid_search.png'")

# %%
# ---------------------------------------------------------------------------
# 6a. BM25 -- production library rank_bm25 (classic Robertson/Okapi formula)
#     Uses hyperparams learned from CV. Shown for CONTEXT: this uses a 
#     DIFFERENT idf formula (allows negative idf, patched with an 
#     epsilon*avg_idf floor) than the Lucene/Elasticsearch variant our 
#     class implements. A mismatch here is EXPECTED and does not indicate 
#     a bug in OceanStats.
# ---------------------------------------------------------------------------
tokenized_docs = [count_vec.build_analyzer()(d) for d in docs]
#k - term frequency saruration, b- Document Length normalization
bm25_rank = BM25Okapi(tokenized_docs, k1=best_k1, b=best_b)

QUERY_TEXT = "space shuttle orbit engine"
query_tokens = count_vec.build_analyzer()(QUERY_TEXT)
rank_bm25_query_scores = np.array(bm25_rank.get_scores(query_tokens))

# %%
# ---------------------------------------------------------------------------
# 6b. BM25 -- direct numpy reproduction of the Elasticsearch/Lucene
#     NON-NEGATIVE idf variant. Uses hyperparams learned from CV.
#     This IS the ground truth for validating our implementation.
# ---------------------------------------------------------------------------
K1, B = best_k1, best_b
df = (X_counts_d > 0).sum(axis=0)                       # (V,)
bm25_idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)     # (V,) Lucene/ES formula
doc_len = X_counts_d.sum(axis=1)                          # (N,)
avgdl = doc_len.mean()
denom = K1 * (1 - B + B * doc_len / avgdl)                # (N,)
bm25_lucene_manual = bm25_idf[None, :] * (
    X_counts_d * (K1 + 1) / (X_counts_d + denom[:, None])
)
bm25_lucene_manual[X_counts_d == 0] = 0.0  # no weight where term absent

# Query scoring: encode the query against the SAME vocabulary (OOV terms get
# no column -- CountVectorizer's transform silently drops unseen terms,
# matching how a fixed-vocabulary production index would behave for terms
# never indexed).
query_counts = count_vec.transform([QUERY_TEXT]).toarray().ravel().astype(np.float64)
bm25_lucene_query_scores = bm25_lucene_manual @ query_counts

print("\n--- Reference values computed ---")
print(f"sklearn TF-IDF matrix shape: {X_tfidf_sklearn.shape}")
print(f"sklearn cosine similarity matrix shape: {sklearn_cosine.shape}")
print(f"Lucene-formula BM25 weight matrix shape: {bm25_lucene_manual.shape}")
print(f"Query: {QUERY_TEXT!r} -> tokens {query_tokens}")
print(f"rank_bm25 (classic, k1={best_k1:.2f}) top-3 doc idx: "
      f"{np.argsort(-rank_bm25_query_scores)[:3]}")
print(f"Lucene-formula (k1={K1:.2f}) top-3 doc idx: "
      f"{np.argsort(-bm25_lucene_query_scores)[:3]}")

# %%
# ---------------------------------------------------------------------------
# 7. Save everything MATLAB needs -- one round trip, no re-tokenization.
# ---------------------------------------------------------------------------
out_path = "results/nlp_validation_data.mat"
savemat(out_path, {
    "X_counts": sparse.csr_matrix(X_counts_d),
    "docs": np.array(docs, dtype=object),
    "vocab": np.array(vocab, dtype=object),
    "labels": np.array(labels, dtype=object),
    "N": N, "V": V,
    "sklearn_idf": sklearn_idf,
    "sklearn_tfidf": X_tfidf_sklearn.toarray(),
    "sklearn_cosine": sklearn_cosine,
    "bm25_lucene_idf": bm25_idf,
    "bm25_lucene_manual": bm25_lucene_manual,
    "bm25_avgdl": avgdl,
    "K1": K1, "B": B,
    "query_text": QUERY_TEXT,
    "query_counts": query_counts,
    "bm25_lucene_query_scores": bm25_lucene_query_scores,
    "rank_bm25_query_scores": rank_bm25_query_scores,
    "cv_best_k1": best_k1,
    "cv_best_b": best_b,
    "cv_grid_scores": grid_scores,
    "k1_vals": k1_vals,
    "b_vals": b_vals,
})
print(f"\nSaved {out_path}")
