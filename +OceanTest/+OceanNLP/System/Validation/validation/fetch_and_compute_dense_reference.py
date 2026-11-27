"""
Dense (modern) retrieval reference: sentence-transformers on a real 20 Newsgroups subset.
--------------------------------------------------------------------------------------
The dense counterpart of fetch_and_compute_reference.py (sparse counts -> TF-IDF / BM25). This is the workflow people use
today:

    SentenceTransformer("all-MiniLM-L6-v2")            pretrained bi-encoder (BERT + mean pooling + L2 normalisation)
      .encode(docs, normalize_embeddings=True)          dense (N x D) document matrix
      .similarity(...) / util.semantic_search(...)      dense cosine matrix and top-k vector search

Output: results/nlp_dense_validation_data.mat, read by compare_dense_in_matlab.m, which runs OceanNLP's pure-MATLAB
BertEncoder (same checkpoint, exported by +OceanTest/+OceanNLP/+Retrieval/export_neural_checkpoints.py) on the SAME texts and
prints the difference.

A reference must be what it says it is, so there are NO silent fallbacks: if sentence-transformers or the model or the corpus
cannot be loaded the script stops with an error (an earlier version fell back to LSA vectors or a synthetic corpus, which
would have made every comparison meaningless). Everything needed to interpret the numbers is stored in the .mat: model name,
library versions, max_seq_length, pooling, normalisation, per-document token counts and how many documents were truncated.

Requires: pip install sentence-transformers scikit-learn scipy     (the model and the dataset are downloaded on first use)
Run:      py fetch_and_compute_dense_reference.py
"""
import os
import sys

import numpy as np
from scipy.io import savemat

try:
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer, util
except ImportError as err:
    sys.exit(f"Missing dependency ({err}).\nInstall with:  py -m pip install sentence-transformers scikit-learn scipy")
from sklearn.datasets import fetch_20newsgroups

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CATEGORIES = ["sci.space", "rec.autos", "comp.graphics", "talk.religion.misc"]
DOCS_PER_CATEGORY = 15
QUERIES = [
    "space rocket launch",
    "fast sports car engine and brakes",
    "3D graphics rendering and texture mapping",
    "religious belief and moral philosophy",
    "satellite orbit and cosmic radiation",
]
TOP_K = 3
BATCH_SIZE = 32
OUT_PATH = "results/nlp_dense_validation_data.mat"

# --------------------------------------------------------------------------------------------------------------------
# 1. Real corpus (no synthetic fallback)
# --------------------------------------------------------------------------------------------------------------------
print("--- Step 1: 20 Newsgroups subset ---")
raw = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=("headers", "footers", "quotes"),
                         shuffle=True, random_state=0)
docs, labels = [], []
for cat_name in CATEGORIES:
    idx = [i for i, t in enumerate(raw.target) if raw.target_names[t] == cat_name][:DOCS_PER_CATEGORY]
    for i in idx:
        text = raw.data[i].strip()
        if len(text) > 20:                       # drop near-empty documents
            docs.append(text)
            labels.append(cat_name)
if len(docs) < 20:
    sys.exit(f"Corpus too small ({len(docs)} documents): refusing to write a reference.")
print(f"Corpus: {len(docs)} documents across {len(set(labels))} categories")

# --------------------------------------------------------------------------------------------------------------------
# 2. Dense embeddings with the standard library
# --------------------------------------------------------------------------------------------------------------------
print("\n--- Step 2: SentenceTransformer encode ---")
device = "cpu"                                    # deterministic reference; the checkpoint is tiny
model = SentenceTransformer(MODEL_NAME, device=device)
model.eval()
max_seq_length = int(model.max_seq_length)        # 256 for this checkpoint: longer texts are truncated
# the pipeline the library actually applies, read from the model itself (modules.json), not assumed
modules = [type(m).__name__ for m in model]                       # e.g. ['Transformer', 'Pooling', 'Normalize']
pool_module = next((m for m in model if type(m).__name__ == "Pooling"), None)
if pool_module is None:
    sys.exit(f"No Pooling module in {modules}: this is not the bi-encoder pipeline the MATLAB side implements.")
pooling = str(pool_module.get_config_dict().get("pooling_mode", "")) if hasattr(pool_module, "get_config_dict") else \
    str(getattr(pool_module, "pooling_mode", ""))
if pooling != "mean":
    sys.exit(f"Pooling mode is {pooling!r}, but the MATLAB bi-encoder implements mean pooling: not a valid reference.")
has_normalize = "Normalize" in modules
dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
print(f"model={MODEL_NAME}  dim={dim}  max_seq_length={max_seq_length}  modules={modules}  pooling={pooling}")

# how many documents does max_seq_length cut? (token counts INCLUDE [CLS] and [SEP], like the model input)
token_counts = np.array([len(ids) for ids in model.tokenizer(docs, truncation=False)["input_ids"]])
n_truncated = int((token_counts > max_seq_length).sum())
print(f"truncated documents: {n_truncated} of {len(docs)} exceed {max_seq_length} tokens")

X_dense = model.encode(docs, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)
# padding must not change the vectors: re-encode one by one and record the worst difference
X_single = model.encode(docs, batch_size=1, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
batch_invariance = float(np.abs(X_dense - X_single).max())
print(f"embeddings: {X_dense.shape} {X_dense.dtype}; batch-size invariance max abs diff = {batch_invariance:.2e}")

# --------------------------------------------------------------------------------------------------------------------
# 3. Dense cosine similarity and vector search
# --------------------------------------------------------------------------------------------------------------------
print("\n--- Step 3: similarity matrix and top-k search ---")
dense_cosine_matrix = model.similarity(X_dense, X_dense).numpy()          # (N x N); rows are unit norm so cosine = dot
q_dense = model.encode(QUERIES, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)                            # (Q x D)
dense_query_scores = model.similarity(q_dense, X_dense).numpy()            # (Q x N)
hits = util.semantic_search(q_dense, X_dense, top_k=TOP_K)                 # the standard dense-retrieval helper
top_k_idx = np.array([[h["corpus_id"] for h in per_query] for per_query in hits])       # 0-based document indices
top_k_scores = np.array([[h["score"] for h in per_query] for per_query in hits])
for qi, q in enumerate(QUERIES):
    print(f"Query {q!r}")
    for rank in range(TOP_K):
        d = top_k_idx[qi, rank]
        print(f"   {rank + 1}. score {top_k_scores[qi, rank]:.4f}  [{labels[d]}]  {docs[d][:70]!r}")

# --------------------------------------------------------------------------------------------------------------------
# 4. Save for MATLAB (with provenance)
# --------------------------------------------------------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
savemat(OUT_PATH, {
    "X_dense": X_dense.astype(np.float64),                    # (N x D) unit-norm, float32 precision
    "q_dense": q_dense.astype(np.float64),                    # (Q x D)
    "dense_cosine_matrix": dense_cosine_matrix.astype(np.float64),
    "dense_query_scores": dense_query_scores.astype(np.float64),
    "top_k_idx": top_k_idx.astype(np.float64),                # 0-based, as returned by util.semantic_search
    "top_k_scores": top_k_scores.astype(np.float64),
    "docs": np.array(docs, dtype=object),
    "labels": np.array(labels, dtype=object),
    "query_texts": np.array(QUERIES, dtype=object),
    "doc_token_counts": token_counts.astype(np.float64),
    # provenance: what exactly produced these numbers
    "framework": f"sentence-transformers {sentence_transformers.__version__}",
    "model_name": MODEL_NAME,
    "max_seq_length": float(max_seq_length),
    "pooling": pooling,
    "normalize_embeddings": 1.0,                              # encode(..., normalize_embeddings=True);
    "modules": " -> ".join(modules),
    "model_has_normalize_module": float(has_normalize),
    "n_truncated_docs": float(n_truncated),
    "batch_invariance_max_abs": batch_invariance,
    "embedding_dim": float(X_dense.shape[1]),
    "versions": f"torch {torch.__version__}; transformers {transformers.__version__}; numpy {np.__version__}",
})
print(f"\nSaved {OUT_PATH}")
