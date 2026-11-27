"""
Real-dataset validation for the retrieval system (+OceanTest/+OceanNLP/+Retrieval).

Dataset: Cranfield (via ir_datasets) -- the classic IR test collection: 1400 real
aerospace-engineering abstracts, 225 real queries, real human relevance judgments.
Not synthetic, not a toy example -- this is literally one of the standard benchmarks
the IR research community has used since the 1960s.

Production references used:
  - scikit-learn CountVectorizer/TfidfTransformer/cosine_similarity (TF-IDF + cosine
    ranking) -- already validated bit-for-bit against OceanStats earlier in this
    project; reused here on REAL data as an extra layer of confidence.
  - A direct numpy reproduction of the Elasticsearch/Lucene BM25 formula (K1=1.2,
    B=0.75) -- the specific variant OceanStats.Data.Weighting.BM25Weighting
    implements, same methodology as the original TF-IDF/BM25 validation.
  - Independently-implemented Precision@K/Recall@K/AveragePrecision/ReciprocalRank/
    NDCG@K in numpy, matching the exact TREC-standard formulas AND the exact
    edge-case conventions OceanStats.Metrics.Retrieval's classes document (P@K
    denominator is always K; Recall/AP/RR are NaN when zero relevant docs are
    judged at all, 0 when relevant docs exist but none were retrieved; NDCG is 0
    when IDCG is 0). pytrec_eval (the actual TREC C tool wrapper) could not be
    installed in this environment (needs a C++ compiler not present) -- this numpy
    fallback is the same approach already used successfully for BM25 earlier in
    this project when a packaged reference wasn't practical to install.

CRANFIELD RELEVANCE SCALE -- READ BEFORE TRUSTING ANY NDCG NUMBER BELOW:
  Cranfield's qrels use 1=most relevant ... 4=least (but still relevant), -1=not
  relevant. This is INVERTED relative to the modern convention (and relative to
  what OceanStats.Metrics.Retrieval.NDCGAtK assumes: higher grade = more relevant).
  Converted here via relevance = 5 - cranfield_value for judged-relevant docs
  (1->4, 2->3, 3->2, 4->1), and -1 -> 0. Docs never judged for a query are treated
  as 0 (unjudged = not relevant), standard practice for an incomplete judgment pool.
"""
import numpy as np
import ir_datasets
from scipy.io import savemat
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

K = 10  # cutoff for Precision@K / Recall@K / NDCG@K, standard IR convention
N_QUERIES = 25  # a manageable, still-real subset of Cranfield's 225 queries

# --------------------------------------------------------------------------- #
# 1. Load the real dataset.
# --------------------------------------------------------------------------- #
print("Loading Cranfield (real IR test collection, 1400 docs / 225 queries)...")
ds = ir_datasets.load("cranfield")

docs = list(ds.docs_iter())
doc_id_to_idx = {d.doc_id: i for i, d in enumerate(docs)}  # 0-indexed Python position
doc_texts = [d.text for d in docs]
N = len(docs)
print(f"Corpus: {N} real documents")

all_queries = list(ds.queries_iter())[:N_QUERIES]
query_ids = [q.query_id for q in all_queries]
query_texts = [q.text for q in all_queries]
print(f"Using {len(all_queries)} real queries (of 225 total)")

# Relevance vectors: (n_queries x N), converted per the scale note above.
relevance = np.zeros((len(all_queries), N), dtype=np.float64)
qrels_by_query = {}
for qrel in ds.qrels_iter():
    qrels_by_query.setdefault(qrel.query_id, []).append(qrel)

for qi, qid in enumerate(query_ids):
    for qrel in qrels_by_query.get(qid, []):
        if qrel.doc_id not in doc_id_to_idx:
            continue
        di = doc_id_to_idx[qrel.doc_id]
        cf = qrel.relevance
        relevance[qi, di] = 0.0 if cf < 0 else (5 - cf)

n_judged_relevant = (relevance > 0).sum(axis=1)
print(f"Judged-relevant docs per query (of these {len(all_queries)}): "
      f"min={n_judged_relevant.min()}, max={n_judged_relevant.max()}, "
      f"mean={n_judged_relevant.mean():.1f}")

# --------------------------------------------------------------------------- #
# 2. Production tokenization + counts (scikit-learn), TF-IDF, over the FULL
#    real 1400-document corpus.
# --------------------------------------------------------------------------- #
count_vec = CountVectorizer(lowercase=True, stop_words="english", max_features=2000)
X_counts = count_vec.fit_transform(doc_texts)
vocab = count_vec.get_feature_names_out().tolist()
V = len(vocab)
X_counts_d = X_counts.toarray().astype(np.float64)
print(f"Vocabulary: {V} terms, count matrix {X_counts.shape}, "
      f"nnz={X_counts.nnz} ({100*X_counts.nnz/(N*V):.2f}% dense)")

tfidf_transformer = TfidfTransformer(norm="l2", smooth_idf=True, sublinear_tf=False)
X_tfidf_sklearn = tfidf_transformer.fit_transform(X_counts)

# --------------------------------------------------------------------------- #
# 3. BM25 (Lucene/Elasticsearch formula, K1=1.2 B=0.75) -- numpy reference,
#    same methodology as the original TF-IDF/BM25 validation.
# --------------------------------------------------------------------------- #
K1, B = 1.2, 0.75
df = (X_counts_d > 0).sum(axis=0)
bm25_idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
doc_len = X_counts_d.sum(axis=1)
avgdl = doc_len.mean()
denom = K1 * (1 - B + B * doc_len / avgdl)
bm25_weights = bm25_idf[None, :] * (X_counts_d * (K1 + 1) / (X_counts_d + denom[:, None]))
bm25_weights[X_counts_d == 0] = 0.0

# --------------------------------------------------------------------------- #
# 4. Standard TREC metric formulas, independently implemented (matching
#    OceanStats.Metrics.Retrieval's exact edge-case conventions).
# --------------------------------------------------------------------------- #
def precision_at_k(ranked_idx, rel, k):
    topk = ranked_idx[:k]
    hits = np.count_nonzero(rel[topk] > 0)
    return hits / k

def recall_at_k(ranked_idx, rel, k):
    total_rel = np.count_nonzero(rel > 0)
    if total_rel == 0:
        return np.nan
    topk = ranked_idx[:k]
    hits = np.count_nonzero(rel[topk] > 0)
    return hits / total_rel

def average_precision(ranked_idx, rel):
    total_rel = np.count_nonzero(rel > 0)
    if total_rel == 0:
        return np.nan
    hits, prec_sum = 0, 0.0
    for rank, di in enumerate(ranked_idx, start=1):
        if rel[di] > 0:
            hits += 1
            prec_sum += hits / rank
    return prec_sum / total_rel

def reciprocal_rank(ranked_idx, rel):
    total_rel = np.count_nonzero(rel > 0)
    if total_rel == 0:
        return np.nan
    for rank, di in enumerate(ranked_idx, start=1):
        if rel[di] > 0:
            return 1.0 / rank
    return 0.0  # relevant docs exist but none retrieved -- well-defined 0, not NaN

def ndcg_at_k(ranked_idx, rel, k):
    def dcg(rels_in_order):
        rels_in_order = np.asarray(rels_in_order, dtype=np.float64)[:k]
        gains = (2.0 ** rels_in_order) - 1.0
        discounts = np.log2(np.arange(2, len(rels_in_order) + 2))
        return np.sum(gains / discounts)
    actual = dcg(rel[ranked_idx[:k]])
    ideal = dcg(np.sort(rel)[::-1])
    return 0.0 if ideal == 0 else actual / ideal

METRIC_FNS = {
    "precision@k": lambda ridx, rel: precision_at_k(ridx, rel, K),
    "recall@k": lambda ridx, rel: recall_at_k(ridx, rel, K),
    "ap": lambda ridx, rel: average_precision(ridx, rel),
    "rr": lambda ridx, rel: reciprocal_rank(ridx, rel),
    "ndcg@k": lambda ridx, rel: ndcg_at_k(ridx, rel, K),
}

# --------------------------------------------------------------------------- #
# 5. Rank each real query with both BM25 and TF-IDF+cosine, score with all 5
#    metrics against the REAL relevance judgments.
# --------------------------------------------------------------------------- #
query_counts_all = np.zeros((len(all_queries), V), dtype=np.float64)
bm25_ranked_idx_all = np.zeros((len(all_queries), N), dtype=np.int64)
bm25_scores_all = np.zeros((len(all_queries), N), dtype=np.float64)
tfidf_ranked_idx_all = np.zeros((len(all_queries), N), dtype=np.int64)
tfidf_scores_all = np.zeros((len(all_queries), N), dtype=np.float64)

metric_names = list(METRIC_FNS.keys())
bm25_metric_vals = np.zeros((len(all_queries), len(metric_names)))
tfidf_metric_vals = np.zeros((len(all_queries), len(metric_names)))

for qi, qtext in enumerate(query_texts):
    qcounts = count_vec.transform([qtext]).toarray().ravel().astype(np.float64)
    query_counts_all[qi] = qcounts

    bm25_scores = bm25_weights @ qcounts
    bm25_order = np.argsort(-bm25_scores, kind="stable")
    bm25_ranked_idx_all[qi] = bm25_order
    bm25_scores_all[qi] = bm25_scores

    qtfidf = tfidf_transformer.transform(count_vec.transform([qtext])).toarray()
    tfidf_scores = cosine_similarity(qtfidf, X_tfidf_sklearn).ravel()
    tfidf_order = np.argsort(-tfidf_scores, kind="stable")
    tfidf_ranked_idx_all[qi] = tfidf_order
    tfidf_scores_all[qi] = tfidf_scores

    rel = relevance[qi]
    for mi, mname in enumerate(metric_names):
        bm25_metric_vals[qi, mi] = METRIC_FNS[mname](bm25_order, rel)
        tfidf_metric_vals[qi, mi] = METRIC_FNS[mname](tfidf_order, rel)

print("\n--- Mean metrics over", len(all_queries), "real Cranfield queries ---")
print(f"{'metric':12s} {'BM25':>10s} {'TF-IDF+cos':>12s}")
for mi, mname in enumerate(metric_names):
    bm25_mean = np.nanmean(bm25_metric_vals[:, mi])
    tfidf_mean = np.nanmean(tfidf_metric_vals[:, mi])
    print(f"{mname:12s} {bm25_mean:10.4f} {tfidf_mean:12.4f}")

# --------------------------------------------------------------------------- #
# 6. Save everything MATLAB needs -- identical inputs on both sides, so any
#    mismatch means an algorithm/formula difference, not a preprocessing one.
# --------------------------------------------------------------------------- #
out_path = "cranfield_validation_data.mat"
savemat(out_path, {
    "X_counts": sparse.csr_matrix(X_counts_d),
    "vocab": np.array(vocab, dtype=object),
    "N": N, "V": V, "K": K,
    "query_ids": np.array(query_ids, dtype=object),
    "query_texts": np.array(query_texts, dtype=object),
    "query_counts": query_counts_all,
    "relevance": relevance,
    "n_judged_relevant": n_judged_relevant,
    "bm25_K1": K1, "bm25_B": B,
    "bm25_weights": bm25_weights,
    "bm25_ranked_idx": bm25_ranked_idx_all + 1,  # +1: MATLAB 1-indexing
    "bm25_scores": bm25_scores_all,
    "sklearn_idf": tfidf_transformer.idf_,
    "sklearn_tfidf": X_tfidf_sklearn.toarray(),
    "tfidf_ranked_idx": tfidf_ranked_idx_all + 1,
    "tfidf_scores": tfidf_scores_all,
    "metric_names": np.array(metric_names, dtype=object),
    "bm25_metric_vals": bm25_metric_vals,
    "tfidf_metric_vals": tfidf_metric_vals,
})
print(f"\nSaved {out_path}")
