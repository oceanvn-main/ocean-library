"""
Python reference for the neural retrieval models (addendum §8 b/c/d) on REAL
Cranfield data. Writes neural_reference_data.mat for compare_neural_in_matlab.m.

Data (same 25 real queries / qrels as cranfield_validation_data.mat):
  * ALL 1400 documents are encoded (an earlier round used a 221-doc subset of
    BM25-top-20 + relevant docs; that guaranteed 100% recall for every
    non-BM25 system and inflated their scores, so it was dropped).
  * cross-encoder pairs = each query's BM25 top-20 from the existing reference.

References
  bi-encoder     sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")
                 if importable, else transformers BertModel + mean pool + L2
                 normalize + max_seq_length 256 (what that pipeline is).
  cross-encoder  transformers AutoModelForSequenceClassification logits, on
                 (query, BM25 top-20) pairs, truncation=True (longest_first),
                 max_length=512. Raw logits (MATLAB Activation="identity").
  ColBERT        sentence_transformers.MultiVectorEncoder if importable
                 (ST >= 6); otherwise a from-scratch reference over
                 transformers BertModel token states following the Stanford
                 ColBERT preprocessing in the checkpoint's artifact.metadata
                 (WEAKER: preprocessing is re-implemented, not the vendor's;
                 the source used is recorded in colbert_ref_source).

Run from this folder after export_neural_checkpoints.py and dump_cranfield_texts.py:  py neural_reference.py
"""
import numpy as np
import torch
from scipy.io import loadmat, savemat

NQ = 25                                  # all 25 real queries of the existing reference
TOPK = 20
BI_REPO = "sentence-transformers/all-MiniLM-L6-v2"
CE_REPO = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
CB_REPO = "answerdotai/answerai-colbert-small-v1"
CB_QLEN, CB_DLEN = 32, 300

ref = loadmat("cranfield_validation_data.mat", squeeze_me=False)
query_ids = [str(x[0]) for x in ref["query_ids"][0]][:NQ]
query_texts = [str(x[0]) for x in ref["query_texts"][0]][:NQ]   # exact texts the BM25 reference used
bm25_ranked = ref["bm25_ranked_idx"][:NQ] - 1          # 0-based doc indices, MATLAB export was 1-based
relevance = ref["relevance"][:NQ]

import json
texts = json.load(open("cranfield_texts.json", encoding="utf-8"))   # from dump_cranfield_texts.py
doc_texts = texts["docs"]
print("queries:", NQ, "docs:", len(doc_texts), "min bm25 rank value:", bm25_ranked.min(), "max:", bm25_ranked.max())

top_docs = bm25_ranked[:, :TOPK]
subset = list(range(len(doc_texts)))          # full corpus; name kept for the .mat schema
sub_texts = [doc_texts[i] for i in subset]
print("doc subset size:", len(subset))

# ----------------------------- bi-encoder ---------------------------------
try:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(BI_REPO, device="cpu")
    m.max_seq_length = 256
    bi_source = "sentence_transformers.SentenceTransformer"
    bi_doc = m.encode(sub_texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
    bi_q = m.encode(query_texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
except ImportError:
    from transformers import AutoTokenizer, AutoModel
    bi_source = "transformers BertModel + mean pool + L2 normalize (sentence-transformers not installed)"
    tk, bm = AutoTokenizer.from_pretrained(BI_REPO), AutoModel.from_pretrained(BI_REPO).eval()

    def enc(texts):
        out = []
        for t in texts:
            x = tk(t, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                h = bm(**x).last_hidden_state[0]
            v = h.mean(0)
            out.append((v / v.norm().clamp_min(1e-12)).numpy())
        return np.stack(out)
    bi_doc, bi_q = enc(sub_texts), enc(query_texts)
print("bi-encoder reference:", bi_source, bi_doc.shape)

# ----------------------------- cross-encoder ------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
ctk = AutoTokenizer.from_pretrained(CE_REPO)
cm = AutoModelForSequenceClassification.from_pretrained(CE_REPO).eval()
ce_scores = np.zeros((NQ, TOPK), dtype=np.float32)
with torch.no_grad():
    for qi in range(NQ):
        for j in range(TOPK):
            x = ctk(query_texts[qi], doc_texts[top_docs[qi, j]], truncation=True, max_length=512, return_tensors="pt")
            ce_scores[qi, j] = cm(**x).logits[0, 0].item()
print("cross-encoder reference scored", ce_scores.size, "pairs")

# ----------------------------- ColBERT ------------------------------------
cb_source = None
try:
    from sentence_transformers import MultiVectorEncoder
    cm2 = MultiVectorEncoder(CB_REPO, device="cpu")
    cb_source = "sentence_transformers.MultiVectorEncoder"
    q_tok = [np.asarray(e, dtype=np.float32) for e in cm2.encode_query(query_texts)]
    d_tok = [np.asarray(e, dtype=np.float32) for e in cm2.encode_document(sub_texts)]
except ImportError:
    from transformers import BertModel, BertTokenizerFast
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    cb_source = "MANUAL transformers BertModel + linear.weight, Stanford ColBERT preprocessing per artifact.metadata (sentence-transformers>=6 not installed)"
    ktk = BertTokenizerFast.from_pretrained(CB_REPO)
    kbert = BertModel.from_pretrained(CB_REPO).eval()
    with safe_open(hf_hub_download(CB_REPO, "model.safetensors"), "pt") as f:
        lin = f.get_tensor("linear.weight")                  # (96, 384)
    Q_ID, D_ID = ktk.convert_tokens_to_ids("[unused0]"), ktk.convert_tokens_to_ids("[unused1]")
    MASK_ID, CLS_ID, SEP_ID = ktk.mask_token_id, ktk.cls_token_id, ktk.sep_token_id
    skip = {ktk.convert_tokens_to_ids(c) for c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"}
    print("marker ids:", Q_ID, D_ID, "mask id:", MASK_ID, "skip ids:", len(skip))

    def run(ids, attn):
        with torch.no_grad():
            h = kbert(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([attn])).last_hidden_state[0]
            y = h @ lin.T
            return torch.nn.functional.normalize(y, dim=-1)

    def enc_q(t):
        sub = ktk(t, add_special_tokens=False)["input_ids"][:CB_QLEN - 3]
        core = [CLS_ID, Q_ID] + sub + [SEP_ID]
        pad = CB_QLEN - len(core)
        return run(core + [MASK_ID] * pad, [1] * len(core) + [0] * pad).numpy()   # attend_to_mask_tokens=False

    def enc_d(t):
        sub = ktk(t, add_special_tokens=False)["input_ids"][:CB_DLEN - 3]
        ids = [CLS_ID, D_ID] + sub + [SEP_ID]
        e = run(ids, [1] * len(ids)).numpy()
        return e[[i not in skip for i in ids]]
    q_tok = [enc_q(t) for t in query_texts]
    d_tok = [enc_d(t) for t in sub_texts]
print("ColBERT reference:", cb_source)

maxsim = np.zeros((NQ, len(subset)), dtype=np.float32)
for qi in range(NQ):
    for j in range(len(subset)):
        maxsim[qi, j] = (q_tok[qi] @ d_tok[j].T).max(axis=1).sum()

d_lens = np.array([d.shape[0] for d in d_tok])
savemat("neural_reference_data.mat", {
    "query_ids": np.array(query_ids, dtype=object), "query_texts": np.array(query_texts, dtype=object),
    "subset_idx": np.array(subset) + 1,                       # 1-based for MATLAB
    "sub_texts": np.array(sub_texts, dtype=object),
    "relevance_sub": relevance[:, subset],
    "top_docs": top_docs + 1,                                 # 1-based full-corpus indices
    "top_doc_texts": np.array([[doc_texts[j] for j in row] for row in top_docs], dtype=object),
    "bi_source": bi_source, "bi_doc": bi_doc, "bi_q": bi_q,
    "ce_scores": ce_scores,
    "cb_source": cb_source, "cb_q_tok": np.stack(q_tok) if len({t.shape[0] for t in q_tok}) == 1 else np.array(q_tok, dtype=object),
    "cb_d_tok_concat": np.concatenate(d_tok, axis=0), "cb_d_lens": d_lens,
    "cb_maxsim": maxsim,
})
print("Saved neural_reference_data.mat")
