"""
Exports real prajjwal1/bert-tiny weights + vocab to a .mat file matching the
EXACT schema documented in +OceanNLP/+Embedding/BertEncoder.m (read directly
from that file's header, not from memory), plus computes Python reference
sentence embeddings (mean-pooled, matching BertEncoder's documented pooling
policy) for a handful of real sentences, for layer-for-layer comparison.

KNOWN QUIRK, resolved: prajjwal1/bert-tiny's config.json predates HuggingFace's
"model_type" field requirement for AutoModel/AutoConfig auto-dispatch. Loading
via the concrete BertModel/BertTokenizerFast classes instead of Auto* sidesteps
this entirely (the architecture is already known, no auto-detection needed).

WEIGHTS SCHEMA NOTE: BertEncoder.m expects a MATLAB struct with Weights.Layers
as a (1 x N) STRUCT ARRAY. scipy.io.savemat does not cleanly produce MATLAB
struct arrays from a Python list-of-dicts. Instead, this script saves the
per-layer weights as flat (N x ...) arrays (one leading dimension = layer
index), and a companion MATLAB script (build_weights_from_export.m) assembles
the proper Weights.Layers(i) struct array from them -- a standard, robust
pattern for cross-language struct-array exchange.

NO-TRANSPOSE RULE (load-bearing, see BertEncoder.m for the full explanation):
PyTorch nn.Linear stores weight as (out_features, in_features) and computes
y_row = x_row @ W^T + b. BertEncoder computes the algebraically identical
Q = WQ*X + b in column-vector form -- the SAME (out,in) matrix, left-multiplying
a column vector instead of right-multiplying a transposed row vector. Every
weight matrix below (WQ/WK/WV/WO/W1/W2 and the embedding tables) is saved
EXACTLY as PyTorch stores it -- .detach().numpy(), no .T anywhere.
"""
import numpy as np
import torch
from scipy.io import savemat
from transformers import BertModel, BertTokenizerFast

print("Loading prajjwal1/bert-tiny (BertModel/BertTokenizerFast directly, "
      "bypassing AutoModel's model_type auto-dispatch)...")
model = BertModel.from_pretrained("prajjwal1/bert-tiny")
model.eval()
tok = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")

cfg = model.config
N_LAYERS = cfg.num_hidden_layers
D = cfg.hidden_size
D_FF = cfg.intermediate_size
NUM_HEADS = cfg.num_attention_heads
VOCAB_SIZE = cfg.vocab_size
print(f"Config: {N_LAYERS} layers, d={D}, d_ff={D_FF}, heads={NUM_HEADS}, "
      f"vocab={VOCAB_SIZE}, CLS={tok.cls_token_id}, SEP={tok.sep_token_id}, "
      f"UNK={tok.unk_token_id}")

def npw(t):
    """PyTorch tensor -> numpy, float64, NO TRANSPOSE (see module docstring)."""
    return t.detach().cpu().numpy().astype(np.float64)

# --------------------------------------------------------------------------- #
# 1. Token/position/segment embedding tables -- AS-IS, no transpose.
# --------------------------------------------------------------------------- #
word_emb = npw(model.embeddings.word_embeddings.weight)          # (VocabSize x d)
pos_emb  = npw(model.embeddings.position_embeddings.weight)      # (MaxPositions x d)
seg_emb  = npw(model.embeddings.token_type_embeddings.weight)    # (NumSegmentTypes x d)
# Real BERT applies a LayerNorm right after summing word+position+segment
# embeddings, BEFORE the first transformer block -- confirmed by inspecting
# model.embeddings (a BertEmbeddings module with word/position/token_type
# Embedding + LayerNorm + dropout). This was missing from the original
# hand-rolled TokenEmbedding/BertEncoder design (no ground truth existed to
# catch it until this real-weights comparison) -- exported here now that
# it's been identified as load-bearing.
emb_ln_gamma = npw(model.embeddings.LayerNorm.weight)            # (d,)
emb_ln_beta  = npw(model.embeddings.LayerNorm.bias)              # (d,)
LN_EPS = model.embeddings.LayerNorm.eps                            # 1e-12 for BERT, not LayerNormLayer's 1e-5 default

# --------------------------------------------------------------------------- #
# 2. Per-layer weights, stacked with a leading layer-index dimension.
# --------------------------------------------------------------------------- #
L_WQ = np.zeros((N_LAYERS, D, D)); L_bQ = np.zeros((N_LAYERS, D))
L_WK = np.zeros((N_LAYERS, D, D)); L_bK = np.zeros((N_LAYERS, D))
L_WV = np.zeros((N_LAYERS, D, D)); L_bV = np.zeros((N_LAYERS, D))
L_WO = np.zeros((N_LAYERS, D, D)); L_bO = np.zeros((N_LAYERS, D))
L_W1 = np.zeros((N_LAYERS, D_FF, D)); L_b1 = np.zeros((N_LAYERS, D_FF))
L_W2 = np.zeros((N_LAYERS, D, D_FF)); L_b2 = np.zeros((N_LAYERS, D))
L_LN1Gamma = np.zeros((N_LAYERS, D)); L_LN1Beta = np.zeros((N_LAYERS, D))
L_LN2Gamma = np.zeros((N_LAYERS, D)); L_LN2Beta = np.zeros((N_LAYERS, D))
L_NumHeads = np.full((N_LAYERS,), NUM_HEADS, dtype=np.float64)

for i in range(N_LAYERS):
    layer = model.encoder.layer[i]
    attn = layer.attention.self
    attn_out = layer.attention.output
    L_WQ[i] = npw(attn.query.weight); L_bQ[i] = npw(attn.query.bias)
    L_WK[i] = npw(attn.key.weight);   L_bK[i] = npw(attn.key.bias)
    L_WV[i] = npw(attn.value.weight); L_bV[i] = npw(attn.value.bias)
    L_WO[i] = npw(attn_out.dense.weight); L_bO[i] = npw(attn_out.dense.bias)
    L_LN1Gamma[i] = npw(attn_out.LayerNorm.weight); L_LN1Beta[i] = npw(attn_out.LayerNorm.bias)

    L_W1[i] = npw(layer.intermediate.dense.weight); L_b1[i] = npw(layer.intermediate.dense.bias)
    L_W2[i] = npw(layer.output.dense.weight);        L_b2[i] = npw(layer.output.dense.bias)
    L_LN2Gamma[i] = npw(layer.output.LayerNorm.weight); L_LN2Beta[i] = npw(layer.output.LayerNorm.bias)

print(f"Extracted {N_LAYERS} layers' weights, verified shapes: "
      f"WQ={L_WQ.shape}, W1={L_W1.shape}, W2={L_W2.shape}")

# --------------------------------------------------------------------------- #
# 3. vocab.txt -- one token per line, in id order (0-indexed line = id),
#    matching WordPieceTokenizer's expected format exactly.
# --------------------------------------------------------------------------- #
vocab_dict = tok.get_vocab()  # token -> id
ordered_tokens = [None] * VOCAB_SIZE
for token, idx in vocab_dict.items():
    ordered_tokens[idx] = token
assert all(t is not None for t in ordered_tokens), "vocab has gaps -- would break 0-indexed line mapping"
with open("bert_tiny_vocab.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(ordered_tokens))
print(f"Wrote bert_tiny_vocab.txt ({VOCAB_SIZE} lines, "
      f"line {tok.cls_token_id+1}='{ordered_tokens[tok.cls_token_id]}', "
      f"line {tok.sep_token_id+1}='{ordered_tokens[tok.sep_token_id]}')")

# --------------------------------------------------------------------------- #
# 4. Reference sentence embeddings -- REAL trained weights, mean-pooled over
#    ALL token positions (including [CLS]/[SEP]) of the LAST hidden layer --
#    matching BertEncoder.m's documented pooling policy exactly. This is the
#    number MATLAB's hand-rolled forward pass must reproduce.
# --------------------------------------------------------------------------- #
sentences = [
    "the cat sat on the mat",
    "a dog ran in the park",
    "rockets launch into space",
    "the stock market rose sharply today",
    "she plays the violin beautifully",
]

py_embeddings = np.zeros((len(sentences), D))
py_token_ids = []
with torch.no_grad():
    for i, sent in enumerate(sentences):
        inputs = tok(sent, return_tensors="pt")
        out = model(**inputs)
        last_hidden = out.last_hidden_state[0]          # (T, d)
        pooled = last_hidden.mean(dim=0).numpy()          # mean over ALL T positions
        py_embeddings[i] = pooled
        py_token_ids.append(inputs["input_ids"][0].numpy().tolist())

print("\nReference embeddings computed for", len(sentences), "real sentences.")
for s, ids in zip(sentences, py_token_ids):
    print(f"  {s!r} -> {len(ids)} tokens: {ids}")

# --------------------------------------------------------------------------- #
# 5. Save everything MATLAB needs.
# --------------------------------------------------------------------------- #
savemat("bert_tiny_weights.mat", {
    "N_LAYERS": N_LAYERS, "D": D, "D_FF": D_FF, "NUM_HEADS": NUM_HEADS,
    "VOCAB_SIZE": VOCAB_SIZE,
    "WordEmbeddings": word_emb, "PositionEmbeddings": pos_emb, "TokenTypeEmbeddings": seg_emb,
    "EmbLNGamma": emb_ln_gamma, "EmbLNBeta": emb_ln_beta, "LN_EPS": LN_EPS,
    "L_WQ": L_WQ, "L_bQ": L_bQ, "L_WK": L_WK, "L_bK": L_bK,
    "L_WV": L_WV, "L_bV": L_bV, "L_WO": L_WO, "L_bO": L_bO,
    "L_W1": L_W1, "L_b1": L_b1, "L_W2": L_W2, "L_b2": L_b2,
    "L_LN1Gamma": L_LN1Gamma, "L_LN1Beta": L_LN1Beta,
    "L_LN2Gamma": L_LN2Gamma, "L_LN2Beta": L_LN2Beta,
    "L_NumHeads": L_NumHeads,
    "sentences": np.array(sentences, dtype=object),
    "py_embeddings": py_embeddings,
})
print("\nSaved bert_tiny_weights.mat and bert_tiny_vocab.txt")
