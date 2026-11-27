"""
Exports the three neural-retrieval checkpoints (neural-retrieval addendum §4,
§7) to <tag>_weights.mat + <tag>_vocab.txt in the schema documented in
+OceanNLP/+Embedding/{BertBackbone,CrossEncoder,ColBERTEncoder}.m:

  bienc   sentence-transformers/all-MiniLM-L6-v2        (bare "embeddings...." keys)
  cross   cross-encoder/ms-marco-TinyBERT-L-2-v2        ("bert." prefix + classifier)
  colbert answerdotai/answerai-colbert-small-v1         ("bert." prefix + linear.weight)

Tensors are read straight from model.safetensors (no AutoModel: the ColBERT
repo is a custom HF_ColBERT architecture). NO-TRANSPOSE RULE: every matrix is
saved exactly as PyTorch stores it, (out, in) -- .numpy(), never .T. Values
stay float32; MATLAB converts to double on load.

Run from this folder (files are written to the cwd):
    py export_neural_checkpoints.py
Requires: torch-free -- only safetensors, numpy, scipy, huggingface_hub.
"""
import re
import numpy as np
from scipy.io import savemat
from safetensors import safe_open
from huggingface_hub import hf_hub_download

CHECKPOINTS = {
    "bienc":   "sentence-transformers/all-MiniLM-L6-v2",
    "cross":   "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "colbert": "answerdotai/answerai-colbert-small-v1",
}

LAYER_KEYS = {  # MATLAB field -> HF suffix under encoder.layer.{i}.
    "WQ": "attention.self.query.weight",  "bQ": "attention.self.query.bias",
    "WK": "attention.self.key.weight",    "bK": "attention.self.key.bias",
    "WV": "attention.self.value.weight",  "bV": "attention.self.value.bias",
    "WO": "attention.output.dense.weight", "bO": "attention.output.dense.bias",
    "W1": "intermediate.dense.weight",    "b1": "intermediate.dense.bias",
    "W2": "output.dense.weight",          "b2": "output.dense.bias",
    "LN1Gamma": "attention.output.LayerNorm.weight", "LN1Beta": "attention.output.LayerNorm.bias",
    "LN2Gamma": "output.LayerNorm.weight",           "LN2Beta": "output.LayerNorm.bias",
}


def load_tensors(repo):
    path = hf_hub_download(repo, "model.safetensors")
    out = {}
    with safe_open(path, "np") as f:
        for k in f.keys():
            out[re.sub(r"^bert\.", "", k)] = f.get_tensor(k)   # both layouts -> bare keys
    return out


def export(tag, repo):
    T = load_tensors(repo)
    cfg_path = hf_hub_download(repo, "config.json")
    import json
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    n_layers = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    assert cfg.get("hidden_act", "gelu") == "gelu"
    eps = cfg.get("layer_norm_eps", 1e-12)
    assert abs(eps - 1e-12) < 1e-15, f"{repo}: layer_norm_eps={eps}; the MATLAB encoder hardcodes 1e-12"

    out = {
        "N_LAYERS": n_layers, "NUM_HEADS": heads, "D": cfg["hidden_size"],
        "WordEmbeddings": T["embeddings.word_embeddings.weight"],
        "PositionEmbeddings": T["embeddings.position_embeddings.weight"],
        "TokenTypeEmbeddings": T["embeddings.token_type_embeddings.weight"],
        "EmbLNGamma": T["embeddings.LayerNorm.weight"], "EmbLNBeta": T["embeddings.LayerNorm.bias"],
    }
    for field, suffix in LAYER_KEYS.items():
        out["L_" + field] = np.stack([T[f"encoder.layer.{i}.{suffix}"] for i in range(n_layers)])
    if "pooler.dense.weight" in T:
        out["PoolerW"], out["PoolerB"] = T["pooler.dense.weight"], T["pooler.dense.bias"]
    if "classifier.weight" in T:
        out["ClassifierW"], out["ClassifierB"] = T["classifier.weight"], T["classifier.bias"]
    if "linear.weight" in T:
        out["ProjectionW"] = T["linear.weight"]

    # vocab: one token per line in id order (line index = id)
    vocab_path = hf_hub_download(repo, "vocab.txt")
    lines = open(vocab_path, encoding="utf-8").read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    assert len(lines) == T["embeddings.word_embeddings.weight"].shape[0], "vocab size != embedding rows"
    with open(f"{tag}_vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    savemat(f"{tag}_weights.mat", out)
    print(f"{tag:8s} {repo}: {n_layers} layers, d={cfg['hidden_size']}, heads={heads}, "
          f"heads-present={[k for k in ('PoolerW','ClassifierW','ProjectionW') if k in out]}")


if __name__ == "__main__":
    for tag, repo in CHECKPOINTS.items():
        export(tag, repo)
