"""PyTorch references for OceanDL.Transformer (MultiHeadAttention, TransformerBlock, LayerNorm, GELU).

Regenerate: py -3.12 gen_torch_transformer_reference.py
Layout: PyTorch time-major (T, N, D) arrays; the MATLAB tests import them with
NpyReader + DimFormat.toCanonical(.., "TBC") -> D x N x T. Weights are (out x in) as in the repo.
Gradients are of L = sum(Y * R) for the saved projection R.
"""
import os, numpy as np, torch
torch.manual_seed(7)
here = os.path.dirname(os.path.abspath(__file__))
save = lambda n, a: np.save(os.path.join(here, n + ".npy"), a.detach().numpy() if torch.is_tensor(a) else np.asarray(a))
T, N, D, H, FF = 5, 3, 8, 2, 16
x = torch.randn(T, N, D, dtype=torch.float64, requires_grad=True)
R = torch.randn(T, N, D, dtype=torch.float64)
save("x_TND", x); save("R_TND", R)

# --- nn.MultiheadAttention with a key padding mask (True = IGNORE in PyTorch) ---
mha = torch.nn.MultiheadAttention(D, H, dropout=0.0).double()
kpm = torch.tensor([[False]*5, [False, False, False, True, True], [False]*4 + [True]])
y, w = mha(x, x, x, key_padding_mask=kpm, need_weights=True, average_attn_weights=False)   # w: (N, H, Tq, Tk)
(y * R).sum().backward()
save("mha_in_proj_weight", mha.in_proj_weight); save("mha_in_proj_bias", mha.in_proj_bias)
save("mha_out_weight", mha.out_proj.weight); save("mha_out_bias", mha.out_proj.bias)
save("mha_key_padding_mask", kpm.numpy().astype(np.uint8))
save("mha_y_TND", y); save("mha_w_NHQK", w); save("mha_dx_TND", x.grad)
save("mha_dWin", mha.in_proj_weight.grad)

# --- nn.TransformerEncoderLayer, Pre-LN and Post-LN, GELU ---
for tag, first in [("pre", True), ("post", False)]:
    x.grad = None
    layer = torch.nn.TransformerEncoderLayer(D, H, FF, dropout=0.0, activation="gelu", norm_first=first).double()
    y = layer(x)
    (y * R).sum().backward()
    p = dict(layer.named_parameters())
    for k in ["self_attn.in_proj_weight", "self_attn.in_proj_bias", "self_attn.out_proj.weight", "self_attn.out_proj.bias",
              "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias",
              "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"]:
        save(f"enc_{tag}_{k.replace('.', '_')}", p[k])
    save(f"enc_{tag}_y_TND", y); save(f"enc_{tag}_dx_TND", x.grad)
    save(f"enc_{tag}_dW1", p["linear1.weight"].grad); save(f"enc_{tag}_dnorm1", p["norm1.weight"].grad)
print("transformer fixtures written, torch", torch.__version__)
