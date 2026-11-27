"""PyTorch references for OceanDL.Transformer.DecoderBlock and ConformerBlock.

Regenerate: ./.venv/Scripts/python.exe gen_torch_decoder_conformer_reference.py   (torch 2.6, float64)
Layout: time-major (T, N, D) arrays; the MATLAB test imports them with NpyReader +
DimFormat.toCanonical(.., "TBC") -> D x N x T. Weights (out x in), conv weights (Cout, Cin/groups, K).
Gradients are of L = sum(Y * R) for the saved projection R.

DecoderBlock reference = torch.nn.TransformerDecoderLayer(norm_first=True, activation=relu, dropout=0),
tgt_mask = causal, memory_key_padding_mask (True = IGNORE in PyTorch).

ConformerBlock reference = an independent torch implementation of Gulati et al. 2020 Fig. 1
(no torchaudio in the venv) with the two documented Ocean choices: LayerNorm after the
depthwise conv (instead of BatchNorm) and rotary position embeddings (GPT-NeoX "half" pairing)
on the per-head queries/keys; key padding mask; padded frames zeroed before the depthwise conv.
"""
import os
import numpy as np
import torch
import torch.nn.functional as Fn

torch.manual_seed(11)
here = os.path.dirname(os.path.abspath(__file__))


def save(n, a):
    np.save(os.path.join(here, n + ".npy"), a.detach().numpy() if torch.is_tensor(a) else np.asarray(a))


# ---------------------------------------------------------------- decoder layer
U, T, N, D, H, FF = 4, 6, 2, 8, 2, 16
x = torch.randn(U, N, D, dtype=torch.float64, requires_grad=True)
mem = torch.randn(T, N, D, dtype=torch.float64, requires_grad=True)
R = torch.randn(U, N, D, dtype=torch.float64)
mpm = torch.tensor([[False] * 6, [False] * 4 + [True] * 2])          # utterance 2: 4 valid frames
layer = torch.nn.TransformerDecoderLayer(D, H, FF, dropout=0.0, activation="relu", norm_first=True).double()
causal = torch.triu(torch.full((U, U), float("-inf"), dtype=torch.float64), diagonal=1)
y = layer(x, mem, tgt_mask=causal, memory_key_padding_mask=mpm)
(y * R).sum().backward()
save("dec_x_TND", x); save("dec_mem_TND", mem); save("dec_R_TND", R)
save("dec_mem_pad_mask", mpm.numpy().astype(np.uint8))
save("dec_y_TND", y); save("dec_dx_TND", x.grad); save("dec_dmem_TND", mem.grad)
for k, p in layer.named_parameters():
    save("dec_" + k.replace(".", "_"), p)
    save("dec_grad_" + k.replace(".", "_"), p.grad)


# ---------------------------------------------------------------- conformer block
def rope(q):
    # q: (N, h, T, dk); half pairing, theta_j = p * 10000^(-j/(dk/2))
    dk, Tq = q.shape[-1], q.shape[-2]
    f = 10000.0 ** (-torch.arange(dk // 2, dtype=torch.float64) / (dk // 2))
    ang = torch.arange(Tq, dtype=torch.float64)[:, None] * f[None, :]       # T x dk/2
    ang = torch.cat([ang, ang], dim=1)
    rot = torch.cat([-q[..., dk // 2:], q[..., :dk // 2]], dim=-1)
    return q * torch.cos(ang) + rot * torch.sin(ang)


class RefConformer(torch.nn.Module):
    def __init__(s, D, H, k, mult=4):
        super().__init__()
        s.D, s.H = D, H
        s.ln_ff1, s.ln_att, s.ln_conv, s.ln_cnorm, s.ln_ff2, s.ln_out = [torch.nn.LayerNorm(D) for _ in range(6)]
        s.ff1_in, s.ff1_out = torch.nn.Linear(D, mult * D), torch.nn.Linear(mult * D, D)
        s.ff2_in, s.ff2_out = torch.nn.Linear(D, mult * D), torch.nn.Linear(mult * D, D)
        s.q, s.k, s.v, s.o = [torch.nn.Linear(D, D) for _ in range(4)]
        s.pw1 = torch.nn.Conv1d(D, 2 * D, 1)
        s.dw = torch.nn.Conv1d(D, D, k, padding=(k - 1) // 2, groups=D)
        s.pw2 = torch.nn.Conv1d(D, D, 1)
        for i, m in enumerate([s.ln_ff1, s.ln_att, s.ln_conv, s.ln_cnorm, s.ln_ff2, s.ln_out]):
            s.add_module(f"ln{i}", m)

    def forward(s, x, valid):            # x: (T, N, D); valid: (N, T) bool
        Tn, Nn, Dn = x.shape
        ffn = lambda z, a, b: b(Fn.silu(a(z)))
        x = x + 0.5 * ffn(s.ln_ff1(x), s.ff1_in, s.ff1_out)
        z = s.ln_att(x)
        dk = Dn // s.H
        sp = lambda t: t.permute(1, 0, 2).reshape(Nn, Tn, s.H, dk).permute(0, 2, 1, 3)   # N h T dk
        Q, K, V = rope(sp(s.q(z))), rope(sp(s.k(z))), sp(s.v(z))
        S = Q @ K.transpose(-1, -2) / dk ** 0.5
        S = S.masked_fill(~valid[:, None, None, :], float("-inf"))
        O = torch.softmax(S, -1) @ V                                   # N h T dk
        O = O.permute(2, 0, 1, 3).reshape(Tn, Nn, Dn)
        x = x + s.o(O)
        c = s.pw1(s.ln_conv(x).permute(1, 2, 0))                       # N 2D T
        g = c[:, :Dn] * torch.sigmoid(c[:, Dn:])
        g = g * valid[:, None, :]
        c = s.dw(g).permute(2, 0, 1)                                   # T N D
        c = s.pw2(Fn.silu(s.ln_cnorm(c)).permute(1, 2, 0)).permute(2, 0, 1)
        x = x + c
        x = x + 0.5 * ffn(s.ln_ff2(x), s.ff2_in, s.ff2_out)
        return s.ln_out(x) * valid.T[:, :, None]


Tc, Nc, Dc, Hc, Kc = 7, 2, 8, 2, 3
blk = RefConformer(Dc, Hc, Kc).double()
with torch.no_grad():                    # non-trivial LayerNorm affine parameters
    for m in blk.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.weight.add_(0.1 * torch.randn_like(m.weight)); m.bias.add_(0.1 * torch.randn_like(m.bias))
xc = torch.randn(Tc, Nc, Dc, dtype=torch.float64, requires_grad=True)
Rc = torch.randn(Tc, Nc, Dc, dtype=torch.float64)
valid = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
yc = blk(xc, valid)
(yc * Rc).sum().backward()
save("conf_x_TND", xc); save("conf_R_TND", Rc); save("conf_valid", valid.numpy().astype(np.uint8))
save("conf_y_TND", yc); save("conf_dx_TND", xc.grad)
for k, p in blk.named_parameters():
    if k.startswith("ln") and k[2].isdigit():
        continue
    save("conf_" + k.replace(".", "_"), p)
    save("conf_grad_" + k.replace(".", "_"), p.grad)
print("saved decoder + conformer fixtures to", here)
