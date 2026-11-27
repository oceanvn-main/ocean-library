"""PyTorch references for OceanDL generative building blocks.

Regenerate: py -3.12 gen_torch_generative_reference.py
Layout: PyTorch NCHW arrays; the MATLAB tests import them with NpyReader +
DimFormat.toCanonical(.., "BCSS") -> H x W x C x N. Conv weights are torch
(Cout, Cin, KH, KW); the tests permute them to (KH, KW, Cin, Cout).
Gradients are of L = sum(Y * R) for a saved projection R (+ the VQ loss).
"""
import os, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
torch.manual_seed(11)
here = os.path.dirname(os.path.abspath(__file__))
save = lambda n, a: np.save(os.path.join(here, n + ".npy"), a.detach().numpy() if torch.is_tensor(a) else np.asarray(a))
dt = torch.float64

# --- GroupNorm(3 groups, 6 channels) with a non-trivial affine ---
x = torch.randn(2, 6, 4, 5, dtype=dt, requires_grad=True)
gn = nn.GroupNorm(3, 6, eps=1e-5).double()
with torch.no_grad():
    gn.weight.copy_(torch.randn(6, dtype=dt)); gn.bias.copy_(torch.randn(6, dtype=dt))
y = gn(x); R = torch.randn_like(y); (y * R).sum().backward()
save("gn_x", x); save("gn_R", R); save("gn_weight", gn.weight); save("gn_bias", gn.bias)
save("gn_y", y); save("gn_dx", x.grad); save("gn_dweight", gn.weight.grad); save("gn_dbias", gn.bias.grad)

# --- nearest x2 upsampling ---
x = torch.randn(2, 3, 3, 4, dtype=dt, requires_grad=True)
y = F.interpolate(x, scale_factor=2, mode="nearest"); R = torch.randn_like(y); (y * R).sum().backward()
save("up_x", x); save("up_R", R); save("up_y", y); save("up_dx", x.grad)

# --- residual block (GN -> SiLU -> conv3x3 -> GN -> SiLU -> conv3x3, 1x1 skip), then stride-2 conv ---
class ResBlock(nn.Module):
    def __init__(s, cin, cout, groups):
        super().__init__()
        s.norm1 = nn.GroupNorm(groups, cin); s.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        s.norm2 = nn.GroupNorm(groups, cout); s.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        s.skip = nn.Conv2d(cin, cout, 1)
    def forward(s, x):
        h = s.conv1(F.silu(s.norm1(x))); h = s.conv2(F.silu(s.norm2(h)))
        return s.skip(x) + h
blk = ResBlock(4, 6, 2).double()
down = nn.Conv2d(6, 6, 3, stride=2, padding=1).double()
with torch.no_grad():
    for n in ["norm1", "norm2"]:
        m = getattr(blk, n); m.weight.copy_(1 + 0.3 * torch.randn_like(m.weight)); m.bias.copy_(0.3 * torch.randn_like(m.bias))
x = torch.randn(2, 4, 6, 6, dtype=dt, requires_grad=True)
y = down(blk(x)); R = torch.randn_like(y); (y * R).sum().backward()
save("rb_x", x); save("rb_R", R); save("rb_y", y); save("rb_dx", x.grad)
for name, p in list(blk.named_parameters()) + [("down." + k, v) for k, v in down.named_parameters()]:
    save("rb_" + name.replace(".", "_"), p)
save("rb_dconv1_weight", blk.conv1.weight.grad)

# --- VQ-VAE quantiser, gradient-trained codebook, straight-through ---
K, D, beta = 7, 3, 0.25
z = torch.randn(2, D, 3, 2, dtype=dt, requires_grad=True)
E = torch.randn(K, D, dtype=dt, requires_grad=True)
zf = z.permute(0, 2, 3, 1).reshape(-1, D)                                     # rows: N, H, W order
dist = (zf ** 2).sum(1, keepdim=True) - 2 * zf @ E.t() + (E ** 2).sum(1)
idx = dist.argmin(1)
zq = E[idx]
loss = F.mse_loss(zq, zf.detach()) + beta * F.mse_loss(zf, zq.detach())
zst = zf + (zq - zf).detach()
zst = zst.reshape(2, 3, 2, D).permute(0, 3, 1, 2)
R = torch.randn_like(zst)
(loss + (zst * R).sum()).backward()
save("vq_z", z); save("vq_E", E); save("vq_R", R); save("vq_loss", loss.detach().reshape(1))
save("vq_idx_NHW", (idx.reshape(2, 3, 2) + 1).numpy().astype(np.float64))   # 1-based
save("vq_zq", zst); save("vq_dz", z.grad); save("vq_dE", E.grad)

# --- FSQ (Mentzer et al. 2023, reference algorithm), levels [8, 5, 5, 5] ---
levels = torch.tensor([8, 5, 5, 5], dtype=dt)
def bound(z, eps=1e-3):
    half_l = (levels - 1) * (1 + eps) / 2
    offset = torch.where(levels % 2 == 0, 0.5, 0.0).to(dt)
    shift = torch.atanh(offset / half_l)
    return torch.tanh(z + shift) * half_l - offset
def round_ste(z):
    return z + (z.round() - z).detach()
z = (2 * torch.randn(3, 4, 2, 2, dtype=dt)).requires_grad_(True)
zc = z.permute(0, 2, 3, 1)                                                     # channels last
q = round_ste(bound(zc))
half_width = torch.div(levels, 2, rounding_mode="floor")
zhat = q / half_width
basis = torch.cumprod(torch.cat([torch.ones(1, dtype=dt), levels[:-1]]), 0)
index = ((zhat * half_width + half_width) * basis).sum(-1)
zhat = zhat.permute(0, 3, 1, 2)
R = torch.randn_like(zhat); (zhat * R).sum().backward()
save("fsq_z", z); save("fsq_R", R); save("fsq_zhat", zhat); save("fsq_dz", z.grad)
save("fsq_idx_NHW", index.detach() + 1)

# --- diffusion: timestep embedding (sinusoid, cos first, + MLP) ---
torch.manual_seed(21)
def sinusoid(t, dim, max_period=10000.0, scale=1000.0):
    half = dim // 2
    f = torch.exp(-np.log(max_period) * torch.arange(half, dtype=dt) / half)
    arg = scale * t[:, None] * f[None]
    return torch.cat([torch.cos(arg), torch.sin(arg)], dim=1)          # (N, dim)
t = torch.tensor([0.0, 0.013, 0.5, 0.97], dtype=dt)
fc1 = nn.Linear(8, 16).double(); fc2 = nn.Linear(16, 16).double()
e = fc2(F.silu(fc1(sinusoid(t, 8))))
save("te_t", t); save("te_phi", sinusoid(t, 8)); save("te_e", e)
save("te_fc1_weight", fc1.weight); save("te_fc1_bias", fc1.bias); save("te_fc2_weight", fc2.weight); save("te_fc2_bias", fc2.bias)

# --- diffusion: ADM residual block with scale-shift (FiLM) time conditioning ---
class FiLMRes(nn.Module):
    def __init__(s, cin, cout, edim, groups):
        super().__init__()
        s.norm1 = nn.GroupNorm(groups, cin); s.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        s.emb = nn.Linear(edim, 2 * cout)
        s.norm2 = nn.GroupNorm(groups, cout); s.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        s.skip = nn.Conv2d(cin, cout, 1)
    def forward(s, x, emb):
        h = s.conv1(F.silu(s.norm1(x)))
        scale, shift = s.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = s.norm2(h) * (1 + scale) + shift
        return s.skip(x) + s.conv2(F.silu(h))
blk = FiLMRes(4, 6, 5, 2).double()
x = torch.randn(3, 4, 5, 5, dtype=dt, requires_grad=True)
emb = torch.randn(3, 5, dtype=dt, requires_grad=True)
y = blk(x, emb); R = torch.randn_like(y); (y * R).sum().backward()
save("film_x", x); save("film_emb", emb); save("film_R", R); save("film_y", y)
save("film_dx", x.grad); save("film_demb", emb.grad)
for name, prm in blk.named_parameters():
    save("film_" + name.replace(".", "_"), prm)

# --- diffusion transformers: DiT block (adaLN-Zero, LayerNorm, GELU) and an
#     MMDiT block with RMSNorm, SwiGLU, per-head RMS QK-norm, 2-D axial RoPE
#     on the image stream and joint attention. Token layout (N, T, D). ---
torch.manual_seed(31)
D, H = 8, 2; dk = D // H
def rmsn(x, w=None, eps=1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w
def lnorm(x): return F.layer_norm(x, (x.shape[-1],), eps=1e-6)
def modulate(x, sh, sc): return x * (1 + sc[:, None]) + sh[:, None]
def split(x): N, T, _ = x.shape; return x.view(N, T, H, dk).transpose(1, 2)      # N, H, T, dk
def merge(o): N, h, T, d = o.shape; return o.transpose(1, 2).reshape(N, T, h * d)
def attend(q, k, v): return torch.softmax(q @ k.transpose(-1, -2) / dk ** 0.5, -1) @ v
def rope2d(Hp, Wp, d, base=10000.0):
    f = base ** (-torch.arange(d // 4, dtype=dt) / (d // 4))
    r = torch.arange(Hp, dtype=dt).repeat(Wp)                  # column-major: row index fastest
    c = torch.arange(Wp, dtype=dt).repeat_interleave(Hp)
    phi = torch.cat([r[:, None] * f, c[:, None] * f], dim=1)    # T x d/2
    th = torch.cat([phi, phi], dim=1)
    return torch.cos(th), torch.sin(th)
def rot(x, cs):
    cos, sin = cs; h = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin
def lin(i, o): return nn.Linear(i, o).double()

class Attn(nn.Module):
    def __init__(s, qknorm):
        super().__init__(); s.q, s.k, s.v, s.o = lin(D, D), lin(D, D), lin(D, D), lin(D, D)
        s.qn = nn.Parameter(1 + 0.2 * torch.randn(dk, dtype=dt)) if qknorm else None
        s.kn = nn.Parameter(1 + 0.2 * torch.randn(dk, dtype=dt)) if qknorm else None
    def qkv(s, x, cs=None):
        q, k, v = split(s.q(x)), split(s.k(x)), split(s.v(x))
        if s.qn is not None: q, k = rmsn(q, s.qn), rmsn(k, s.kn)
        if cs is not None: q, k = rot(q, cs), rot(k, cs)
        return q, k, v

# DiT block (official structure), adaLN weights random so the test exercises modulation
class DiTBlock(nn.Module):
    def __init__(s):
        super().__init__(); s.attn = Attn(False); s.fc1, s.fc2 = lin(D, 4 * D), lin(4 * D, D); s.ada = lin(D, 6 * D)
    def forward(s, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = s.ada(F.silu(c)).chunk(6, 1)
        q, k, v = s.attn.qkv(modulate(lnorm(x), sh1, sc1))
        x = x + g1[:, None] * s.attn.o(merge(attend(q, k, v)))
        return x + g2[:, None] * s.fc2(F.gelu(s.fc1(modulate(lnorm(x), sh2, sc2))))
blk = DiTBlock()
x = torch.randn(2, 6, D, dtype=dt, requires_grad=True); c = torch.randn(2, D, dtype=dt, requires_grad=True)
y = blk(x, c); R = torch.randn_like(y); (y * R).sum().backward()
save("dit_x", x); save("dit_c", c); save("dit_R", R); save("dit_y", y); save("dit_dx", x.grad); save("dit_dc", c.grad)
save("dit_dada_weight", blk.ada.weight.grad)
for name, prm in blk.named_parameters(): save("dit_" + name.replace(".", "_"), prm)

# MMDiT block: image stream x (grid 2 x 3 -> 6 tokens, RoPE) + context stream z (3 tokens)
class MMBlock(nn.Module):
    def __init__(s):
        super().__init__()
        s.ax, s.az = Attn(True), Attn(True)
        s.x1, s.x2, s.z1, s.z2 = lin(D, 2 * 12), lin(12, D), lin(D, 2 * 12), lin(12, D)   # SwiGLU, hidden 12
        s.adax, s.adaz = lin(D, 6 * D), lin(D, 6 * D)
    def mlp(s, f1, f2, h):
        g, v = f1(h).chunk(2, -1); return f2(F.silu(g) * v)
    def forward(s, x, z, c, cs):
        mx = s.adax(F.silu(c)).chunk(6, 1); mz = s.adaz(F.silu(c)).chunk(6, 1)
        qx, kx, vx = s.ax.qkv(modulate(rmsn(x), mx[0], mx[1]), cs)
        qz, kz, vz = s.az.qkv(modulate(rmsn(z), mz[0], mz[1]))
        o = attend(torch.cat([qx, qz], 2), torch.cat([kx, kz], 2), torch.cat([vx, vz], 2))
        Tx = x.shape[1]
        x = x + mx[2][:, None] * s.ax.o(merge(o[:, :, :Tx])); z = z + mz[2][:, None] * s.az.o(merge(o[:, :, Tx:]))
        x = x + mx[5][:, None] * s.mlp(s.x1, s.x2, modulate(rmsn(x), mx[3], mx[4]))
        z = z + mz[5][:, None] * s.mlp(s.z1, s.z2, modulate(rmsn(z), mz[3], mz[4]))
        return x, z
mm = MMBlock()
x = torch.randn(2, 6, D, dtype=dt, requires_grad=True); z = torch.randn(2, 3, D, dtype=dt, requires_grad=True)
c = torch.randn(2, D, dtype=dt, requires_grad=True)
yx, yz = mm(x, z, c, rope2d(2, 3, dk))
Rx, Rz = torch.randn_like(yx), torch.randn_like(yz); ((yx * Rx).sum() + (yz * Rz).sum()).backward()
for n_, a_ in dict(x=x, z=z, c=c, Rx=Rx, Rz=Rz, yx=yx, yz=yz).items(): save("mm_" + n_, a_)
save("mm_dx", x.grad); save("mm_dz", z.grad); save("mm_dc", c.grad); save("mm_dqn", mm.ax.qn.grad)
for name, prm in mm.named_parameters(): save("mm_" + name.replace(".", "_"), prm)

# --- hierarchical VAE: VDVAE bottleneck block and one top-down latent layer ---
torch.manual_seed(41)
class Bottleneck(nn.Module):
    def __init__(s, cin, cout, mid):
        super().__init__()
        s.c1 = nn.Conv2d(cin, mid, 1); s.c2 = nn.Conv2d(mid, mid, 3, padding=1)
        s.c3 = nn.Conv2d(mid, mid, 3, padding=1); s.c4 = nn.Conv2d(mid, cout, 1); s.skip = nn.Conv2d(cin, cout, 1)
    def forward(s, x):
        h = s.c4(F.gelu(s.c3(F.gelu(s.c2(F.gelu(s.c1(F.gelu(x))))))))
        return s.skip(x) + h
bn = Bottleneck(6, 8, 2).double()
x = torch.randn(2, 6, 5, 4, dtype=dt, requires_grad=True)
y = bn(x); R = torch.randn_like(y); (y * R).sum().backward()
save("bn_x", x); save("bn_R", R); save("bn_y", y); save("bn_dx", x.grad)
for name, prm in bn.named_parameters(): save("bn_" + name.replace(".", "_"), prm)

class Res(nn.Module):          # ConvResBlock without conditioning; identity skip when widths match
    def __init__(s, cin, cout, groups):
        super().__init__()
        s.norm1 = nn.GroupNorm(groups, cin); s.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        s.norm2 = nn.GroupNorm(groups, cout); s.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        s.skip = nn.Conv2d(cin, cout, 1) if cin != cout else None
    def forward(s, x):
        h = s.conv2(F.silu(s.norm2(s.conv1(F.silu(s.norm1(x))))))
        return (x if s.skip is None else s.skip(x)) + h
class TopDown(nn.Module):
    def __init__(s, w, zc, g):
        super().__init__()
        s.prior_block = Res(w, w, g); s.prior_head = nn.Conv2d(w, 2 * zc, 3, padding=1)
        s.post_block = Res(2 * w, w, g); s.post_head = nn.Conv2d(w, 2 * zc, 3, padding=1)
        s.zproj = nn.Conv2d(zc, w, 1); s.block = Res(w, w, g)
    def forward(s, h, e):
        mp, lp = s.prior_head(s.prior_block(h)).chunk(2, 1)
        dm, dl = s.post_head(s.post_block(torch.cat([h, e], 1))).chunk(2, 1)
        mq, lq = mp + dm, lp + dl                                   # residual posterior (NVAE)
        kl = 0.5 * (lp - lq + (torch.exp(lq) + (mq - mp) ** 2) / torch.exp(lp) - 1)
        return s.block(h + s.zproj(mq)), kl                        # z = posterior mean ("mode")
td = TopDown(4, 2, 2).double()
h = torch.randn(2, 4, 3, 3, dtype=dt, requires_grad=True); e = torch.randn(2, 4, 3, 3, dtype=dt, requires_grad=True)
ho, kl = td(h, e); Rh = torch.randn_like(ho); Rk = torch.randn_like(kl)
((ho * Rh).sum() + (kl * Rk).sum()).backward()
for n_, a_ in dict(h=h, e=e, ho=ho, kl=kl, Rh=Rh, Rk=Rk).items(): save("td_" + n_, a_)
save("td_dh", h.grad); save("td_de", e.grad); save("td_dprior_head_weight", td.prior_head.weight.grad)
for name, prm in td.named_parameters(): save("td_" + name.replace(".", "_"), prm)
print("generative fixtures written, torch", torch.__version__)
