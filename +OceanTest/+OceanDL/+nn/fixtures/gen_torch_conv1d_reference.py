"""PyTorch torch.nn.functional.conv1d reference for OceanDL.Autograd.Functions.Conv1d / nn.Conv1d.

Regenerate (repo root): ./.venv/Scripts/python.exe +OceanTest/+OceanDL/+nn/fixtures/gen_torch_conv1d_reference.py
PyTorch conventions: x (N, C, T) -- imported in MATLAB with DimFormat.toCanonical(x, "BCT") -> C x N x T;
weight (Cout, Cin/groups, K) -- the SAME order as OceanDL.nn.Conv1d.Weight; bias (Cout,).
Gradients are of the scalar  L = sum(y * R)  with a random projection R (saved as conv1d_<case>_R, (N, Cout, To)).
"""
import os, numpy as np, torch
torch.manual_seed(11)
here = os.path.dirname(os.path.abspath(__file__))
#        name        N  Cin Cout K  T   stride pad dil groups bias
cases = [("dense",   2, 3,  4,   3, 11, 2,     1,  2,  1,     True),
         ("dw_mult", 3, 4,  8,   5, 9,  1,     2,  1,  4,     True),    # depthwise, channel multiplier 2
         ("dw_plain",2, 5,  5,   3, 13, 3,     0,  2,  5,     False),   # depthwise, no bias
         ("grouped", 2, 4,  6,   2, 7,  1,     1,  1,  2,     True)]    # general groups (beyond the contract)
save = lambda n, a: np.save(os.path.join(here, n + ".npy"), np.asarray(a))
for name, N, cin, cout, K, T, s, p, d, g, has_b in cases:
    x = torch.randn(N, cin, T, dtype=torch.float64, requires_grad=True)
    w = torch.randn(cout, cin // g, K, dtype=torch.float64, requires_grad=True)
    b = torch.randn(cout, dtype=torch.float64, requires_grad=True) if has_b else None
    y = torch.nn.functional.conv1d(x, w, b, stride=s, padding=p, dilation=d, groups=g)
    R = torch.randn_like(y)
    grads = torch.autograd.grad((y * R).sum(), [x, w] + ([b] if has_b else []))
    pre = "conv1d_" + name + "_"
    save(pre + "x_NCT", x.detach().numpy()); save(pre + "w", w.detach().numpy())
    save(pre + "y_NCT", y.detach().numpy()); save(pre + "R_NCT", R.numpy())
    save(pre + "gx_NCT", grads[0].numpy()); save(pre + "gw", grads[1].numpy())
    save(pre + "cfg", np.array([s, p, d, g], dtype=np.int64))
    if has_b:
        save(pre + "b", b.detach().numpy()); save(pre + "gb", grads[2].numpy())
    print(name, tuple(y.shape))
