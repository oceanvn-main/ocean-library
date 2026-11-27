"""PyTorch nn.CTCLoss reference for OceanDL.Autograd.Functions.CTCLoss.

Regenerate: py -3.12 gen_torch_ctc_reference.py
PyTorch conventions: logits (T, N, C) time-major, blank = 0, targets 1..C-1 (0-based classes).
The MATLAB test maps class c (0-based) -> c+1 and blank 0 -> 1, and imports logits via "TBC".
"""
import os, numpy as np, torch
torch.manual_seed(3)
here = os.path.dirname(os.path.abspath(__file__))
T, N, C = 12, 4, 5
logits = torch.randn(T, N, C, dtype=torch.float64, requires_grad=True)
targets = [[1, 2, 2, 3], [4], [3, 1, 3], [2, 2]]           # repeats force blanks between equal labels
in_len = torch.tensor([12, 9, 12, 7]); tg_len = torch.tensor([len(t) for t in targets])
flat = torch.tensor(sum(targets, []))
out = {}
for red in ["sum", "mean"]:
    lp = torch.log_softmax(logits, dim=2)
    loss = torch.nn.CTCLoss(blank=0, reduction=red)(lp, flat, in_len, tg_len)
    g, = torch.autograd.grad(loss, logits)
    out[red] = (loss.item(), g.numpy())
save = lambda n, a: np.save(os.path.join(here, n + ".npy"), np.asarray(a))
save("ctc_logits_TNC", logits.detach().numpy())
save("ctc_targets_padded", np.array([t + [0] * (4 - len(t)) for t in targets], dtype=np.int64))
save("ctc_target_lengths", tg_len.numpy().astype(np.int64)); save("ctc_input_lengths", in_len.numpy().astype(np.int64))
save("ctc_loss_sum", np.array([out["sum"][0]])); save("ctc_grad_sum_TNC", out["sum"][1])
save("ctc_loss_mean", np.array([out["mean"][0]])); save("ctc_grad_mean_TNC", out["mean"][1])
print("CTC fixtures written:", out["sum"][0], out["mean"][0])
