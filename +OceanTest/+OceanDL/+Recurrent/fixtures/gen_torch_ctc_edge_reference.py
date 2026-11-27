"""PyTorch nn.CTCLoss reference for CTC EDGE CASES (CTCAuditTest):
empty targets, zero input lengths, an infeasible sample (zero_infinity=True), padded steps.

Regenerate (repo root): ./.venv/Scripts/python.exe +OceanTest/+OceanDL/+Recurrent/fixtures/gen_torch_ctc_edge_reference.py
Conventions as gen_torch_ctc_reference.py: logits (T, N, C), blank = 0; MATLAB maps class c -> c+1, imports "TBC".
"""
import os, numpy as np, torch
torch.manual_seed(5)
here = os.path.dirname(os.path.abspath(__file__))
T, N, C = 6, 5, 4
logits = torch.randn(T, N, C, dtype=torch.float64, requires_grad=True)
targets = [[], [1, 2], [3], [], [2, 2]]        # empty / normal / T_n = 0 with a label (inf) / T_n = 0 empty / repeat
in_len = torch.tensor([6, 5, 0, 0, 4]); tg_len = torch.tensor([len(t) for t in targets])
flat = torch.tensor(sum(targets, []), dtype=torch.long)
save = lambda n, a: np.save(os.path.join(here, n + ".npy"), np.asarray(a))
lp = torch.log_softmax(logits, dim=2)
per = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)(lp, flat, in_len, tg_len)
for red in ["sum", "mean"]:
    lp = torch.log_softmax(logits, dim=2)
    loss = torch.nn.CTCLoss(blank=0, reduction=red, zero_infinity=True)(lp, flat, in_len, tg_len)
    g, = torch.autograd.grad(loss, logits)
    save("ctcedge_loss_" + red, np.array([loss.item()])); save("ctcedge_grad_" + red + "_TNC", g.numpy())
save("ctcedge_logits_TNC", logits.detach().numpy())
save("ctcedge_targets_padded", np.array([t + [0] * (2 - len(t)) for t in targets], dtype=np.int64))
save("ctcedge_target_lengths", tg_len.numpy().astype(np.int64)); save("ctcedge_input_lengths", in_len.numpy().astype(np.int64))
save("ctcedge_per_sample", per.detach().numpy())
print("per-sample", per.detach().numpy())
