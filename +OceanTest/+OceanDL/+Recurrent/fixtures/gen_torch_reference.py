"""Reference outputs from PyTorch for OceanDL.Recurrent cell tests.

Regenerate:  py -3.12 gen_torch_reference.py   (writes *.npy next to this file)
All arrays are saved in PyTorch's own layout (C order, time-major x: (T, N, D));
the MATLAB test must import them through NpyReader + DimFormat.toCanonical.
"""
import os, numpy as np, torch
torch.manual_seed(0)
here = os.path.dirname(os.path.abspath(__file__))
T, N, D, H = 5, 3, 4, 6
x = torch.randn(T, N, D, dtype=torch.float64)
h0 = torch.randn(1, N, H, dtype=torch.float64)
c0 = torch.randn(1, N, H, dtype=torch.float64)
def save(name, t): np.save(os.path.join(here, name + ".npy"), t.detach().numpy())
save("x_TND", x); save("h0", h0[0]); save("c0", c0[0])
with torch.no_grad():
    rnn = torch.nn.RNN(D, H, nonlinearity="tanh").double()
    y, hn = rnn(x, h0)
    save("rnn_w_ih", rnn.weight_ih_l0); save("rnn_w_hh", rnn.weight_hh_l0)
    save("rnn_b", rnn.bias_ih_l0 + rnn.bias_hh_l0); save("rnn_y", y)
    lstm = torch.nn.LSTM(D, H).double()
    y, (hn, cn) = lstm(x, (h0, c0))
    save("lstm_w_ih", lstm.weight_ih_l0); save("lstm_w_hh", lstm.weight_hh_l0)
    save("lstm_b", lstm.bias_ih_l0 + lstm.bias_hh_l0); save("lstm_y", y); save("lstm_cn", cn[0])
    gru = torch.nn.GRU(D, H).double()
    y, hn = gru(x, h0)
    save("gru_w_ih", gru.weight_ih_l0); save("gru_w_hh", gru.weight_hh_l0)
    save("gru_b_ih", gru.bias_ih_l0); save("gru_b_hh", gru.bias_hh_l0); save("gru_y", y)
# A tiny mixed fixture for NpyReader: float32 C-order 3-D and int64 Fortran-order.
np.save(os.path.join(here, "npy_c_f32.npy"), np.arange(24, dtype=np.float32).reshape(2, 3, 4))
np.save(os.path.join(here, "npy_f_i64.npy"), np.asfortranarray(np.arange(6, dtype=np.int64).reshape(2, 3)))
print("torch", torch.__version__, "-> fixtures written to", here)
