"""Deterministic test domain -- the Python twin of OceanTest.OceanReplay.ToyRecordable.

Outer axis t = 10, 8, ..., 0 (descending, like a diffusion sampler), inner k = 1, 2, 3;
one channel of every kind, a channel it never provides ("ghost"), and a view that
draws from its own generator AND from numpy's global generator (the run also uses
the global generator, so an unguarded view would change the output).
"""
import numpy as np

from ocean_replay import Axis, ChannelInfo, IRecordable, Sampling, Spec, ViewInfo

T = [10.0, 8.0, 6.0, 4.0, 2.0, 0.0]
K = [1.0, 2.0, 3.0]


class Toy(IRecordable):
    def __init__(self, open_axis=False, seed=5):
        self.open_axis, self.seed = open_axis, seed
        self.calls = {}
        self.history = {}

    def axes_info(self):
        return [Axis("t", [] if self.open_axis else T, "t", "", "outer step, descending"),
                Axis("k", K, "k", "", "inner index")]

    def catalog(self):
        C = ChannelInfo
        c = [C("energy", "scalar", ["t"], symbol="E", description="sum of squares of x"),
             C("vec", "vector", ["t"], [4]),
             C("mat", "matrix", ["t", "k"], [2, 3], row_labels=["r1", "r2"]),
             C("img", "image", ["t"], [2, 3, 2], display_range=[-5, 5]),
             C("seq", "sequence", ["t"], [5, 2]),
             C("pts", "points", [], [4, 2]),
             C("tok", "tokens", ["k"], [3]),
             C("label", "vector", ["t"], [4], dtype="int32", categories=["a", "b", "c"]),
             C("ghost", "scalar", ["t"])]
        v = [ViewInfo("noisy", "image", ["t"], [2, 3, 2], display_range=[-5, 5],
                      compute=lambda s, p, r: s["img"]() + p["scale"] * r.standard_normal((2, 3, 2)) + 0 * np.random.rand(),
                      how_made="img plus Gaussian noise from the view stream (test only)",
                      reference="test fixture", in_texts=False)]
        return c, v

    def tick(self, name, coords, v):
        self.calls[name] = self.calls.get(name, 0) + 1
        self.history.setdefault(name, []).append((coords, v))
        return v

    def run(self, rec):
        rs = np.random.default_rng(self.seed)
        x = np.zeros((2, 3, 2))
        total = 0.0
        for t in T:
            x = 0.5 * x + rs.standard_normal((2, 3, 2)) + 0.01 * np.random.randn()   # global draw on purpose
            for k in K:
                y = x[:, :, 0] * k
                total += y.sum()
                xs, tt, kk = x.copy(), t, k
                st = {
                    "energy": lambda: self.tick("energy", (tt, kk), float((xs ** 2).sum())),
                    "vec": lambda: self.tick("vec", (tt, kk), np.array([tt, xs.flat[0], xs.flat[-1], xs.sum()])),
                    "mat": lambda: self.tick("mat", (tt, kk), xs[:, :, 0] * kk),
                    "img": lambda: self.tick("img", (tt, kk), xs),
                    "seq": lambda: self.tick("seq", (tt, kk), np.column_stack([np.arange(1, 6) * tt, np.arange(1, 6) * xs.flat[0]])),
                    "pts": lambda: self.tick("pts", (tt, kk), np.arange(1, 9, dtype=float).reshape(4, 2, order="F")),
                    "tok": lambda: self.tick("tok", (tt, kk), [f"w{int(kk)}", "ñ", "中"]),
                    "label": lambda: self.tick("label", (tt, kk), np.array([1, 2, 3, int(tt) % 3 + 1], np.int32)),
                }
                rec.point({"t": t, "k": k}, st)
        return {"x": x, "total": total, "last_global": np.random.rand()}


def full_spec(precision="float32"):
    S = Sampling
    return (Spec(precision=precision)
            .add_group("all", 0, ["energy", "vec", "mat", "img", "seq", "pts", "tok", "label"])
            .add_view("noisy", "noisy", {"scale": 0.1}, {"t": S.list_([8, 2])}))
