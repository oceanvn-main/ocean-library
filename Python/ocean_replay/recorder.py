"""Record a domain's run following a Spec -- mirror of OceanReplay.Record.

A domain implements IRecordable:
    axes_info()      -> list[Axis]         (empty values = open axis)
    catalog()        -> (list[ChannelInfo], list[ViewInfo])
    run(recorder)    -> output; calls recorder.point(coords, state) at every point,
                        state = {name: zero-argument callable}  (lazy: evaluated only if wanted)
    describe()       -> dict(model, references, notation[, layout])   (optional)

At each point the Recorder evaluates ONLY the wanted channels, computes wanted views
inside rng_guard() with their own numpy Generator (seed view_seed), appends, and
returns. A channel with no axes is stored the first time the state provides it.
verify_invariance() checks that recording does not change the run's output.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import platform
import subprocess
import tempfile
import warnings
from contextlib import contextmanager

import numpy as np

from .spec import Spec
from .writer import Hdf5Writer


class IRecordable:
    def axes_info(self):
        raise NotImplementedError

    def catalog(self):
        raise NotImplementedError

    def run(self, recorder):
        raise NotImplementedError

    def describe(self) -> dict:
        return {"model": {"name": type(self).__name__}, "references": [], "notation": {}}


class NullRecorder:
    """Records nothing: run(NullRecorder()) is the computation exactly as without recording."""

    def point(self, coords, state):
        pass


@contextmanager
def rng_guard():
    """Save and restore Python, NumPy and (if present) torch CPU/CUDA random states."""
    import random
    py, npst = random.getstate(), np.random.get_state()
    torch_states = None
    try:
        import torch
        torch_states = (torch.random.get_rng_state(),
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    except ImportError:
        torch = None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(npst)
        if torch_states is not None:
            torch.random.set_rng_state(torch_states[0])
            if torch_states[1] is not None:
                torch.cuda.set_rng_state_all(torch_states[1])


def _commit() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "uncommitted"
    except Exception:
        return "uncommitted"


class Recorder:
    def __init__(self, recordable: IRecordable, spec: Spec, path, overwrite=False, view_seed=1007,
                 title="", run_id="", seeds=None):
        self.recordable, self.spec, self.path = recordable, spec, str(path)
        self.overwrite, self.view_seed = overwrite, view_seed
        self.title, self.run_id, self.seeds = title, run_id, dict(seeds or {})
        self.axes = recordable.axes_info()
        self.channels, self.view_catalog = recordable.catalog()
        spec.validate(self.channels, self.view_catalog, self.axes)
        self.estimated_bytes, per = spec.estimate_bytes(self.channels, self.view_catalog, self.axes)
        self.report = {"estimate": per}
        if math.isfinite(spec.size_cap_bytes):
            if math.isnan(self.estimated_bytes):
                warnings.warn("Size cannot be estimated in advance (open axis or run-time shape); the cap is not enforced.")
            elif self.estimated_bytes > spec.size_cap_bytes:
                raise ValueError(f"Estimated size {self.estimated_bytes:.3g} bytes exceeds the spec's cap of "
                                 f"{spec.size_cap_bytes:.3g} bytes.")

    def meta(self) -> dict:
        d = self.recordable.describe()
        now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
        versions = {"python": platform.python_version(), "numpy": np.__version__}
        try:
            import h5py
            versions["h5py"] = h5py.__version__
        except ImportError:
            pass
        try:
            import torch
            versions["torch"] = torch.__version__
        except ImportError:
            pass
        seeds = dict(self.seeds); seeds["views"] = self.view_seed
        m = {"title": self.title or type(self.recordable).__name__,
             "created": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "runId": self.run_id or "run-" + now.strftime("%Y%m%d-%H%M%S"),
             "source": {"language": "python", "library": "ocean_replay (Python)", "versions": versions,
                        "oceanCommit": _commit()},
             "model": d.get("model", {}), "seeds": seeds, "references": list(d.get("references", [])),
             "notation": d.get("notation", {}), "spec": self.spec.to_dict()}
        if "layout" in d:
            m["layout"] = d["layout"]
        return m

    def record(self):
        meta = self.meta()
        self.writer = Hdf5Writer(self.path, overwrite=self.overwrite, precision=self.spec.precision)
        try:
            self.writer.begin(meta, self.axes)
            cat = {c.name: c for c in self.channels}
            self.channel_names = self.spec.channel_names()
            self.done = {}
            for n in self.channel_names:
                self.writer.define(cat[n].copy(level=self.spec.level(n)), "channels")
                self.done[n] = set()
            vcat = {v.name: v for v in self.view_catalog}
            self.view_infos = []
            for sv in self.spec.views:
                v = vcat[sv.method].copy(name=sv.name, method_id=sv.method, params=dict(sv.params))
                self.view_infos.append(v)
                self.writer.define(v, "views")
                self.done[sv.name] = set()
            self.view_rng = np.random.default_rng(self.view_seed)
            out = self.recordable.run(self)
            self._finish(meta)
        except BaseException:
            self.writer.f.close()                  # leaves a readable "partial" file
            raise
        return out

    def point(self, coords: dict, state: dict):
        idx = {a: self.writer.axis_index(a, v) for a, v in coords.items()}
        for n in self.channel_names:
            info = self.writer.info_of(n)
            ok, key, ix = self._point_of(info, coords, idx, n)
            if not ok or not self.spec.wants(n, coords, idx):
                continue
            if n not in state:
                if not info.axes:
                    continue
                raise KeyError(f'The spec wants "{n}" at {coords}, but the domain\'s state has no provider for it.')
            p = state[n]
            self.writer.append(n, p() if callable(p) else p, ix)
            self.done[n].add(key)
        for j, v in enumerate(self.view_infos):
            ok, key, ix = self._point_of(v, coords, idx, v.name)
            if not ok or not self.spec.view_wants(j, coords, idx):
                continue
            with rng_guard():
                value = v.compute(state, v.params, self.view_rng)
            self.writer.append(v.name, value, ix)
            self.done[v.name].add(key)

    def _point_of(self, info, coords, idx, name):
        if any(a not in coords for a in info.axes):
            return False, None, None
        ix = [idx[a] for a in info.axes]
        key = tuple(ix)
        return key not in self.done[name], key, ix

    def _finish(self, meta):
        not_visited = []
        for n in self.channel_names:
            info = self.writer.info_of(n)
            for g in self.spec.groups:
                if n not in g.channels:
                    continue
                for a, rule in g.at.items():
                    if a in info.axes:
                        not_visited += [f"{n}@{a}={v:g}" for v in rule.missing(self.writer.axis_values[a])]
        self.report["not_visited"] = sorted(set(not_visited))
        for n in self.channel_names:
            if not self.writer.info_of(n).axes and self.writer.count(n) == 0:
                warnings.warn(f'The spec wants "{n}" (no axes), but the run never provided it.')
        if not_visited:
            meta["notVisited"] = self.report["not_visited"]
            self.writer.set_meta(meta)
            warnings.warn("Requested points never visited by the run: " + ", ".join(self.report["not_visited"]))
        self.writer.close()

    @staticmethod
    def run_to(recordable, spec, path, **kw):
        return Recorder(recordable, spec, path, **kw).record()


def verify_invariance(factory, spec, path=None, seed=0, equal=None, **kw):
    """Run a fresh domain without and with recording (same global seeds); compare outputs.

    equal: comparison function (default: exact, element-wise for numpy/torch outputs)."""
    import random
    path = path or os.path.join(tempfile.mkdtemp(), "invariance.h5")

    def seed_all():
        random.seed(seed); np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass

    with rng_guard():
        seed_all()
        without = factory().run(NullRecorder())
        seed_all()
        with_ = Recorder(factory(), spec, path, overwrite=True, **kw).record()
    eq = equal or _exact_equal
    return eq(without, with_), {"without": without, "with": with_, "path": path}


def _exact_equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_exact_equal(a[k], b[k]) for k in a)
    if hasattr(a, "detach"):
        a, b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    return a == b
