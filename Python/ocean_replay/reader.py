"""Read-only view of a Replay file -- mirror of OceanReplay.Core.Recording.

get(name, coords) returns the NEAREST STORED point and its real axis values; it never
interpolates or blends (Format spec Sec. 6). Attributes are normalised across writers
(MATLAB writes scalar strings as shape-(1,) arrays; empty lists may read back as "").
"""
from __future__ import annotations

import json
import math

import h5py
import numpy as np

from .info import Axis, ChannelInfo, ViewInfo


def _one(a):
    if isinstance(a, np.ndarray) and a.size == 1:
        a = a.reshape(-1)[0]
    if isinstance(a, bytes):
        a = a.decode("utf-8")
    if isinstance(a, np.generic):
        a = a.item()
    return a


def _text(attrs, key, default=""):
    if key not in attrs:
        return default
    v = attrs[key]
    if isinstance(v, h5py.Empty):
        return default
    v = _one(v)
    return "" if v is None else str(v)


def _list(attrs, key):
    if key not in attrs:
        return []
    v = attrs[key]
    if isinstance(v, h5py.Empty):
        return []
    arr = np.atleast_1d(v).reshape(-1)
    out = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]
    return [x for x in out if x]


def _numbers(attrs, key, default=None):
    if key not in attrs:
        return default
    v = attrs[key]
    if isinstance(v, h5py.Empty):
        return default
    return [float(x) for x in np.atleast_1d(v).reshape(-1)]


class Recording:
    def __init__(self, path):
        self.path = str(path)
        self.f = h5py.File(self.path, "r")
        a = self.f.attrs
        if _text(a, "format") != "ocean-replay":
            raise ValueError(f'"{path}" is not an ocean-replay file.')
        self.schema_version = _text(a, "schemaVersion")
        if not self.schema_version.startswith("1."):
            raise ValueError(f'Unsupported schema version "{self.schema_version}" (this reader: 1.x).')
        self.status = _text(a, "status")
        m = _text(a, "meta")
        self.meta = json.loads(m) if m else {}
        self.axes = {}
        for n, ds in self.f.get("axes", {}).items():
            self.axes[n] = Axis(n, list(np.asarray(ds[()], np.float64).reshape(-1)), _text(ds.attrs, "symbol"),
                                _text(ds.attrs, "units"), _text(ds.attrs, "description"))
        self.channels = {n: self._info(g, False, n) for n, g in self.f.get("channels", {}).items()}
        self.views = {n: self._info(g, True, n) for n, g in self.f.get("views", {}).items()}
        self._points = {}
        self.read_count = 0

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _info(self, g, is_view, name):
        a = g.attrs
        lvl = _numbers(a, "level", [-1])[0]
        kw = dict(axes=_list(a, "axes"), value_shape=[int(x) for x in _numbers(a, "valueShape", [])],
                  dtype=_text(a, "dtype"), symbol=_text(a, "symbol"), units=_text(a, "units"),
                  description=_text(a, "description"), level=math.nan if lvl < 0 else lvl,
                  display_range=_numbers(a, "displayRange", []) or [], dim_format=_text(a, "dimFormat"),
                  categories=_list(a, "categories"), row_labels=_list(a, "rowLabels"),
                  col_labels=_list(a, "colLabels"), labels_channel=_text(a, "labelsChannel"),
                  symbol_latex=_text(a, "symbolLatex"))
        if not is_view:
            return ChannelInfo(name, _text(a, "kind"), **kw)
        p = _text(a, "params")
        params = json.loads(p) if p else {}
        return ViewInfo(name, _text(a, "kind"), **kw, how_made=_text(a, "method"), reference=_text(a, "reference"),
                        in_texts=bool(_numbers(a, "inTexts", [0])[0]), params=params if isinstance(params, dict) else {},
                        method_id=_text(a, "methodId"))

    # ---- queries
    def names(self):
        return list(self.channels) + list(self.views)

    def info(self, name):
        if name in self.channels:
            return self.channels[name]
        if name in self.views:
            return self.views[name]
        raise KeyError(f'No channel or view "{name}" (available: {", ".join(self.names())}).')

    def _group(self, name):
        return self.f[("/views/" if name in self.views else "/channels/") + name]

    @property
    def is_partial(self):
        return self.status != "complete"

    def stored(self, name):
        """(index P x A, 0-based; values P x A axis values), recording order."""
        if name in self._points:
            return self._points[name]
        info, g = self.info(name), self._group(name)
        P = g["data"].shape[0]
        if not info.axes:
            out = (np.zeros((P, 0), int), np.zeros((P, 0)))
        else:
            I = np.asarray(g["points"][()], np.int64).reshape(P, len(info.axes))
            V = np.column_stack([np.asarray(self.axes[a].values)[I[:, k]] for k, a in enumerate(info.axes)])
            out = (I, V)
        self._points[name] = out
        return out

    def get(self, name, coords=None, mode="nearest"):
        """(value, actual coords) at the nearest stored point ("nearest") or exactly ("exact")."""
        info = self.info(name)
        I, V = self.stored(name)
        if not info.axes:
            p, actual = 0, {}
        else:
            if isinstance(coords, dict):
                missing = [a for a in info.axes if a not in coords]
                if missing:
                    raise KeyError(f'"{name}" needs a coordinate on axis "{missing[0]}".')
                c = [float(coords[a]) for a in info.axes]
            else:
                c = [float(x) for x in np.atleast_1d(coords)]
            cand = np.arange(V.shape[0])
            for k in range(len(info.axes)):
                d = np.abs(V[cand, k] - c[k])
                if mode == "exact":
                    cand = cand[d <= 1e-9 * max(1.0, abs(c[k]))]
                else:
                    cand = cand[d == d.min()]
                if cand.size == 0:
                    break
            if cand.size == 0:
                raise KeyError(f'"{name}" is not stored at {c}.')
            p = int(cand[0])                                  # ties -> recorded first
            actual = {a: float(V[p, k]) for k, a in enumerate(info.axes)}
        return self._value(name, info, p), actual

    def _value(self, name, info, p):
        self.read_count += 1
        raw = self._group(name)["data"][p]
        if info.kind == "tokens":
            return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in np.atleast_1d(raw)]
        a = np.asarray(raw)
        if info.kind == "scalar":
            return float(a.reshape(-1)[0]) if a.dtype.kind == "f" else a.reshape(-1)[0].item()
        if info.kind == "vector":
            return a.reshape(-1)
        return np.transpose(a)                                # reversed on disk -> logical shape

    def series(self, name):
        info = self.info(name)
        if info.kind != "scalar" or len(info.axes) != 1:
            raise ValueError(f'series() needs a scalar channel over one axis; "{name}" is {info.kind} over {len(info.axes)} axes.')
        _, V = self.stored(name)
        return V[:, 0], np.asarray(self._group(name)["data"][()], np.float64).reshape(-1)
