"""Streams a recording to HDF5 (Format spec Sec. 3) -- mirror of OceanReplay.IO.Hdf5Writer.

On disk a value of logical shape S at point p is data[p] with shape reversed(S):
data has shape (P, *reversed(S)); MATLAB then reads [S P] with no permute (G0 rule).
Chunks of about 8 KiB (at most 64 points) along P, like the MATLAB writer.
"""
from __future__ import annotations

import json
import math
import os

import h5py
import numpy as np

from .info import BYTES, NUMPY, ChannelInfo, normalise, shape_of

STR = h5py.string_dtype("utf-8")


def _write_attr(obj, key, value):
    if isinstance(value, str):
        obj.attrs[key] = value
    elif isinstance(value, (list, tuple)) and len(value) == 0:
        obj.attrs[key] = h5py.Empty(STR)            # null dataspace, as MATLAB writes an empty list
                                                    # (MATLAB cannot read a zero-length string array)
    elif isinstance(value, (list, tuple)) and isinstance(value[0], str):
        obj.attrs.create(key, np.array(list(value), dtype=object), dtype=STR)
    else:
        obj.attrs[key] = value


class Hdf5Writer:
    def __init__(self, path, overwrite: bool = False, precision: str = "float32"):
        if os.path.exists(path):
            if not overwrite:
                raise FileExistsError(f'"{path}" exists (use overwrite=True).')
            os.remove(path)
        self.path = str(path)
        self.precision = precision
        self.f = h5py.File(self.path, "w")
        self.infos, self.paths, self.counts = {}, {}, {}
        self.axis_values, self.axis_open = {}, {}
        self.closed = False

    # ---- header
    def begin(self, meta: dict, axes):
        for a in axes:
            p = f"/axes/{a.name}"
            if a.is_open:
                ds = self.f.create_dataset(p, shape=(0,), maxshape=(None,), chunks=(64,), dtype="f8")
            else:
                ds = self.f.create_dataset(p, data=np.asarray(a.values, np.float64))
            for k, v in (("symbol", a.symbol), ("units", a.units), ("description", a.description)):
                _write_attr(ds, k, v)
            self.axis_values[a.name] = list(a.values)
            self.axis_open[a.name] = a.is_open
        _write_attr(self.f, "format", "ocean-replay")
        _write_attr(self.f, "schemaVersion", "1.0")
        _write_attr(self.f, "status", "partial")
        self.set_meta(meta)
        self.f.flush()

    def set_meta(self, meta: dict):
        _write_attr(self.f, "meta", json.dumps(meta, ensure_ascii=False, default=_json_default))

    def define(self, info: ChannelInfo, where: str):
        if where not in ("channels", "views"):
            raise ValueError(where)
        if info.name in self.infos:
            raise ValueError(f'Name "{info.name}" defined twice.')
        if not info.dtype:
            info = info.copy(dtype=self.precision)
        if info.kind == "image" and len(info.display_range) != 2:
            raise ValueError(f'Image "{info.name}" needs display_range [lo, hi].')
        for a in info.axes:
            if a not in self.axis_values:
                raise ValueError(f'"{info.name}" depends on axis "{a}", which was not declared.')
        self.infos[info.name] = info
        self.paths[info.name] = f"/{where}/{info.name}"
        self.counts[info.name] = 0

    def axis_index(self, axis: str, value: float) -> int:
        vals = self.axis_values[axis]
        for i, v in enumerate(vals):
            if v == value:
                return i
        if not self.axis_open[axis]:
            raise ValueError(f'Value {value} is not on the closed axis "{axis}".')
        vals.append(float(value))
        ds = self.f[f"/axes/{axis}"]
        ds.resize((len(vals),))
        ds[-1] = float(value)
        return len(vals) - 1

    # ---- data
    def append(self, name: str, value, idx):
        info = self.infos[name]
        if len(idx) != len(info.axes):
            raise ValueError(f'"{name}" needs {len(info.axes)} axis indices, got {len(idx)}.')
        from .info import check_value
        value = normalise(info.kind, value)
        shape = info.value_shape or shape_of(info.kind, value)
        msg = check_value(info.kind, shape, value)
        if msg:
            raise ValueError(f'"{name}": {msg}.')
        if self.counts[name] == 0:
            if not info.value_shape:
                info = info.copy(value_shape=shape)
                self.infos[name] = info
            self._create(name, info)
        P = self.counts[name]
        if not info.axes and P >= 1:
            raise ValueError(f'"{name}" has no axes and was already recorded once.')
        g = self.f[self.paths[name]]
        d = g["data"]
        d.resize(P + 1, axis=0)
        if info.kind == "tokens":
            d[P] = np.array(value, dtype=object)
        else:
            arr = np.asarray(value, dtype=NUMPY[info.dtype]).reshape(info.value_shape, order="F") \
                if info.kind not in ("scalar", "vector") else np.asarray(value, dtype=NUMPY[info.dtype]).reshape(-1)
            d[P] = np.transpose(arr) if arr.ndim > 1 else arr
        if info.axes:
            p = g["points"]
            p.resize(P + 1, axis=0)
            p[P] = np.asarray(idx, np.int32)
        self.counts[name] = P + 1

    def _create(self, name, info):
        g = self.f.require_group(self.paths[name])
        rev = tuple(reversed(info.value_shape))
        per = int(math.prod(info.value_shape)) * BYTES[info.dtype]
        chunk_p = max(1, min(64, 8192 // per))
        dt = STR if info.kind == "tokens" else NUMPY[info.dtype]
        g.create_dataset("data", shape=(0, *rev), maxshape=(None, *rev), chunks=(chunk_p, *rev), dtype=dt)
        if info.axes:
            A = len(info.axes)
            g.create_dataset("points", shape=(0, A), maxshape=(None, A), chunks=(64, A), dtype="i4")
        for k, v in info.attrs().items():
            _write_attr(g, k, v)

    def count(self, name) -> int:
        return self.counts[name]

    def info_of(self, name) -> ChannelInfo:
        return self.infos[name]

    def close(self):
        if self.closed:
            return
        _write_attr(self.f, "status", "complete")
        self.f.close()
        self.closed = True


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
