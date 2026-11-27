"""Axis, ChannelInfo, ViewInfo and the value kinds (Format spec Sec. 1-3).

Mirrors OceanReplay.Core.{Axis, ChannelInfo, ViewInfo, Kinds}. Shapes are LOGICAL
(as MATLAB sees one value): scalar [1], vector [n], tokens [n], matrix [r c],
sequence [T D], points [N d] (d = 2, 3), image [H W C].
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

KINDS = ("scalar", "vector", "matrix", "image", "sequence", "points", "tokens")
RANK = {"scalar": 1, "vector": 1, "tokens": 1, "matrix": 2, "sequence": 2, "points": 2, "image": 3}
BYTES = {"float32": 4, "int32": 4, "float64": 8, "string": 16}
NUMPY = {"float32": np.float32, "float64": np.float64, "int32": np.int32}


def check_shape(kind: str, shape) -> str:
    """'' when shape is legal for kind, otherwise the reason (same messages as MATLAB)."""
    if kind not in KINDS:
        return f'unknown kind "{kind}"'
    shape = [int(s) for s in shape]
    r = RANK[kind]
    if len(shape) != r or any(s < 1 for s in shape):
        return f'kind "{kind}" needs {r} positive integer dimension(s), got {shape}'
    if kind == "scalar" and shape != [1]:
        return f'kind "scalar" needs valueShape [1], got {shape}'
    if kind == "points" and shape[1] not in (2, 3):
        return f'kind "points" needs N x 2 or N x 3, got {shape}'
    return ""


def normalise(kind: str, v):
    """Canonical in-memory value: numpy array of the logical shape (vectors 1-D), tokens list[str]."""
    if hasattr(v, "detach"):                       # torch tensor
        v = v.detach().cpu().numpy()
    if kind == "tokens":
        return [str(t) for t in np.asarray(v, dtype=object).reshape(-1)]
    a = np.asarray(v)
    if kind in ("scalar", "vector"):
        a = a.reshape(-1)
    return a


def shape_of(kind: str, v):
    v = normalise(kind, v)
    if kind == "scalar":
        return [1]
    if kind in ("vector", "tokens"):
        return [len(v)]
    s = list(np.asarray(v).shape) + [1] * RANK[kind]
    return s[:RANK[kind]]


def check_value(kind: str, shape, v) -> str:
    msg = check_shape(kind, shape)
    if msg:
        return msg
    v = normalise(kind, v)
    if kind == "tokens":
        return "" if len(v) == shape[0] else f'kind "tokens" needs {shape[0]} strings'
    a = np.asarray(v)
    if not (np.issubdtype(a.dtype, np.number) or a.dtype == bool) or np.iscomplexobj(a):
        return f'kind "{kind}" needs real numeric values, got {a.dtype}'
    if a.size != int(np.prod(shape)) or (RANK[kind] >= 2 and shape_of(kind, a) != list(shape)):
        return f'kind "{kind}" needs size {list(shape)}, got {list(a.shape)}'
    return ""


@dataclass
class Axis:
    """A named dimension in run order; empty values = open (filled as visited)."""
    name: str
    values: list = field(default_factory=list)
    symbol: str = ""
    units: str = ""
    description: str = ""

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError(f'Axis name "{self.name}" must be a valid identifier.')
        self.values = [float(v) for v in self.values]
        if len(set(self.values)) != len(self.values):
            raise ValueError(f'Axis "{self.name}" has repeated values; values must be unique.')

    @property
    def is_open(self) -> bool:
        return len(self.values) == 0


@dataclass
class ChannelInfo:
    """One recorded quantity: kind, axes, logical shape, textbook symbol/units/description."""
    name: str
    kind: str
    axes: list = field(default_factory=list)
    value_shape: list = field(default_factory=list)
    dtype: str = ""                       # "" = the spec's precision ("string" for tokens)
    symbol: str = ""
    units: str = ""
    description: str = ""
    symbol_latex: str = ""                # optional: the symbol as LaTeX (Replay Studio renders it)
    display_range: list = field(default_factory=list)
    dim_format: str = ""
    categories: list = field(default_factory=list)
    row_labels: list = field(default_factory=list)
    col_labels: list = field(default_factory=list)
    labels_channel: str = ""
    level: float = float("nan")

    def __post_init__(self):
        if not self.name or "/" in self.name:
            raise ValueError(f'Channel name "{self.name}" must be non-empty and contain no "/".')
        if self.kind not in KINDS:
            raise ValueError(f'Unknown kind "{self.kind}". Known kinds: {", ".join(KINDS)}.')
        self.axes = [str(a) for a in self.axes]
        self.value_shape = [int(s) for s in self.value_shape]
        if self.kind == "image" and not self.dim_format:
            self.dim_format = "HWC"
        if self.kind == "scalar" and not self.value_shape:
            self.value_shape = [1]
        if self.kind == "tokens":
            self.dtype = "string"
        if self.value_shape:
            msg = check_shape(self.kind, self.value_shape)
            if msg:
                raise ValueError(f'Channel "{self.name}": {msg}.')
        for a in self.axes:
            if not a.isidentifier():
                raise ValueError(f'Channel "{self.name}": axis name "{a}" is not a valid identifier.')

    def attrs(self) -> dict:
        """Group attributes (Format spec Sec. 3.1); optional ones only when set."""
        lvl = -1 if np.isnan(self.level) else int(self.level)
        a = {"kind": self.kind, "axes": list(self.axes), "valueShape": np.asarray(self.value_shape, np.int32),
             "dtype": self.dtype, "symbol": self.symbol, "units": self.units,
             "description": self.description, "level": np.int32(lvl)}
        if self.display_range:
            a["displayRange"] = np.asarray(self.display_range, np.float64)
        if self.dim_format:
            a["dimFormat"] = self.dim_format
        for key, val in (("categories", self.categories), ("rowLabels", self.row_labels), ("colLabels", self.col_labels)):
            if val:
                a[key] = list(val)
        if self.labels_channel:
            a["labelsChannel"] = self.labels_channel
        if self.symbol_latex:
            a["symbolLatex"] = self.symbol_latex
        return a

    def copy(self, **changes):
        return replace(self, **changes)


@dataclass
class ViewInfo(ChannelInfo):
    """A channel produced by an explanation METHOD at a point (catalog: name = method id)."""
    how_made: str = ""
    reference: str = ""
    in_texts: bool = False
    params: dict = field(default_factory=dict)
    method_id: str = ""
    compute: Optional[Callable] = None     # compute(state, params, rng) -> value; catalog only

    def __post_init__(self):
        super().__post_init__()
        if not self.method_id:
            self.method_id = self.name

    def attrs(self) -> dict:
        a = super().attrs()
        a.update({"method": self.how_made, "reference": self.reference, "inTexts": np.uint8(bool(self.in_texts)),
                  "params": json.dumps(self.params), "methodId": self.method_id})
        return a
