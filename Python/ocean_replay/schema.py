"""Validate a Replay file (Format spec Sec. 7) -- same rules as OceanReplay.IO.Schema.

validate(path) -> list of violations, each starting with its HDF5 path; [] = valid.
Note: h5py shapes are C order, so data is (P, *reversed(valueShape)) and points (P, A).
"""
from __future__ import annotations

import json

import h5py
import numpy as np

from .info import KINDS, check_shape
from .reader import _list, _numbers, _text


def validate(path) -> list:
    issues = []
    with h5py.File(path, "r") as f:
        a = f.attrs
        if _text(a, "format") != "ocean-replay":
            return ['/: attribute format must be "ocean-replay"']
        v = _text(a, "schemaVersion")
        if not v.startswith("1."):
            issues.append(f'/: unsupported schemaVersion "{v}" (major 1 expected)')
        if _text(a, "status") not in ("partial", "complete"):
            issues.append('/: status must be "partial" or "complete"')
        try:
            meta = json.loads(_text(a, "meta"))
            for k in ("source", "seeds", "spec"):
                if k not in meta:
                    issues.append(f'/: meta has no "{k}"')
        except Exception:
            issues.append("/: meta is missing or not valid JSON")
        ax_len = {}
        for n, ds in f.get("axes", {}).items():
            if ds.dtype != np.float64:
                issues.append(f"/axes/{n}: axis must be float64")
            if sum(s > 1 for s in ds.shape) > 1:
                issues.append(f"/axes/{n}: axis must be 1-D")
            ax_len[n] = int(np.prod(ds.shape))
        for where in ("channels", "views"):
            for n, g in f.get(where, {}).items():
                issues += _check_entry(g, f"/{where}/{n}", where == "views", ax_len)
    return issues


def _check_entry(g, p, is_view, ax_len):
    issues = []
    a = g.attrs
    kind = _text(a, "kind")
    if kind not in KINDS:
        return [f'{p}: unknown kind "{kind}"']
    vs = [int(x) for x in (_numbers(a, "valueShape", []) or [])]
    msg = check_shape(kind, vs)
    if msg:
        return [f"{p}: {msg}"]
    axes = _list(a, "axes")
    for ax in axes:
        if ax not in ax_len:
            issues.append(f'{p}: axis "{ax}" does not exist under /axes')
    if "data" not in g:
        return issues + [f"{p}: no data dataset"]
    d = g["data"]
    if tuple(d.shape[1:]) != tuple(reversed(vs)):
        issues.append(f"{p}/data: size {list(d.shape)} does not match valueShape {vs} plus points")
    P = d.shape[0]
    if not axes:
        if P != 1:
            issues.append(f"{p}: a channel without axes must store exactly 1 point, has {P}")
        if "points" in g:
            issues.append(f"{p}: a channel without axes must not have points")
    elif "points" not in g:
        issues.append(f"{p}: no points dataset")
    else:
        pts = g["points"]
        if pts.shape != (P, len(axes)):
            issues.append(f"{p}/points: size {list(pts.shape)}, expected [{P} {len(axes)}] (points x axes)")
        elif all(ax in ax_len for ax in axes):
            I = np.asarray(pts[()])
            for k, ax in enumerate(axes):
                if np.any((I[:, k] < 0) | (I[:, k] >= ax_len[ax])):
                    issues.append(f'{p}/points: index outside axis "{ax}" (length {ax_len[ax]})')
    if kind == "image":
        if len(_numbers(a, "displayRange", []) or []) != 2:
            issues.append(f"{p}: an image needs displayRange [lo hi]")
        if not _text(a, "dimFormat"):
            issues.append(f"{p}: an image needs dimFormat")
    if is_view:
        if not _text(a, "method"):
            issues.append(f"{p}: a view needs a non-empty method (HOW IT WAS MADE)")
        if not _text(a, "reference"):
            issues.append(f"{p}: a view needs a non-empty reference")
        t = _numbers(a, "inTexts", None)
        if t is None or len(t) != 1 or t[0] not in (0, 1):
            issues.append(f"{p}: a view needs inTexts = 0 or 1")
    return issues
