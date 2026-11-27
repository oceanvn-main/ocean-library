"""Recording spec (Format spec Sec. 5) -- mirror of OceanReplay.Core.Spec.

The SAME JSON drives the MATLAB recorder, this recorder and both extractors.
Groups: {name, level, channels, at}; views: {name, method, params, at};
at = {axis: sampling rule}; an axis not named = "all".
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .info import BYTES
from .sampling import Sampling


def _at_contains(at: dict, values: dict, index: dict) -> bool:
    for a, rule in at.items():
        if a not in values:
            continue
        if not rule.contains(values[a], index.get(a, math.nan)):
            return False
    return True


@dataclass
class Group:
    name: str
    level: float
    channels: list
    at: dict = field(default_factory=dict)


@dataclass
class View:
    name: str
    method: str
    params: dict = field(default_factory=dict)
    at: dict = field(default_factory=dict)


@dataclass
class Spec:
    schema_version: str = "1.0"
    precision: str = "float32"
    size_cap_bytes: float = math.inf
    groups: list = field(default_factory=list)
    views: list = field(default_factory=list)

    def __post_init__(self):
        if self.precision not in ("float32", "float64"):
            raise ValueError(f'precision must be "float32" or "float64", got "{self.precision}"')

    # ---- building (chainable, like MATLAB)
    def add_group(self, name, level, channels, at=None) -> "Spec":
        channels = [channels] if isinstance(channels, str) else list(channels)
        self.groups.append(Group(name, float(level), channels, dict(at or {})))
        return self

    def add_view(self, name, method, params=None, at=None) -> "Spec":
        self.views.append(View(name, method, dict(params or {}), dict(at or {})))
        return self

    # ---- queries
    def channel_names(self) -> list:
        out = []
        for g in self.groups:
            for c in g.channels:
                if c not in out:
                    out.append(c)
        return out

    def level(self, name) -> float:
        lv = [g.level for g in self.groups if name in g.channels and not math.isnan(g.level)]
        return min(lv) if lv else math.nan

    def wants(self, name, values: dict, index: dict) -> bool:
        return any(name in g.channels and _at_contains(g.at, values, index) for g in self.groups)

    def view_wants(self, j, values: dict, index: dict) -> bool:
        return _at_contains(self.views[j].at, values, index)

    def check(self, channels, views, axes) -> list:
        """Problems of this spec against a catalog and its axes (same rules as MATLAB). [] = valid."""
        issues = []
        cn = {c.name: c for c in channels}
        vn = {v.name: v for v in views}
        an = {a.name: a for a in axes}
        for g in self.groups:
            for c in g.channels:
                if c not in cn:
                    issues.append(f'group "{g.name}": unknown channel "{c}" (catalog: {", ".join(cn)})')
                    continue
                issues += self._check_at(f'group "{g.name}", channel "{c}"', g.at, cn[c].axes, an)
        out_names = self.channel_names()
        for v in self.views:
            if v.method not in vn:
                issues.append(f'view "{v.name}": unknown method "{v.method}" (catalog: {", ".join(vn)})')
                continue
            if v.name in out_names or sum(w.name == v.name for w in self.views) > 1:
                issues.append(f'view "{v.name}": name used twice (views and channels share one namespace)')
            issues += self._check_at(f'view "{v.name}"', v.at, vn[v.method].axes, an)
        return issues

    def validate(self, channels, views, axes):
        issues = self.check(channels, views, axes)
        if issues:
            raise ValueError("Invalid recording spec:\n  - " + "\n  - ".join(issues))

    @staticmethod
    def _check_at(where, at, chan_axes, an):
        issues = []
        for a, rule in at.items():
            if a not in chan_axes:
                issues.append(f'{where}: rule on axis "{a}", which it does not depend on (axes: {", ".join(chan_axes)})')
                continue
            if a in an and not an[a].is_open:
                bad = rule.missing(an[a].values)
                if bad:
                    issues.append(f'{where}: list values {bad} are not on axis "{a}"')
        for a in chan_axes:
            if a not in an:
                issues.append(f'{where}: depends on axis "{a}", which the domain does not declare')
        return issues

    def estimate_bytes(self, channels, views, axes):
        """Upper-bound size (closed axes, known shapes); NaN when unknown."""
        an = {a.name: a for a in axes}
        cn = {c.name: c for c in channels}
        vn = {v.name: v for v in views}
        per = []

        def count(chan_axes, wants):
            if not chan_axes:
                return 1
            vals = []
            for a in chan_axes:
                if an[a].is_open:
                    return math.nan
                vals.append(an[a].values)
            n = 0
            import itertools
            for combo in itertools.product(*[list(enumerate(v)) for v in vals]):
                v = {a: val for a, (i, val) in zip(chan_axes, combo)}
                ix = {a: i for a, (i, val) in zip(chan_axes, combo)}
                n += bool(wants(v, ix))
            return n

        def bpv(info):
            if not info.value_shape:
                return math.nan
            d = info.dtype or self.precision
            return int(math.prod(info.value_shape)) * BYTES[d]

        for c in self.channel_names():
            info = cn[c]
            n = count(info.axes, lambda v, i, c=c: self.wants(c, v, i))
            per.append({"name": c, "points": n, "bytes": n * bpv(info)})
        for j, v in enumerate(self.views):
            info = vn[v.method]
            n = count(info.axes, lambda vv, i, j=j: self.view_wants(j, vv, i))
            per.append({"name": v.name, "points": n, "bytes": n * bpv(info)})
        return sum(p["bytes"] for p in per), per

    # ---- JSON
    def to_dict(self) -> dict:
        t = {"schemaVersion": self.schema_version, "precision": self.precision}
        if math.isfinite(self.size_cap_bytes):
            t["sizeCapBytes"] = self.size_cap_bytes
        t["groups"] = [{"name": g.name, "level": g.level, "channels": list(g.channels),
                        "at": {a: r.to_dict() for a, r in g.at.items()}} for g in self.groups]
        t["views"] = [{"name": v.name, "method": v.method, "params": v.params,
                       "at": {a: r.to_dict() for a, r in v.at.items()}} for v in self.views]
        return t

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_dict(t: dict) -> "Spec":
        s = Spec(schema_version=str(t.get("schemaVersion", "1.0")), precision=t.get("precision", "float32"),
                 size_cap_bytes=float(t["sizeCapBytes"]) if t.get("sizeCapBytes") is not None else math.inf)
        for g in _as_list(t.get("groups")):
            s.add_group(g["name"], g.get("level", math.nan), g["channels"],
                        {a: Sampling.from_dict(r) for a, r in (g.get("at") or {}).items()})
        for v in _as_list(t.get("views")):
            s.add_view(v["name"], v.get("method", ""), v.get("params") or {},
                       {a: Sampling.from_dict(r) for a, r in (v.get("at") or {}).items()})
        return s

    @staticmethod
    def from_json(txt: str) -> "Spec":
        return Spec.from_dict(json.loads(txt))

    @staticmethod
    def from_file(path) -> "Spec":
        with open(path, encoding="utf-8") as f:
            return Spec.from_json(f.read())


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]
