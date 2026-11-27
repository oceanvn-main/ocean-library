"""Sampling rules on ONE axis (Format spec Sec. 5.1) -- mirror of OceanReplay.Core.Sampling.

Rules act on axis VALUES (the real step / epoch / lambda), except every with
kIndex, which counts the 0-based run-order index:
  all | every(k, from=0, to) | every(kIndex) | list(values) | window(center, half) | union(rules)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

REL_TOL = 1e-9


def _tol(v: float) -> float:
    return REL_TOL * max(1.0, abs(v))


def _as_list(x):
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]


@dataclass
class Sampling:
    rule: str = "all"
    k: float | None = None
    k_index: int | None = None
    from_: float = 0.0
    to: float | None = None
    values: list = field(default_factory=list)
    center: float | None = None
    half: float | None = None
    rules: list = field(default_factory=list)

    # ---- constructors (same names as MATLAB)
    @staticmethod
    def all() -> "Sampling":
        return Sampling()

    @staticmethod
    def every(k: float, from_: float = 0.0, to: float | None = None) -> "Sampling":
        if k <= 0:
            raise ValueError("every(k): k must be positive")
        return Sampling("every", k=float(k), from_=float(from_), to=None if to is None else float(to))

    @staticmethod
    def every_index(k: int) -> "Sampling":
        return Sampling("every", k_index=int(k))

    @staticmethod
    def list_(values) -> "Sampling":
        return Sampling("list", values=_as_list(values))

    @staticmethod
    def window(center: float, half: float) -> "Sampling":
        return Sampling("window", center=float(center), half=float(half))

    @staticmethod
    def union(*rules: "Sampling") -> "Sampling":
        return Sampling("union", rules=list(rules))

    @staticmethod
    def from_dict(t: dict) -> "Sampling":
        if "rule" not in t:
            raise ValueError('A sampling rule needs a "rule" field.')
        r = t["rule"]
        if r == "all":
            return Sampling.all()
        if r == "every":
            if "kIndex" in t:
                return Sampling.every_index(t["kIndex"])
            return Sampling.every(t["k"], t.get("from", 0.0), t.get("to"))
        if r == "list":
            return Sampling.list_(t["values"])
        if r == "window":
            return Sampling.window(t["center"], t["half"])
        if r == "union":
            return Sampling.union(*[Sampling.from_dict(x) for x in t["rules"]])
        raise ValueError(f'Unknown sampling rule "{r}" (all | every | list | window | union).')

    def to_dict(self) -> dict:
        t = {"rule": self.rule}
        if self.rule == "every":
            if self.k_index is not None:
                t["kIndex"] = self.k_index
            else:
                t["k"] = self.k
                t["from"] = self.from_
                if self.to is not None:
                    t["to"] = self.to
        elif self.rule == "list":
            t["values"] = list(self.values)
        elif self.rule == "window":
            t["center"], t["half"] = self.center, self.half
        elif self.rule == "union":
            t["rules"] = [r.to_dict() for r in self.rules]
        return t

    # ---- semantics
    def contains(self, value: float, index: float = math.nan) -> bool:
        tol = _tol(value)
        if self.rule == "all":
            return True
        if self.rule == "every":
            if self.k_index is not None:
                return not math.isnan(index) and int(index) % self.k_index == 0
            r = math.fmod(value - self.from_, self.k)
            if r < 0:
                r += self.k
            ok = min(r, self.k - r) <= tol
            if self.to is not None:
                lo, hi = min(self.from_, self.to), max(self.from_, self.to)
                ok = ok and lo - tol <= value <= hi + tol
            return ok
        if self.rule == "list":
            return any(abs(v - value) <= tol for v in self.values)
        if self.rule == "window":
            return abs(value - self.center) <= self.half + tol
        if self.rule == "union":
            return any(r.contains(value, index) for r in self.rules)
        raise ValueError(self.rule)

    def select(self, values) -> list:
        """Boolean mask over a known axis (values in run order)."""
        return [self.contains(float(v), i) for i, v in enumerate(values)]

    def missing(self, values) -> list:
        """list values (also inside unions) that are not on an axis with these values."""
        values = [float(v) for v in values]
        if self.rule == "list":
            return [v for v in self.values if not any(abs(a - v) <= _tol(v) for a in values)]
        if self.rule == "union":
            out = []
            for r in self.rules:
                out += r.missing(values)
            return out
        return []
