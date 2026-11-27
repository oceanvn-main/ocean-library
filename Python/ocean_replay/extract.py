"""Copy a subset of a recording into a smaller file of the SAME format (Format spec 5.2).

Nothing is recomputed. Mirror of OceanReplay.IO.extract; the two give identical
datasets for the same inputs (cross-language test).

    python -m ocean_replay.extract ARCHIVE --spec EXTRACT.json --out SMALL.h5 [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import sys

from .reader import Recording
from .spec import Spec
from .writer import Hdf5Writer


def extract(archive, spec, out, overwrite=False) -> dict:
    if not isinstance(spec, Spec):
        spec = Spec.from_file(spec)
    for v in spec.views:                              # extract refers to recorded views by name
        v.method = v.name
    src = Recording(archive)
    try:
        spec.validate(list(src.channels.values()), list(src.views.values()), list(src.axes.values()))
        meta = dict(src.meta)
        meta["extractedFrom"] = {"archive": str(archive), "archiveRunId": meta.get("runId", ""),
                                 "extractSpec": spec.to_dict()}
        prec = meta.get("spec", {}).get("precision", "float32")
        w = Hdf5Writer(out, overwrite=overwrite, precision=prec)
        w.begin(meta, list(src.axes.values()))
        report = {"copied": {}, "not_stored": []}
        for n in spec.channel_names():
            _copy(src, w, n, "channels", lambda v, i, n=n: spec.wants(n, v, i), report)
            report["not_stored"] += _not_stored(src, spec, n)
        for j, v in enumerate(spec.views):
            _copy(src, w, v.name, "views", lambda vv, i, j=j: spec.view_wants(j, vv, i), report)
        w.close()
    finally:
        src.close()
    return report


def _copy(src, w, name, where, wants, report):
    info = src.info(name)
    w.define(info.copy(), where)
    I, V = src.stored(name)
    k = 0
    for p in range(I.shape[0]):
        vals = {a: float(V[p, j]) for j, a in enumerate(info.axes)}
        ix = {a: int(I[p, j]) for j, a in enumerate(info.axes)}
        if not wants(vals, ix):
            continue
        w.append(name, src._value(name, info, p), [int(x) for x in I[p]])
        k += 1
    report["copied"][name] = k


def _not_stored(src, spec, name):
    info = src.info(name)
    _, V = src.stored(name)
    bad = []
    for g in spec.groups:
        if name not in g.channels:
            continue
        for a, rule in g.at.items():
            if a in info.axes:
                k = info.axes.index(a)
                bad += [f"{name}@{a}={v:g}" for v in rule.missing(list(V[:, k]))]
    return bad


def main(argv=None):
    p = argparse.ArgumentParser(description="Extract a smaller replay file from a recording archive.")
    p.add_argument("archive")
    p.add_argument("--spec", required=True, help="extract spec (JSON, same grammar as a recording spec)")
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    rep = extract(a.archive, a.spec, a.out, overwrite=a.overwrite)
    print(json.dumps(rep, indent=2))
    if rep["not_stored"]:
        print("requested but not in the archive:", ", ".join(rep["not_stored"]), file=sys.stderr)


if __name__ == "__main__":
    main()
