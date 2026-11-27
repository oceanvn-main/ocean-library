"""Core tests of ocean_replay (Master plan Sec. 5.7) -- the Python side of the MATLAB
G1 tests: sampling, spec JSON and validation, file round trips, recorder invariance,
laziness, open axes, the size cap, schema violations and extract.

    cd Python && python -m unittest discover -s ocean_replay/tests -t .
"""
import json
import math
import os
import shutil
import tempfile
import unittest
import warnings

import h5py
import numpy as np

from ocean_replay import Recorder, Recording, Sampling, Spec, extract, validate, verify_invariance
from ocean_replay.tests.toy import Toy, full_spec


class Tmp(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def f(self, name):
        return os.path.join(self.dir, name)


class TestSampling(unittest.TestCase):
    def test_rules(self):
        S, ax = Sampling, list(range(1000, -1, -1))
        pick = lambda r, a: [v for v, m in zip(a, r.select(a)) if m]
        self.assertEqual(sorted(pick(S.every(20), ax)), list(range(0, 1001, 20)))
        self.assertEqual(sorted(pick(S.window(51, 1), ax)), [50, 51, 52])
        self.assertEqual(pick(S.every(5, 1, 21), list(range(31))), [1, 6, 11, 16, 21])
        lam = [10, 5, 2, 1, 0.5, 0.2, 0.1]
        self.assertEqual(pick(S.every_index(3), lam), [10, 1, 0.1])
        self.assertEqual(S.list_([50, 5000]).missing(ax), [5000.0])

    def test_spec_json_round_trip(self):
        S = Sampling
        spec = (Spec().add_group("numbers", 0, "energy", {"t": S.all()})
                .add_group("detail", 2, ["img", "mat"], {"t": S.union(S.every(4, 0, 8), S.list_(6))})
                .add_view("noisy", "noisy", {"scale": 0.1}, {"t": S.list_(8)}))
        txt = spec.to_json()
        self.assertEqual(Spec.from_json(txt).to_json(), txt)
        self.assertEqual(json.loads(txt)["groups"][0]["channels"], ["energy"])

    def test_validation(self):
        toy = Toy(); c, v = toy.catalog(); ax = toy.axes_info()
        self.assertEqual(full_spec().check(c, v, ax), [])
        bad = (Spec().add_group("g", 0, ["energy", "nope"]).add_group("h", 0, "tok", {"t": Sampling.all()})
               .add_group("i", 0, "img", {"t": Sampling.list_([4, 7])}).add_view("v", "unknownMethod"))
        issues = bad.check(c, v, ax)
        self.assertEqual(len(issues), 4)
        with self.assertRaises(ValueError):
            bad.validate(c, v, ax)


class TestRecordReplay(Tmp):
    def test_every_kind_round_trips(self):
        toy = Toy()
        Recorder(toy, full_spec("float64"), self.f("toy.h5")).record()
        self.assertEqual(validate(self.f("toy.h5")), [])
        with Recording(self.f("toy.h5")) as r:
            self.assertEqual(r.status, "complete")
            for n in ["energy", "vec", "mat", "img", "seq", "pts", "tok", "label"]:
                info = r.info(n)
                I, V = r.stored(n)
                for p in range(V.shape[0]):
                    coords = {a: V[p, j] for j, a in enumerate(info.axes)}
                    want = next(v for (tt, kk), v in toy.history[n]
                                if all({"t": tt, "k": kk}[a] == coords[a] for a in info.axes))
                    got, _ = r.get(n, coords, "exact")
                    if n == "tok":
                        self.assertEqual(got, want)
                    else:
                        np.testing.assert_array_equal(np.asarray(got).reshape(np.shape(want)), want, err_msg=n)
            self.assertEqual(r.info("label").categories, ["a", "b", "c"])
            self.assertEqual(r.get("tok", {"k": 2})[0], ["w2", "ñ", "中"])
            self.assertFalse(r.info("noisy").in_texts)
            self.assertEqual(list(r.stored("noisy")[1][:, 0]), [8, 2])

    def test_invariance_and_laziness(self):
        ok, rep = verify_invariance(lambda: Toy(), full_spec(), self.f("inv.h5"))
        self.assertTrue(ok)
        toy = Toy()
        Recorder(toy, Spec().add_group("n", 0, "energy"), self.f("lazy.h5")).record()
        self.assertEqual(toy.calls, {"energy": 6})               # once per t, nothing else evaluated

    def test_nearest_never_blends(self):
        S = Sampling
        Recorder(Toy(), Spec().add_group("g", 1, ["img", "mat"], {"t": S.every(4)}), self.f("s.h5")).record()
        with Recording(self.f("s.h5")) as r:
            self.assertEqual(r.get("img", {"t": 5})[1], {"t": 4.0})
            self.assertEqual(r.get("img", {"t": 6})[1], {"t": 8.0})   # tie -> recorded first
            self.assertEqual(r.get("mat", {"t": 6, "k": 2.6})[1], {"t": 8.0, "k": 3.0})
            stored = [r.get("img", {"t": t}, "exact")[0] for t in (8, 4, 0)]
            for q in np.arange(0, 10.5, 0.5):
                v = r.get("img", {"t": q})[0]
                self.assertTrue(any(np.array_equal(v, s) for s in stored))
            with self.assertRaises(KeyError):
                r.get("img", {"t": 6}, "exact")

    def test_open_axis_and_not_visited(self):
        S = Sampling
        spec = Spec().add_group("g", 0, "energy", {"t": S.union(S.every(4), S.list_(7))})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rec = Recorder(Toy(open_axis=True), spec, self.f("open.h5"))
            rec.record()
        self.assertTrue(any("never visited" in str(x.message) for x in w))
        with Recording(self.f("open.h5")) as r:
            self.assertEqual(r.axes["t"].values, T_VALUES)
            self.assertEqual(list(r.stored("energy")[1][:, 0]), [8, 4, 0])
            self.assertEqual(r.meta["notVisited"], ["energy@t=7"])

    def test_size_cap_and_missing_provider(self):
        spec = full_spec(); spec.size_cap_bytes = 10
        with self.assertRaises(ValueError):
            Recorder(Toy(), spec, self.f("cap.h5"))
        self.assertFalse(os.path.exists(self.f("cap.h5")))
        with self.assertRaises(KeyError):
            Recorder(Toy(), Spec().add_group("g", 0, "ghost"), self.f("ghost.h5")).record()

    def test_schema_violations(self):
        Recorder(Toy(), full_spec(), self.f("ok.h5")).record()
        cases = [
            (lambda f: f.attrs.__setitem__("schemaVersion", "2.0"), "unsupported schemaVersion"),
            (lambda f: f["channels/img"].attrs.__setitem__("kind", "blob"), 'unknown kind "blob"'),
            (lambda f: f["channels/img"].attrs.__setitem__("valueShape", np.int32([2, 3, 3])), "does not match valueShape"),
            (lambda f: f["channels/mat/points"].__setitem__(0, np.int32([99, 0])), 'index outside axis "t"'),
            (lambda f: f["views/noisy"].attrs.__setitem__("method", ""), "needs a non-empty method"),
            (lambda f: f["views/noisy"].attrs.__delitem__("inTexts"), "inTexts = 0 or 1"),
            (lambda f: f["channels/img"].attrs.__delitem__("displayRange"), "needs displayRange"),
            (lambda f: f.attrs.__setitem__("meta", "{not json"), "not valid JSON"),
        ]
        for i, (mutate, msg) in enumerate(cases):
            p = self.f(f"bad{i}.h5")
            shutil.copy(self.f("ok.h5"), p)
            with h5py.File(p, "r+") as f:
                mutate(f)
            issues = validate(p)
            self.assertTrue(any(msg in s for s in issues), f"case {i}: {issues}")

    def test_extract(self):
        S = Sampling
        Recorder(Toy(), full_spec(), self.f("a.h5"), run_id="toy-archive").record()
        spec = Spec().add_group("small", 1, ["img", "energy"], {"t": S.list_([4, 6])}).add_view("noisy", "", {}, {"t": S.list_(2)})
        rep = extract(self.f("a.h5"), spec, self.f("x.h5"))
        self.assertEqual(validate(self.f("x.h5")), [])
        self.assertEqual(rep["copied"]["img"], 2)
        with Recording(self.f("a.h5")) as a, Recording(self.f("x.h5")) as x:
            self.assertEqual(sorted(x.names()), ["energy", "img", "noisy"])
            for t in (6, 4):
                np.testing.assert_array_equal(x.get("img", {"t": t}, "exact")[0], a.get("img", {"t": t}, "exact")[0])
            self.assertEqual(x.meta["extractedFrom"]["archiveRunId"], "toy-archive")
            self.assertEqual(x.meta["source"], a.meta["source"])


T_VALUES = [10.0, 8.0, 6.0, 4.0, 2.0, 0.0]

if __name__ == "__main__":
    unittest.main()
