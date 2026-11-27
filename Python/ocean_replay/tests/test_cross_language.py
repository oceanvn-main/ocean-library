"""MATLAB <-> Python parity (Master plan Sec. 5.7).

Reads the fixtures MATLAB wrote (OceanTest.OceanReplay.makeCrossLanguageFixtures):
  * toy_matlab.h5 read here gives exactly the values MATLAB stored (toy_matlab_expected.json);
  * both schemas accept it;
  * every sampling rule selects the same points as MATLAB (sampling_cases.json);
  * Python's extract of toy_matlab.h5 with extract_spec.json equals MATLAB's (toy_matlab_extract.h5).
The reverse direction (MATLAB reading Python files) is +OceanTest/+OceanReplay/CrossLanguageTest.m,
using the files make_fixtures.py writes.
"""
import json
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from ocean_replay import Recording, Sampling, extract, validate

FX = os.path.join(os.path.dirname(__file__), "..", "..", "..", "+OceanTest", "+OceanReplay", "fixtures")
FX = os.path.normpath(FX)


def fx(name):
    return os.path.join(FX, name)


class TestReadMatlab(unittest.TestCase):
    def test_values_equal_matlab(self):
        self.assertEqual(validate(fx("toy_matlab.h5")), [])
        with open(fx("toy_matlab_expected.json"), encoding="utf-8") as f:
            exp = json.load(f)
        with Recording(fx("toy_matlab.h5")) as r:
            self.assertEqual(r.meta["source"]["language"], "matlab")
            self.assertEqual(sorted(r.names()), sorted(exp["channels"]))
            for n, pts in exp["channels"].items():
                pts = pts if isinstance(pts, list) else [pts]
                info = r.info(n)
                for e in pts:
                    got, _ = r.get(n, e["coords"], "exact")
                    if "tokens" in e:
                        self.assertEqual(got, e["tokens"], n)
                        continue
                    shape = np.atleast_1d(e["shape"]).astype(int).tolist()
                    want = np.asarray(e["data"], np.float64).reshape(shape, order="F")   # MATLAB column-major
                    np.testing.assert_array_equal(np.asarray(got, np.float64).reshape(shape), want, err_msg=n)
            self.assertIsInstance(r.info("noisy").params, dict)
            self.assertEqual(r.info("noisy").params["scale"], 0.1)

    def test_sampling_matches_matlab(self):
        with open(fx("sampling_cases.json"), encoding="utf-8") as f:
            cases = json.load(f)
        self.assertGreater(len(cases), 20)
        for c in cases:
            axis = np.atleast_1d(c["axis"]).tolist()
            mask = [bool(m) for m in np.atleast_1d(c["mask"])]
            self.assertEqual(Sampling.from_dict(c["rule"]).select(axis), mask, c["rule"])

    def test_extract_equals_matlab_extract(self):
        d = tempfile.mkdtemp()
        try:
            out = os.path.join(d, "py_extract.h5")
            extract(fx("toy_matlab.h5"), fx("extract_spec.json"), out)
            self.assertEqual(validate(out), [])
            assert_same_datasets(self, fx("toy_matlab_extract.h5"), out)
        finally:
            shutil.rmtree(d, ignore_errors=True)


def assert_same_datasets(tc, a_path, b_path):
    """Every channel/view/axis dataset of two files is identical (values, points, shapes)."""
    with h5py.File(a_path, "r") as a, h5py.File(b_path, "r") as b:
        for grp in ("axes", "channels", "views"):
            names_a = sorted(a.get(grp, {}).keys()); names_b = sorted(b.get(grp, {}).keys())
            tc.assertEqual(names_a, names_b, grp)
            for n in names_a:
                if grp == "axes":
                    np.testing.assert_array_equal(a[grp][n][()], b[grp][n][()])
                    continue
                for ds in ("data", "points"):
                    ina, inb = ds in a[grp][n], ds in b[grp][n]
                    tc.assertEqual(ina, inb, f"{grp}/{n}/{ds}")
                    if ina:
                        x, y = a[grp][n][ds][()], b[grp][n][ds][()]
                        tc.assertEqual(x.shape, y.shape, f"{grp}/{n}/{ds}")
                        tc.assertTrue(np.array_equal(x, y), f"{grp}/{n}/{ds}")


if __name__ == "__main__":
    unittest.main()
