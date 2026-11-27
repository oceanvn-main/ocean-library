"""DiffusersDDPMRecordable with a tiny network + the DDPM step of diffusers (fake_ddpm.py).

  * recording does not change the final sample (verify_invariance);
  * recorded x_i equal the plain loop's (sample.py's loop), step axis = t + 1;
  * recorded mu / sigma^2(i) equal what the scheduler used (Eq. 2.2.4-2.2.5);
  * book notation: alphaBar = sqrt(alphas_cumprod); x_i = alphaBar x_phi + sigma eps_phi
    without clipping; the occlusion view is a normalised map; the file validates.
"""
import math
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch

from ocean_replay import Recorder, Recording, Sampling, Spec, validate, verify_invariance
from ocean_replay.adapters.diffusers_ddpm import DiffusersDDPMRecordable
from ocean_replay.tests.fake_ddpm import FakeDDPMScheduler, TinyUNet


def make(clip=True, steps=50):
    s = FakeDDPMScheduler(L=50, clip_sample=clip); s.set_timesteps(steps)
    return DiffusersDDPMRecordable(TinyUNet(), s, (3, 8, 8), seed=3, model_name="tiny")


def spec_all():
    return (Spec(precision="float64")
            .add_group("numbers", 0, ["tLevel", "tNext", "alphaBar", "alphaBarNext", "sigma", "varNext", "epsRms",
                                      "identityResidual"])
            .add_group("images", 1, ["x", "eps", "x0hat", "mu"])
            .add_group("final", 1, "result")
            .add_view("occlusion", "occlusion", {"size": 2, "stride": 2}, {"step": Sampling.list_([30, 5])}))


class TestAdapter(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_invariance_and_trajectory(self):
        f = os.path.join(self.dir, "d.h5")
        ok, rep = verify_invariance(lambda: make(), spec_all(), f, equal=lambda a, b: torch.equal(a["x"], b["x"]))
        self.assertTrue(ok)
        self.assertEqual(validate(f), [])
        # the plain loop of sample.py
        d = make(); gen = torch.Generator().manual_seed(3)
        x = torch.randn((1, 3, 8, 8), generator=gen)
        xs = {}
        with torch.no_grad():
            for t in d.sched.timesteps:
                xs[int(t) + 1] = x.clone()
                x = d.sched.step(d.unet(x, t).sample, t, x, generator=gen).prev_sample
        with Recording(f) as r:
            self.assertEqual(r.axes["step"].values, [float(v) for v in range(50, 0, -1)])
            for i in (50, 30, 1):
                got, _ = r.get("x", {"step": i}, "exact")
                np.testing.assert_allclose(got, xs[i][0].permute(1, 2, 0).double().numpy(), atol=1e-7)
            np.testing.assert_allclose(r.get("result")[0], x[0].permute(1, 2, 0).double().numpy(), atol=1e-7)
            self.assertEqual(r.meta["layout"], "diffusion")
            occ, _ = r.get("occlusion", {"step": 30}, "exact")
            self.assertEqual(occ.shape, (8, 8, 1)); self.assertAlmostEqual(float(occ.max()), 1.0)

    def test_mu_var_and_notation(self):
        f = os.path.join(self.dir, "n.h5")
        d = make(clip=False)
        Recorder(d, spec_all(), f).record()
        ac = d.sched.alphas_cumprod.double()
        with Recording(f) as r:
            for i in (50, 25, 2, 1):
                t = i - 1
                mu, _ = r.get("mu", {"step": i}, "exact")
                # the scheduler computes Eq. (2.2.4) in float32 (like diffusers), the recorder in float64
                np.testing.assert_allclose(mu, d.sched.means[t][0].permute(1, 2, 0).double().numpy(), rtol=1e-4, atol=1e-5)
                self.assertAlmostEqual(r.get("varNext", {"step": i})[0], d.sched.vars[t], places=7)
                self.assertAlmostEqual(r.get("alphaBar", {"step": i})[0], math.sqrt(float(ac[t])), places=12)
                self.assertAlmostEqual(r.get("tLevel", {"step": i})[0], i / 50, places=12)
            _, resid = r.series("identityResidual")
            self.assertLess(float(resid.max()), 1e-5)               # no clipping -> the identity holds

    def test_ddim_like_subsampled_steps(self):
        f = os.path.join(self.dir, "s.h5")
        d = make(steps=10)                                         # every 5th timestep, like a 50-step DDIM grid
        Recorder(d, Spec().add_group("n", 0, ["alphaBar", "tNext"]), f).record()
        with Recording(f) as r:
            self.assertEqual(r.axes["step"].values, [float(v) for v in range(46, 0, -5)])
            self.assertAlmostEqual(r.get("tNext", {"step": 46})[0], 41 / 50, places=6)   # float32 file


if __name__ == "__main__":
    unittest.main()
