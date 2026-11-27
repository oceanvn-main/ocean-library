"""The adapter against the REAL diffusers classes (run where diffusers is installed:
Cloud Shell or the GPU VM -- skipped locally). A small randomly initialised UNet2DModel
is built in memory: nothing is downloaded.

  * sampling with the recorder == sample.py's plain loop (bit-identical on CPU);
  * varNext == DDPMScheduler._get_variance(t); alphaBar == sqrt(alphas_cumprod[t]);
  * prev_sample == mu + sqrt(sigma^2(i)) z, with z replayed from the same generator
    -> the recorded mu is exactly the mean diffusers used (Eq. 2.2.4), up to float32.

    cd Python && python -m unittest ocean_replay.tests.test_diffusers_cloud -v
"""
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

try:
    import torch
    from diffusers import DDPMScheduler, UNet2DModel
    from diffusers.utils.torch_utils import randn_tensor
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "diffusers not installed (run in Cloud Shell / on the VM)")
class TestRealDiffusers(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        torch.manual_seed(0)
        self.unet = UNet2DModel(sample_size=16, in_channels=3, out_channels=3, layers_per_block=1,
                                block_out_channels=(32, 64), down_block_types=("DownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "UpBlock2D")).eval()
        self.sched = DDPMScheduler(num_train_timesteps=50, beta_schedule="linear", beta_start=1e-4, beta_end=0.02,
                                   clip_sample=True, prediction_type="epsilon")
        self.sched.set_timesteps(50)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def recordable(self):
        from ocean_replay.adapters.diffusers_ddpm import DiffusersDDPMRecordable
        return DiffusersDDPMRecordable(self.unet, self.sched, (3, 16, 16), seed=7, model_name="random tiny UNet2DModel")

    def spec(self):
        from ocean_replay import Spec
        return (Spec(precision="float64").add_group("n", 0, ["alphaBar", "varNext", "identityResidual"])
                .add_group("i", 1, ["x", "mu", "x0hat", "eps"]).add_group("r", 1, "result"))

    def test_identical_to_plain_loop(self):
        from ocean_replay import Recorder, Recording
        f = os.path.join(self.dir, "d.h5")
        Recorder(self.recordable(), self.spec(), f).record()
        gen = torch.Generator().manual_seed(7)                    # sample.py's reverse(), one image
        x = torch.randn((1, 3, 16, 16), generator=gen)
        with torch.no_grad():
            for t in self.sched.timesteps:
                x = self.sched.step(self.unet(x, t).sample, t, x, generator=gen).prev_sample
        with Recording(f) as r:
            np.testing.assert_array_equal(r.get("result")[0], x[0].permute(1, 2, 0).double().numpy())

    def test_notation_mu_and_variance(self):
        from ocean_replay import Recorder, Recording
        f = os.path.join(self.dir, "n.h5")
        Recorder(self.recordable(), self.spec(), f).record()
        ac = self.sched.alphas_cumprod
        gen = torch.Generator().manual_seed(7)
        x = randn_tensor((1, 3, 16, 16), generator=gen)            # replay the generator: x_L, then z per step
        with Recording(f) as r:
            steps = [int(t) for t in self.sched.timesteps]
            for k, t in enumerate(steps):
                i = t + 1
                self.assertAlmostEqual(r.get("alphaBar", {"step": i})[0], math.sqrt(float(ac[t])), places=6)
                if t > 0:
                    self.assertAlmostEqual(r.get("varNext", {"step": i})[0], float(self.sched._get_variance(t)), places=6)
                    z = randn_tensor((1, 3, 16, 16), generator=gen)
                    mu = torch.from_numpy(r.get("mu", {"step": i})[0]).permute(2, 0, 1)[None]
                    nxt = torch.from_numpy(r.get("x", {"step": steps[k + 1] + 1})[0]).permute(2, 0, 1)[None]
                    var = r.get("varNext", {"step": i})[0]
                    torch.testing.assert_close(nxt, mu + math.sqrt(var) * z.double(), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
