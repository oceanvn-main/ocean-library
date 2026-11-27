"""Training resumes exactly: 4 steps straight == 2 steps, checkpoint, restart, 2 steps. (VM: needs diffusers.)
Runs on the CPU and, where available, on the GPU (the resume path differs: RNG states must stay on the CPU).
"""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
import train  # noqa: E402

CFG = {
    "data": {"image_size": 8, "test_fraction": 0.2, "split_seed": 0, "hflip": True},
    "model": {"sample_size": 8, "in_channels": 3, "out_channels": 3, "layers_per_block": 1,
              "block_out_channels": [8, 16], "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
              "up_block_types": ["AttnUpBlock2D", "UpBlock2D"], "norm_num_groups": 4, "dropout": 0.1},
    "diffusion": {"num_train_timesteps": 50, "beta_schedule": "linear", "beta_start": 1e-4, "beta_end": 0.02,
                  "prediction_type": "epsilon", "clip_sample": True},
    "train": {"steps": 4, "batch_size": 4, "lr": 1e-3, "warmup_steps": 2, "weight_decay": 0.0, "grad_clip": 1.0,
              "ema_decay": 0.99, "precision": "fp32", "seed": 0, "log_every": 1, "checkpoint_every": 1000,
              "sample_every": 1000, "preview_images": 4, "preview_ddim_steps": 5},
    "sample": {},
}


def _losses(run_dir):
    """{step: loss} from train_log.csv (a resumed run appends to the same file)."""
    import csv
    with open(run_dir / "train_log.csv") as f:
        return {int(r["step"]): float(r["loss"]) for r in csv.DictReader(f)}


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_resume_is_exact(tmp_path, device):
    faces = np.random.default_rng(0).integers(0, 256, (10, 3, 8, 8), dtype=np.uint8)
    a = train.Trainer(copy.deepcopy(CFG), faces, str(tmp_path / "a"), device=device)
    a.fit(4)
    b = train.Trainer(copy.deepcopy(CFG), faces, str(tmp_path / "b"), device=device)
    b.fit(2)                                                     # saves at the last step
    c = train.Trainer(copy.deepcopy(CFG), faces, str(tmp_path / "b"), device=device)
    assert c.try_resume() and c.step == 2
    c.fit(4)
    # CPU: bit-exact. GPU: cuDNN may use non-deterministic kernels; Adam divides by sqrt(v), which
    # amplifies last-digit gradient differences of near-zero-gradient weights up to ~lr (1e-3).
    # So on the GPU the proof of a correct resume is: identical logged losses + weights within ~lr.
    same = torch.equal if device == "cpu" else (lambda u, v: torch.allclose(u, v, rtol=1e-2, atol=2e-3))
    if device != "cpu":
        la = _losses(tmp_path / "a"); lc = _losses(tmp_path / "b")
        assert sorted(la) == sorted(lc) == [1, 2, 3, 4]
        for step in (3, 4):                                      # the steps AFTER the restart
            assert abs(la[step] - lc[step]) < 1e-3, (step, la[step], lc[step])
    for pa, pc in zip(a.unet.parameters(), c.unet.parameters()):
        assert same(pa, pc)
    for sa, sc in zip(a.ema.shadow_params, c.ema.shadow_params):
        assert same(sa, sc)
    assert all(p.device.type == device for p in c.ema.shadow_params)


def test_resume_refuses_other_config(tmp_path):
    faces = np.zeros((4, 3, 8, 8), np.uint8)
    a = train.Trainer(copy.deepcopy(CFG), faces, str(tmp_path), device="cpu")
    a.fit(1)
    other = copy.deepcopy(CFG); other["train"]["lr"] = 5e-4
    b = train.Trainer(other, faces, str(tmp_path), device="cpu")
    with pytest.raises(RuntimeError):
        b.try_resume()
