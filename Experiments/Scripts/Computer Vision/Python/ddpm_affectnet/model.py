"""Model, noise schedule and book notation for the unconditional DDPM.

The network and scheduler are the production `diffusers` components; this file
only builds them from the config and translates names to Lai et al., "The
Principles of Diffusion Models", Sec. 2.2:

  diffusers                          book
  ---------------------------------  -----------------------------------------
  scheduler.betas[i]                 beta_i^2  (Ho's variances; beta_i is a STD in the book)
  scheduler.alphas[i]                alpha_i^2 = 1 - beta_i^2
  scheduler.alphas_cumprod[i]        alphaBar_i^2
  add_noise(x0, eps, i)              x_i = alphaBar_i x0 + sqrt(1 - alphaBar_i^2) eps      Eq. (2.2.1)
  unet(x_i, i).sample                eps_phi(x_i, i)                                        (eps-prediction)
  step(...).pred_original_sample     x_phi(x_i, i)
  step(...).prev_sample  (DDPM)      x_{i-1} = mu + sigma(i) z                               Eq. (2.2.4)-(2.2.5)
  training loss                      || eps - eps_phi(x_i, i) ||^2                            Eq. (2.2.10)
  (diffusers counts timesteps 0..L-1; the book's step i = timestep + 1)
"""
from __future__ import annotations

import hashlib
import json
import subprocess

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def config_hash(cfg: dict) -> str:
    """Fingerprint of everything that defines the trained model (not the sampling settings)."""
    key = {k: cfg[k] for k in ("data", "model", "diffusion", "train")}
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def build_unet(cfg: dict) -> UNet2DModel:
    m = dict(cfg["model"])
    for k in ("block_out_channels", "down_block_types", "up_block_types"):
        m[k] = tuple(m[k])
    return UNet2DModel(**m)


def build_scheduler(cfg: dict, kind: str = "ddpm"):
    d = cfg["diffusion"]
    cls = DDIMScheduler if kind == "ddim" else DDPMScheduler
    return cls(**d)


def alpha_bar(scheduler, timestep: int) -> float:
    """Book alphaBar_i (the SIGNAL SCALE) at diffusers timestep t (= book step t + 1)."""
    return float(scheduler.alphas_cumprod[timestep].sqrt())


def autocast_dtype(precision: str, device: str):
    """bf16 on GPUs with native support (compute capability >= 8, e.g. L4); else fp32 (e.g. T4)."""
    if device != "cuda" or precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else None


def to_uint8(x: torch.Tensor):
    """[-1, 1] (N, C, H, W) tensor -> uint8 numpy (N, H, W, C)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


# ---------------- Cloud Storage (gcloud CLI, present on Google's VM images) ----------------
def gcs_cp(src: str, dst: str) -> bool:
    """Copy local <-> gs://; returns False (and does not raise) on failure."""
    r = subprocess.run(["gcloud", "storage", "cp", "-q", src, dst], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[gcs] copy {src} -> {dst} failed: {r.stderr.strip()[:300]}", flush=True)
    return r.returncode == 0


def gcs_exists(uri: str) -> bool:
    return subprocess.run(["gcloud", "storage", "ls", uri], capture_output=True).returncode == 0
