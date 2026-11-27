"""Generate and evaluate with the trained unconditional DDPM (EMA weights).

Outputs (in --out/results, copied to --gcs/results):
  samples_ddpm1000.png      grid of faces, textbook DDPM: all L = 1000 reverse steps (Eq. 2.2.4-2.2.5)
  backward_pass.png         for a few faces: x_i (noise -> face) and x_phi(x_i, i) at evenly spaced steps
  forward_pass.png          real TEST faces -> noise with Eq. (2.2.1) at the same levels
  results.json              timings, copying check, FID
Copying check: distance from each generated face to its nearest TRAINING face,
compared with the same distance for real held-out TEST faces. Samples much
closer than test faces would mean the model memorised training images.
FID (Frechet Inception Distance, lower = closer to real faces) uses
torchmetrics; the Inception weights it downloads stay on the VM. If FID fails,
everything else is still produced.

usage: python sample.py --config configs/affectnet64.json --data /path/to/unzipped --out out [--gcs gs://...]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from diffusers.training_utils import EMAModel
from PIL import Image

import data as facedata
import model as M


def grid(imgs: np.ndarray, cols: int) -> Image.Image:
    n, h, w, c = imgs.shape
    rows = int(np.ceil(n / cols))
    g = np.zeros((rows * h, cols * w, c), np.uint8)
    for k in range(n):
        r, cc = divmod(k, cols)
        g[r*h:(r+1)*h, cc*w:(cc+1)*w] = imgs[k]
    return Image.fromarray(g)


def load_ema_unet(cfg, ckpt_path, device):
    s = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if s["config_hash"] != M.config_hash(cfg):
        raise RuntimeError("checkpoint does not match this config")
    unet = M.build_unet(cfg).to(device)
    ema = EMAModel(unet.parameters(), decay=cfg["train"]["ema_decay"])
    ema.load_state_dict(s["ema"])
    ema.to(device)
    ema.copy_to(unet.parameters())
    return unet.eval(), s["step"]


@torch.no_grad()
def reverse(unet, sched, n, size, gen, device, record=0, frames=11):
    """Run the reverse chain from x_L ~ N(0, I); optionally record the first `record` images."""
    x = torch.randn((n, 3, size, size), generator=gen, device=device)
    T = len(sched.timesteps)
    keep = set(np.linspace(0, T - 1, frames).round().astype(int).tolist()) if record else set()
    xs, x0s = [], []
    for k, t in enumerate(sched.timesteps):
        res = sched.step(unet(x, t).sample, t, x, generator=gen)
        if k in keep:
            xs.append(M.to_uint8(x[:record])); x0s.append(M.to_uint8(res.pred_original_sample[:record]))
        x = res.prev_sample
    if record:
        xs.append(M.to_uint8(x[:record])); x0s.append(M.to_uint8(x[:record]))
    return x, xs, x0s


def nn_distance(a: torch.Tensor, b: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """For each row of a: L2 distance to the nearest row of b (per-pixel RMS, images in [0, 1])."""
    out = []
    for i in range(0, len(a), chunk):
        out.append(torch.cdist(a[i:i+chunk], b).min(dim=1).values)
    return torch.cat(out) / np.sqrt(a.shape[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gcs", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cfg, dev = M.load_config(a.config), a.device
    sc = cfg["sample"]
    res_dir = os.path.join(a.out, "results"); os.makedirs(res_dir, exist_ok=True)
    unet, step = load_ema_unet(cfg, os.path.join(a.out, "checkpoint.pt"), dev)
    size = cfg["model"]["sample_size"]
    info = {"trained_steps": step, "config_hash": M.config_hash(cfg)}

    # 1. textbook DDPM samples (+ recorded backward pass)
    sched = M.build_scheduler(cfg, "ddpm"); sched.set_timesteps(sc["ddpm_steps"])
    gen = torch.Generator(dev).manual_seed(sc["seed"])
    t0 = time.time()
    x, xs, x0s = reverse(unet, sched, sc["grid_images"], size, gen, dev,
                         record=sc["trajectory_images"], frames=sc["trajectory_frames"])
    info["ddpm_seconds"] = time.time() - t0
    grid(M.to_uint8(x), 8).save(os.path.join(res_dir, "samples_ddpm1000.png"))
    rows = []
    for j in range(sc["trajectory_images"]):
        rows.append(np.stack([f[j] for f in xs])); rows.append(np.stack([f[j] for f in x0s]))
    grid(np.concatenate(rows), len(xs)).save(os.path.join(res_dir, "backward_pass.png"))

    # 2. forward pass of real test faces (Eq. 2.2.1)
    split = facedata.select_and_split(a.data, cfg["data"]["test_fraction"], cfg["data"]["split_seed"])
    test = torch.from_numpy(facedata.load_images(split.test_files, size)).float() / 127.5 - 1
    x0 = test[:sc["trajectory_images"]].to(dev)
    eps = torch.randn(x0.shape, generator=gen, device=dev)
    L = sched.config.num_train_timesteps
    levels = np.linspace(0, L - 1, sc["trajectory_frames"]).round().astype(int)
    fwd = [M.to_uint8(sched.add_noise(x0, eps, torch.full((len(x0),), int(i), device=dev))) for i in levels]
    grid(np.concatenate([np.stack([f[j] for f in fwd]) for j in range(len(x0))]), len(levels)) \
        .save(os.path.join(res_dir, "forward_pass.png"))
    info["forward_levels_timestep"] = levels.tolist()
    info["forward_alphaBar"] = [M.alpha_bar(sched, int(i)) for i in levels]

    # 3. more samples (fast DDIM) for the copying check and FID
    ddim = M.build_scheduler(cfg, "ddim"); ddim.set_timesteps(sc["fid_ddim_steps"])
    fake = []
    for i in range(0, sc["fid_images"], 250):
        xb, _, _ = reverse(unet, ddim, min(250, sc["fid_images"] - i), size, gen, dev)
        fake.append(((xb.clamp(-1, 1) + 1) / 2).cpu())
    fake = torch.cat(fake)
    train = torch.from_numpy(facedata.load_images(split.train_files, size)).float() / 255
    real = (test + 1) / 2
    ds = lambda t: torch.nn.functional.avg_pool2d(t, 2).flatten(1).to(dev)      # 32x32 for speed
    tr = ds(train)
    d_fake = nn_distance(ds(fake[:1000]), tr).cpu()
    d_real = nn_distance(ds(real[:1000]), tr).cpu()
    info["copy_check"] = {"median_nn_dist_samples": float(d_fake.median()), "median_nn_dist_real_test": float(d_real.median()),
                          "note": "samples much closer to training faces than real test faces are => memorisation"}
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
        for i in range(0, len(real), 250):
            fid.update(real[i:i+250].to(dev), real=True)
        for i in range(0, len(fake), 250):
            fid.update(fake[i:i+250].to(dev), real=False)
        info["fid"] = float(fid.compute()); info["fid_images"] = {"real_test": len(real), "generated": len(fake)}
    except Exception as e:                                      # FID is optional
        info["fid_error"] = str(e)[:300]
    with open(os.path.join(res_dir, "results.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1), flush=True)
    if a.gcs:
        for fn in os.listdir(res_dir):
            M.gcs_cp(os.path.join(res_dir, fn), f"{a.gcs.rstrip('/')}/results/{fn}")


if __name__ == "__main__":
    main()
