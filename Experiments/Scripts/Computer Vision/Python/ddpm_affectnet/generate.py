"""Generate NEW faces with the trained DDPM (EMA weights) -- no training, no evaluation.

Each seed gives a different face (the seed fixes the starting noise x_L and the noise
added at every step). Writes one PNG per face plus a grid.

usage (Cloud Shell or VM, from this folder):
  python generate.py --config configs/affectnet64.json --checkpoint ~/replay/checkpoint.pt \
      --out ~/faces --seeds 1-16 [--sampler ddpm|ddim] [--steps 50] [--device cpu|cuda]

  --seeds 1-16    a range, or a list: 3,7,11
  --sampler ddpm  the textbook sampler, all L = 1000 steps (Eq. 2.2.4-2.2.5)   [default]
  --sampler ddim  DDIM with --steps steps (default 50): ~20x faster, a different sampler
                  (deterministic given x_L; Lai et al. Sec. 9.2 / Song et al. 2021)
Seeds are generated in batches, so 16 faces cost little more than 1.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

import model as M
from sample import grid, load_ema_unet


def parse_seeds(text: str):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


@torch.no_grad()
def generate(unet, sched, seeds, size, channels, device):
    # one generator per face, so face k depends only on its own seed (same face for the same seed,
    # whatever else is in the batch); the step noise is drawn per face too
    gens = [torch.Generator(device).manual_seed(s) for s in seeds]
    x = torch.stack([torch.randn((channels, size, size), generator=g, device=device) for g in gens])
    for t in sched.timesteps:
        eps = unet(x, t).sample
        outs = [sched.step(eps[k:k + 1], t, x[k:k + 1], generator=gens[k]).prev_sample for k in range(len(seeds))]
        x = torch.cat(outs)
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="1-16")
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps (ignored for ddpm: all L steps)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    cfg = M.load_config(a.config)
    unet, trained = load_ema_unet(cfg, os.path.expanduser(a.checkpoint), a.device)
    sched = M.build_scheduler(cfg, a.sampler)
    sched.set_timesteps(cfg["diffusion"]["num_train_timesteps"] if a.sampler == "ddpm" else a.steps)
    seeds = parse_seeds(a.seeds)
    out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    size, ch = cfg["model"]["sample_size"], cfg["model"]["in_channels"]
    t0 = time.time()
    faces = M.to_uint8(generate(unet, sched, seeds, size, ch, a.device))
    for s, img in zip(seeds, faces):
        Image.fromarray(img).save(os.path.join(out, f"face_seed{s}_{a.sampler}.png"))
    cols = int(np.ceil(np.sqrt(len(seeds))))
    grid(faces, cols).save(os.path.join(out, f"grid_{a.sampler}_seeds_{a.seeds.replace(',', '_')}.png"))
    print(f"{len(seeds)} faces ({a.sampler}, {len(sched.timesteps)} steps, model trained {trained} steps) "
          f"in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
