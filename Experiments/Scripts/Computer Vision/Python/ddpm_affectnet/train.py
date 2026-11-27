"""Train an UNCONDITIONAL DDPM on AffectNet faces (Ho et al. 2020; Lai et al. Sec. 2.2).

One training step (Eq. 2.2.10, "L_simple"):
    x0            a batch of real faces in [-1, 1] (random horizontal flips)
    i, eps        a random step per face (uniform over 1..L) and Gaussian noise
    x_i           = alphaBar_i x0 + sqrt(1 - alphaBar_i^2) eps          Eq. (2.2.1)  (scheduler.add_noise)
    loss          = mean || eps - eps_phi(x_i, i) ||^2                   Eq. (2.2.10)
    AdamW step (linear warm-up, then constant LR as in Ho et al.), gradient clipping,
    EMA of the weights (the EMA weights are the ones used for sampling).

Checkpoints (model, EMA, optimiser, step, every RNG state, config hash) are
written atomically every `checkpoint_every` steps -- and on SIGTERM (VM shutdown
or time limit) -- and copied to Cloud Storage when --gcs is given. A rerun with
the same config RESUMES from the newest checkpoint (local, else Cloud Storage);
a checkpoint from a different config is refused.

usage: python train.py --config configs/affectnet64.json --data /path/to/unzipped --out out [--gcs gs://bucket/runs/name]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import time

import numpy as np
import torch
from diffusers.training_utils import EMAModel
from PIL import Image

import data as facedata
import model as M


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True, help="folder containing the unzipped Train/ and Test/")
    p.add_argument("--out", required=True)
    p.add_argument("--gcs", default="", help="gs://bucket/runs/<name> (optional)")
    p.add_argument("--steps", type=int, default=0, help="override train.steps (e.g. a short timing run)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class Trainer:
    def __init__(self, cfg: dict, faces: np.ndarray, out: str, gcs: str = "", device: str = "cuda"):
        self.cfg, self.out, self.gcs, self.device = cfg, out, gcs.rstrip("/"), device
        t = cfg["train"]
        torch.manual_seed(t["seed"])
        self.unet = M.build_unet(cfg).to(device)
        self.sched = M.build_scheduler(cfg, "ddpm")
        self.opt = torch.optim.AdamW(self.unet.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        self.ema = EMAModel(self.unet.parameters(), decay=t["ema_decay"], use_ema_warmup=True, power=0.75)
        self.faces = torch.from_numpy(faces)                      # uint8 (N, 3, H, W), stays on the CPU
        self.gen = torch.Generator().manual_seed(t["seed"] + 1)   # batches, flips, steps, noise
        self.step = 0
        self.amp = M.autocast_dtype(t["precision"], device)
        self.hash = M.config_hash(cfg)
        os.makedirs(out, exist_ok=True)
        self.ckpt = os.path.join(out, "checkpoint.pt")
        self.stop_requested = False

    # ---------------- one step of Eq. (2.2.10) ----------------
    def batch(self):
        t = self.cfg["train"]
        idx = torch.randint(len(self.faces), (t["batch_size"],), generator=self.gen)
        x0 = self.faces[idx].float() / 127.5 - 1.0                # [0, 255] -> [-1, 1]
        if self.cfg["data"]["hflip"]:
            flip = torch.rand(len(idx), generator=self.gen) < 0.5
            x0[flip] = x0[flip].flip(-1)
        L = self.sched.config.num_train_timesteps
        timesteps = torch.randint(L, (len(idx),), generator=self.gen)   # diffusers 0..L-1 = book i - 1
        eps = torch.randn(x0.shape, generator=self.gen)
        return x0.to(self.device), eps.to(self.device), timesteps.to(self.device)

    def train_step(self) -> float:
        t = self.cfg["train"]
        x0, eps, ts = self.batch()
        xi = self.sched.add_noise(x0, eps, ts)                     # Eq. (2.2.1)
        lr = t["lr"] * min(1.0, (self.step + 1) / max(1, t["warmup_steps"]))
        for g in self.opt.param_groups:
            g["lr"] = lr
        with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu", dtype=self.amp,
                            enabled=self.amp is not None):
            eps_hat = self.unet(xi, ts).sample                     # eps_phi(x_i, i)
        loss = torch.nn.functional.mse_loss(eps_hat.float(), eps)  # Eq. (2.2.10)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if t["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), t["grad_clip"])
        self.opt.step()
        self.ema.step(self.unet.parameters())
        self.step += 1
        return loss.item()

    # ---------------- checkpoint / resume ----------------
    def state(self) -> dict:
        return {"step": self.step, "config_hash": self.hash, "config": self.cfg,
                "unet": self.unet.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
                "gen": self.gen.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device == "cuda" else []}

    def save(self):
        tmp = self.ckpt + ".tmp"
        torch.save(self.state(), tmp)
        os.replace(tmp, self.ckpt)                                  # atomic: never a half-written file
        if self.gcs:
            M.gcs_cp(self.ckpt, f"{self.gcs}/checkpoint.pt")
            for f in ("train_log.csv",):
                if os.path.exists(os.path.join(self.out, f)):
                    M.gcs_cp(os.path.join(self.out, f), f"{self.gcs}/{f}")
        print(f"[checkpoint] step {self.step} saved", flush=True)

    def try_resume(self) -> bool:
        if not os.path.exists(self.ckpt) and self.gcs and M.gcs_exists(f"{self.gcs}/checkpoint.pt"):
            M.gcs_cp(f"{self.gcs}/checkpoint.pt", self.ckpt)
            if M.gcs_exists(f"{self.gcs}/train_log.csv"):
                M.gcs_cp(f"{self.gcs}/train_log.csv", os.path.join(self.out, "train_log.csv"))
        if not os.path.exists(self.ckpt):
            return False
        # Load into CPU memory: the RNG states must stay CPU ByteTensors; the model,
        # EMA and optimiser are moved to the training device below.
        s = torch.load(self.ckpt, map_location="cpu", weights_only=False)
        if s["config_hash"] != self.hash:
            raise RuntimeError(f"Checkpoint {self.ckpt} was trained with a different config "
                               f"({s['config_hash']} != {self.hash}); use a new --out / run name.")
        self.unet.load_state_dict(s["unet"])
        self.ema.load_state_dict(s["ema"])
        self.ema.to(self.device)
        self.opt.load_state_dict(s["opt"])                        # moves its state to the parameters' device
        self.gen.set_state(s["gen"])
        torch.set_rng_state(s["torch_rng"])
        if self.device == "cuda" and s["cuda_rng"]:
            torch.cuda.set_rng_state_all(s["cuda_rng"])
        self.step = s["step"]
        print(f"[resume] continuing from step {self.step}", flush=True)
        return True

    # ---------------- previews ----------------
    @torch.no_grad()
    def preview(self):
        """A few EMA samples with fast DDIM, to watch training progress."""
        t = self.cfg["train"]
        ema_unet = M.build_unet(self.cfg).to(self.device)
        self.ema.copy_to(ema_unet.parameters())
        ema_unet.eval()
        sch = M.build_scheduler(self.cfg, "ddim")
        sch.set_timesteps(t["preview_ddim_steps"])
        g = torch.Generator(self.device).manual_seed(123)
        s = self.cfg["model"]["sample_size"]
        x = torch.randn((t["preview_images"], 3, s, s), generator=g, device=self.device)
        for ts in sch.timesteps:
            x = sch.step(ema_unet(x, ts).sample, ts, x).prev_sample
        imgs = M.to_uint8(x)
        n = len(imgs); cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
        grid = np.zeros((rows * s, cols * s, 3), np.uint8)
        for k in range(n):
            r, c = divmod(k, cols)
            grid[r*s:(r+1)*s, c*s:(c+1)*s] = imgs[k]
        path = os.path.join(self.out, f"preview_step{self.step:06d}.png")
        Image.fromarray(grid).save(path)
        if self.gcs:
            M.gcs_cp(path, f"{self.gcs}/previews/{os.path.basename(path)}")

    # ---------------- the loop ----------------
    def fit(self, steps: int):
        t = self.cfg["train"]
        log_path = os.path.join(self.out, "train_log.csv")
        new_log = not os.path.exists(log_path)
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop_requested", True))
        self.unet.train()
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_log:
                w.writerow(["step", "loss", "lr", "sec_per_step"])
            acc, n, t0 = 0.0, 0, time.time()
            while self.step < steps:
                acc += self.train_step(); n += 1
                if self.step % t["log_every"] == 0 or self.step == steps:
                    sps = (time.time() - t0) / n
                    lr = self.opt.param_groups[0]["lr"]
                    w.writerow([self.step, acc / n, lr, sps]); f.flush()
                    print(f"step {self.step:6d}/{steps}  loss {acc / n:.4f}  {sps:.3f} s/step  "
                          f"(~{sps * (steps - self.step) / 60:.0f} min left)", flush=True)
                    acc, n, t0 = 0.0, 0, time.time()
                if self.step % t["sample_every"] == 0:
                    self.preview(); self.unet.train()
                if self.step % t["checkpoint_every"] == 0 or self.step == steps:
                    self.save()
                if self.stop_requested:
                    print("[stop] SIGTERM received: saving and exiting", flush=True)
                    self.save()
                    return False
        return True


def main():
    a = parse()
    cfg = M.load_config(a.config)
    steps = a.steps or cfg["train"]["steps"]
    split = facedata.select_and_split(a.data, cfg["data"]["test_fraction"], cfg["data"]["split_seed"])
    print(json.dumps(split.report, indent=1), flush=True)
    faces = facedata.load_images(split.train_files, cfg["data"]["image_size"])
    tr = Trainer(cfg, faces, a.out, a.gcs, a.device)
    print(f"U-Net: {sum(p.numel() for p in tr.unet.parameters()) / 1e6:.1f}M parameters | device {a.device} | "
          f"autocast {tr.amp} | config {tr.hash}", flush=True)
    tr.try_resume()
    with open(os.path.join(a.out, "data_report.json"), "w") as f:
        json.dump(split.report, f, indent=1)
    finished = tr.fit(steps)
    print("TRAINING FINISHED" if finished else "TRAINING STOPPED (resumable)", flush=True)
    raise SystemExit(0 if finished else 3)


if __name__ == "__main__":
    main()
