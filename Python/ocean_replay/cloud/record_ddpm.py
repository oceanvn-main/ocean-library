r"""Record the reverse process of the TRAINED AffectNet DDPM (option D run) for replay in MATLAB.

No training: loads the EMA weights from checkpoint.pt (written by ddpm_affectnet/train.py),
runs the textbook DDPM sampler once with the recorder, and writes a Replay archive.

    python -m ocean_replay.cloud.record_ddpm \
        --ddpm-code  ../Experiments/Scripts/Computer\ Vision/Python/ddpm_affectnet \
        --config     .../ddpm_affectnet/configs/affectnet64.json \
        --checkpoint out/checkpoint.pt \
        --spec       ocean_replay/cloud/spec_ddpm_archive.json \
        --out        archive.h5  --seed 7  [--device cuda]  [--gcs gs://BUCKET/recordings/RUN_ID]

Then (Cloud Shell, free):  python -m ocean_replay extract archive.h5 --spec ocean_replay/cloud/extract_ddpm_small.json --out small.h5
and in MATLAB:             OceanViz.App.Replay.ReplayStudio("small.h5")
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ddpm-code", required=True, help="folder with model.py / sample.py of the training run")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=0, help="sampler steps (0 = all L, the textbook DDPM)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-id", default="")
    p.add_argument("--gcs", default="", help="optional gs:// folder: archive, spec and log are copied there")
    a = p.parse_args(argv)

    sys.path.insert(0, os.path.abspath(a.ddpm_code))
    import model as M                                   # the training run's own code (same UNet, same schedule)
    from sample import load_ema_unet

    from ocean_replay import Recorder, Spec, validate
    from ocean_replay.adapters.diffusers_ddpm import DiffusersDDPMRecordable

    cfg = M.load_config(a.config)
    unet, trained_steps = load_ema_unet(cfg, a.checkpoint, a.device)
    sched = M.build_scheduler(cfg, "ddpm")
    sched.set_timesteps(a.steps or cfg["diffusion"]["num_train_timesteps"])
    size = cfg["model"]["sample_size"]
    d = DiffusersDDPMRecordable(unet, sched, (cfg["model"]["in_channels"], size, size), seed=a.seed, device=a.device,
                                model_name=f"{cfg['run_name']} (EMA, {trained_steps} training steps)")
    spec = Spec.from_file(a.spec)
    rec = Recorder(d, spec, a.out, overwrite=True, run_id=a.run_id, seeds={"sampler": a.seed},
                   title=f"{cfg['run_name']} -- reverse process")
    print(f"estimated size: {rec.estimated_bytes / 1e6:.1f} MB", flush=True)
    t0 = time.time()
    rec.record()
    issues = validate(a.out)
    print(f"recorded in {time.time() - t0:.0f} s -> {a.out}; schema: {'valid' if not issues else issues}", flush=True)
    if a.gcs:
        for f in (a.out, a.spec):
            subprocess.run(["gcloud", "storage", "cp", f, a.gcs.rstrip("/") + "/"], check=False)
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
