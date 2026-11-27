"""Record a diffusers DDPM reverse process for replay in MATLAB (Replay Studio).

Twin of OceanDL.Generative.Inspect.DiffusionRecordable: SAME channel names, kinds and
meanings, so MATLAB's Diffusion layout opens these recordings unchanged. Notation:
Lai et al., "The Principles of Diffusion Models", Sec. 2.2.

diffusers <-> book (tested in test_diffusers_cloud.py):
    timestep t (0-based)        -> step index i = t + 1        (axis "step": 1000 .. 1)
    alphas_cumprod[t]           =  alphaBar_i^2                (alphaBar = sqrt(alphas_cumprod))
    betas[t]                    =  beta_i^2
    model output (epsilon)      =  eps_phi(x_i, i)
    pred_original_sample        =  x_phi(x_i, i)  (clipped to [-1, 1] when clip_sample)
    mean of prev_sample         =  mu, Eq. (2.2.4) with x -> x_phi
    variance ("fixed_small")    =  sigma^2(i) = tildeBeta_i^2, Eq. (2.2.5)

The loop is exactly sample.py's:  eps = unet(x, t).sample;  res = scheduler.step(eps, t, x, generator=gen).
Recording reads values only; views run without gradients and never touch `gen`, so the
final sample is identical with and without recording (verify_invariance).

Only duck typing is used (unet(x, t).sample, scheduler.timesteps / alphas_cumprod /
config.num_train_timesteps / step), so this module does not import diffusers.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..info import Axis, ChannelInfo, ViewInfo
from ..recorder import IRecordable


def _hwc(x: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) tensor -> H x W x C float64 numpy (logical image of the Replay format)."""
    return x[0].detach().permute(1, 2, 0).to(torch.float64).cpu().numpy()


class DiffusersDDPMRecordable(IRecordable):
    def __init__(self, unet, scheduler, image_shape, seed=0, device="cpu", data_range=(-1.0, 1.0),
                 model_name="", references=None):
        """image_shape = (C, H, W); the scheduler must already have set_timesteps(...) called."""
        self.unet, self.sched = unet, scheduler
        self.C, self.H, self.W = (int(s) for s in image_shape)
        self.seed, self.device = int(seed), device
        self.data_range = [float(data_range[0]), float(data_range[1])]
        self.model_name = model_name
        self.references = references or [
            "Lai et al., The Principles of Diffusion Models, Sec. 2.2.2-2.2.6, Eq. (2.2.4)-(2.2.5)",
            "Ho et al. 2020 (DDPM); diffusers DDPMScheduler"]
        self.L = int(scheduler.config.num_train_timesteps)
        self.timesteps = [int(t) for t in scheduler.timesteps]
        self.step_values = [t + 1 for t in self.timesteps]

    # ---- IRecordable
    def axes_info(self):
        return [Axis("step", self.step_values, "i",
                     description=f"reverse-process step: book step index i = t + 1 (diffusers timestep t), L = {self.L} "
                                 "(Lai et al. Sec. 2.2)")]

    def catalog(self):
        C = ChannelInfo
        S, R = [self.H, self.W, self.C], self.data_range
        c = [C("x", "image", ["step"], S, display_range=R, symbol="x_i", symbol_latex=r"$x_i$",
               description="the sampler's current sample -- the INPUT of the network at step i"),
             C("eps", "image", ["step"], S, display_range=[-2, 2], symbol="eps_phi(x_i, i)", symbol_latex=r"$\epsilon_\phi(x_i, i)$",
               description="the network's estimate of the noise in x_i (OUTPUT); grey = 0, white/black = +-2"),
             C("x0hat", "image", ["step"], S, display_range=R, symbol="x_phi(x_i, i)", symbol_latex=r"$x_\phi(x_i, i)$",
               description="the implied clean-sample estimate (x_i - sigma_i eps_phi) / alphaBar_i (OUTPUT), "
                           "clipped to the data range when the scheduler clips (diffusers clip_sample)"),
             C("mu", "image", ["step"], S, display_range=R, symbol="mu", symbol_latex=r"$\mu$",
               description="mean of p_phi(x_{i-1} | x_i), Eq. (2.2.4) with x -> x_phi: where the next sample is centred"),
             C("tLevel", "scalar", ["step"], symbol="t_i", symbol_latex=r"$t_i$", description="continuous noise level of step i (t = i / L)"),
             C("tNext", "scalar", ["step"], symbol="t_{i-1}", symbol_latex=r"$t_{i-1}$", description="noise level after the step"),
             C("alphaBar", "scalar", ["step"], symbol="alphaBar_i", symbol_latex=r"$\bar{\alpha}_i$",
               description="signal scale of x_i (book alphaBar_i = sqrt(diffusers alphas_cumprod))"),
             C("alphaBarNext", "scalar", ["step"], symbol="alphaBar_{i-1}", symbol_latex=r"$\bar{\alpha}_{i-1}$", description="signal scale after the step"),
             C("sigma", "scalar", ["step"], symbol="sqrt(1 - alphaBar_i^2)", symbol_latex=r"$\sqrt{1-\bar{\alpha}_i^2}$", description="noise scale of x_i"),
             C("varNext", "scalar", ["step"], symbol="sigma^2(i)", symbol_latex=r"$\sigma^2(i)$",
               description="variance of the fresh noise added by DDPM, tildeBeta_i^2 (Eq. 2.2.5)"),
             C("epsRms", "scalar", ["step"], symbol="rms(eps_phi)", symbol_latex=r"$\mathrm{rms}(\epsilon_\phi)$", description="root-mean-square of the predicted noise"),
             C("identityResidual", "scalar", ["step"], symbol="||x_i - (alphaBar x_phi + sigma eps_phi)||", symbol_latex=r"$\|x_i - (\bar{\alpha}_i x_\phi + \sigma_i \epsilon_\phi)\|$",
               description="x_i = alphaBar_i x_phi + sigma_i eps_phi holds exactly before clipping; a non-zero value "
                           "measures how much the clipping of x_phi changed it"),
             C("result", "image", [], S, display_range=R, symbol="x_0", symbol_latex=r"$x_0$",
               description="the final sample (the scheduler's last prev_sample)")]
        v = [ViewInfo("occlusion", "image", ["step"], [self.H, self.W, 1], display_range=[0, 1],
                      symbol="occlusion(eps_phi)", symbol_latex=r"$\mathrm{occlusion}(\epsilon_\phi)$", compute=self._occlusion,
                      how_made="Occlusion (Zeiler & Fergus): a Size x Size patch of x_i (moved by Stride) is replaced "
                               "by Fill; each pixel gets the average rms change of eps_phi(x_i, i) over the patches "
                               "covering it; the map is divided by its maximum (1 = the most influential region). "
                               "Forward passes only; no gradients.",
                      reference="Zeiler & Fergus 2014; Szeliski 2nd ed. Sec. 5.4 -- applied here to a diffusion denoiser",
                      in_texts=False)]
        return c, v

    def describe(self):
        m = {"name": self.model_name, "library": "diffusers", "scheduler": type(self.sched).__name__,
             "steps": len(self.timesteps), "L": self.L, "imageShape": [self.H, self.W, self.C],
             "dataRange": self.data_range}
        cfg = getattr(self.sched, "config", None)
        for k in ("beta_schedule", "beta_start", "beta_end", "clip_sample", "prediction_type", "variance_type"):
            if cfg is not None and k in cfg:
                m[k] = cfg[k]
        return {"model": m, "references": list(self.references), "layout": "diffusion",
                "notation": {"alphaBar": "sqrt(alphas_cumprod[t]) = alphaBar_i", "beta2": "betas[t] = beta_i^2",
                             "step": "i = t + 1 (diffusers timestep t is 0-based)"}}

    def run(self, rec):
        gen = torch.Generator(self.device).manual_seed(self.seed)
        x = torch.randn((1, self.C, self.H, self.W), generator=gen, device=self.device)
        ac = self.sched.alphas_cumprod.to(torch.float64).cpu()
        for t in self.sched.timesteps:
            with torch.no_grad():
                eps = self.unet(x, t).sample
            res = self.sched.step(eps, t, x, generator=gen)
            x0 = res.pred_original_sample
            ti = int(t)
            prev = self._previous(ti)
            ab = float(ac[ti]); abp = float(ac[prev]) if prev >= 0 else 1.0
            beta, alpha = 1.0 - ab / abp, ab / abp                     # current_beta_t, current_alpha_t (diffusers)
            xi, e, xp = x, eps, x0
            state = {
                "x": lambda xi=xi: _hwc(xi), "eps": lambda e=e: _hwc(e), "x0hat": lambda xp=xp: _hwc(xp),
                "mu": lambda xi=xi, xp=xp: _hwc((math.sqrt(abp) * beta / (1 - ab)) * xp.double()
                                                 + (math.sqrt(alpha) * (1 - abp) / (1 - ab)) * xi.double()),
                "tLevel": lambda: (ti + 1) / self.L, "tNext": lambda: (prev + 1) / self.L,
                "alphaBar": lambda: math.sqrt(ab), "alphaBarNext": lambda: math.sqrt(abp),
                "sigma": lambda: math.sqrt(1 - ab),
                "varNext": lambda: max((1 - abp) / (1 - ab) * beta, 1e-20) if prev >= 0 else 0.0,
                "epsRms": lambda e=e: float(e.double().pow(2).mean().sqrt()),
                "identityResidual": lambda xi=xi, e=e, xp=xp: float(
                    (xi.double() - (math.sqrt(ab) * xp.double() + math.sqrt(1 - ab) * e.double())).norm()),
                "_x": lambda xi=xi: xi, "_t": lambda: t,
            }
            rec.point({"step": ti + 1}, state)
            x = res.prev_sample
        final = x
        rec.point({}, {"result": lambda: _hwc(final)})
        return {"x": final.detach().cpu()}

    # ---- helpers
    def _previous(self, t: int) -> int:
        if hasattr(self.sched, "previous_timestep"):
            p = self.sched.previous_timestep(t)
            return int(p)
        k = self.timesteps.index(t)
        return self.timesteps[k + 1] if k + 1 < len(self.timesteps) else -1

    def _occlusion(self, state, params, rng):
        size = int(params.get("size", 4)); stride = int(params.get("stride", 2))
        fill = float(params.get("fill", sum(self.data_range) / 2))
        x, t = state["_x"](), state["_t"]()
        with torch.no_grad():
            e0 = self.unet(x, t).sample
            acc = np.zeros((self.H, self.W)); cnt = np.zeros((self.H, self.W))
            rows = sorted(set(list(range(0, max(1, self.H - size + 1), stride)) + [max(0, self.H - size)]))
            cols = sorted(set(list(range(0, max(1, self.W - size + 1), stride)) + [max(0, self.W - size)]))
            for r in rows:
                for c in cols:
                    xo = x.clone()
                    xo[:, :, r:r + size, c:c + size] = fill
                    d = float((self.unet(xo, t).sample - e0).double().pow(2).mean().sqrt())
                    acc[r:r + size, c:c + size] += d; cnt[r:r + size, c:c + size] += 1
        m = acc / np.maximum(cnt, 1)
        return (m / max(m.max(), np.finfo(float).eps))[:, :, None]
