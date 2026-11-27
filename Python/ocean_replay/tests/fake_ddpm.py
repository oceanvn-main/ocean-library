"""A tiny network and a DDPM scheduler that reproduces diffusers.DDPMScheduler.step
(epsilon prediction, linear betas, clip_sample, variance "fixed_small") -- so the
adapter can be tested locally without diffusers. test_diffusers_cloud.py runs the
same checks against the real diffusers classes (in Cloud Shell / on the VM).
"""
from types import SimpleNamespace

import torch


class Cfg(dict):
    __getattr__ = dict.__getitem__


class FakeDDPMScheduler:
    def __init__(self, L=50, beta_start=1e-4, beta_end=0.02, clip_sample=True):
        self.config = Cfg(num_train_timesteps=L, beta_start=beta_start, beta_end=beta_end, beta_schedule="linear",
                          clip_sample=clip_sample, prediction_type="epsilon", variance_type="fixed_small")
        self.betas = torch.linspace(beta_start, beta_end, L, dtype=torch.float32)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)
        self.timesteps = torch.arange(L - 1, -1, -1)
        self.means, self.vars = {}, {}                  # what step() used, per timestep (for tests)

    def set_timesteps(self, n):
        ratio = self.config.num_train_timesteps // n
        self.timesteps = torch.arange(0, n).mul(ratio).flip(0)
        self.ratio = ratio

    def previous_timestep(self, t):
        return int(t) - getattr(self, "ratio", 1)

    def step(self, eps, t, x, generator=None):
        t = int(t); prev = self.previous_timestep(t)
        ab = self.alphas_cumprod[t]; abp = self.alphas_cumprod[prev] if prev >= 0 else torch.tensor(1.0)
        beta = 1 - ab / abp; alpha = ab / abp
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        if self.config.clip_sample:
            x0 = x0.clamp(-1, 1)
        mean = (abp.sqrt() * beta / (1 - ab)) * x0 + (alpha.sqrt() * (1 - abp) / (1 - ab)) * x
        prev_sample = mean
        var = torch.tensor(0.0)
        if t > 0:
            var = ((1 - abp) / (1 - ab) * beta).clamp(min=1e-20)
            z = torch.randn(eps.shape, generator=generator, device=eps.device, dtype=eps.dtype)
            prev_sample = mean + var.sqrt() * z
        self.means[t], self.vars[t] = mean.clone(), float(var)
        return SimpleNamespace(prev_sample=prev_sample, pred_original_sample=x0)


class TinyUNet(torch.nn.Module):
    """eps_phi(x, t): one 3x3 convolution + a time-dependent scale (deterministic weights)."""

    def __init__(self, channels=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(0.2 * torch.randn(self.conv.weight.shape, generator=g))
            self.conv.bias.zero_()

    def forward(self, x, t):
        return SimpleNamespace(sample=torch.tanh(self.conv(x)) * (1.0 + 0.001 * float(t)))
