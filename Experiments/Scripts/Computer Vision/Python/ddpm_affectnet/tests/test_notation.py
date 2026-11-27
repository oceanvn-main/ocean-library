"""diffusers' DDPM == the book's equations (Lai et al. Sec. 2.2). Runs where diffusers is installed (the VM).

diffusers keeps the schedule in float32: alphas_cumprod (= alphaBar_i^2) is a float32
running product of 1000 factors, and the reverse step forms e.g. 1 - alphaBar_1^2 =
1 - 0.9998 in float32 (a cancellation that costs ~3 digits at the last step). These
tests therefore build every book quantity FROM THE SCHEDULER'S OWN float32 values, in
the same arithmetic, and check that the formulas agree tightly -- what is tested is
that diffusers implements Eq. (2.2.1) and Eq. (2.2.4)-(2.2.5), not float32 rounding.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers import DDPMScheduler  # noqa: E402


def _sched():
    return DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=0.02,
                         prediction_type="epsilon", clip_sample=False)


def test_schedule_names_and_forward_process():
    s = _sched()
    # names: betas = beta_i^2 and alphas_cumprod = alphaBar_i^2 = prod_k (1 - beta_k^2)
    exact = torch.cumprod(1 - s.betas.double(), 0)
    assert torch.allclose(s.alphas_cumprod.double(), exact, rtol=1e-4, atol=1e-7)   # float32 running product
    alphaBar = torch.sqrt(s.alphas_cumprod.double())                                  # book alphaBar_i
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(2, 3, 8, 8, generator=g).double()
    eps = torch.randn(2, 3, 8, 8, generator=g).double()
    for t in (0, 499, 999):
        xi = s.add_noise(x0, eps, torch.tensor([t, t]))
        book = alphaBar[t] * x0 + torch.sqrt(1 - alphaBar[t] ** 2) * eps              # Eq. (2.2.1)
        assert torch.allclose(xi, book, rtol=1e-9, atol=1e-12)


def test_reverse_step_is_eq_224_225():
    s = _sched(); s.set_timesteps(1000)
    ac = s.alphas_cumprod                                  # float32, alphaBar^2
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 3, 8, 8, generator=g).double()
    eps_hat = torch.randn(1, 3, 8, 8, generator=g).double()
    for t in (999, 500, 1):
        res = s.step(eps_hat, t, x, generator=torch.Generator().manual_seed(7))
        abSq_i, abSq_im1 = ac[t], ac[t - 1]                # alphaBar_i^2, alphaBar_{i-1}^2      (float32)
        one_m_i, one_m_im1 = 1 - abSq_i, 1 - abSq_im1      # 1 - alphaBar_i^2, 1 - alphaBar_{i-1}^2
        alphaSq_i = abSq_i / abSq_im1                      # alpha_i^2
        betaSq_i = 1 - alphaSq_i                           # beta_i^2
        x_phi = (x - one_m_i ** 0.5 * eps_hat) / abSq_i ** 0.5                       # x_phi(x_i, i)
        c_x0 = (abSq_im1 ** 0.5 * betaSq_i) / one_m_i      # alphaBar_{i-1} beta_i^2 / (1 - alphaBar_i^2)
        c_xi = alphaSq_i ** 0.5 * one_m_im1 / one_m_i      # alpha_i (1 - alphaBar_{i-1}^2) / (1 - alphaBar_i^2)
        mu = c_x0 * x_phi + c_xi * x                                                   # Eq. (2.2.4)
        tildeBetaSq = one_m_im1 / one_m_i * betaSq_i       # sigma^2(i) = tildeBeta_i^2
        z = torch.randn(x.shape, generator=torch.Generator().manual_seed(7), dtype=x.dtype)
        assert torch.allclose(res.pred_original_sample, x_phi, rtol=1e-9, atol=1e-12)
        assert torch.allclose(res.prev_sample, mu + tildeBetaSq ** 0.5 * z, rtol=1e-9, atol=1e-12)   # Eq. (2.2.5)
        # and in exact (double) arithmetic the same formulas agree to float32 accuracy
        ab = torch.sqrt(ac.double())
        a_i, a_im1 = ab[t], ab[t - 1]
        mu64 = (a_im1 * (1 - (a_i / a_im1) ** 2) / (1 - a_i ** 2)) * x_phi + ((1 - a_im1 ** 2) * (a_i / a_im1) / (1 - a_i ** 2)) * x
        assert torch.allclose(mu, mu64, rtol=1e-2, atol=1e-3)
