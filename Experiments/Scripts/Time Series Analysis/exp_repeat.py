import torch
import math

# ---------------------------------------------------------------------------
# 0.  Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)

T = 10000 #Seq length
B = 1000 #Batch size
theta = 0.5
sigma2 = 1.0
sigma = math.sqrt(sigma2)

#Non-invertible twin params
theta_tidle = 1.0 / theta #2.0
sigma2_tidle = sigma2 / theta_tidle**2
sigma_tidle = math.sqrt(sigma2_tidle)