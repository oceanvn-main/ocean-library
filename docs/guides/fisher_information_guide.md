# Fisher Information Matrix & Score Function V Guide

## Overview

The **Fisher Information Matrix** measures the amount of information that an observable random variable $X$ carries about an unknown parameter $\theta$ upon which the probability distribution $f(X; \theta)$ depends.

It plays a central role in:
1. **Asymptotic Variance & Minimum Variance Unbiased Estimators (MVUE)**: The Cramér-Rao Lower Bound (CRLB) states that $\text{Var}(\hat{\theta}) \ge J(\theta)^{-1}$.
2. **Asymptotic Covariance of Maximum Likelihood Estimators (MLE)**: Under regularity conditions, $\sqrt{N}(\hat{\theta}_{MLE} - \theta_0) \xrightarrow{d} \mathcal{N}\left(0, \mathcal{I}_n(\hat{\theta})^{-1}\right)$.
3. **Statistical Hypothesis Testing**: Forms the denominator for Wald tests $W = (\hat{\theta} - \theta_0)^T \mathcal{I}_n(\hat{\theta}) (\hat{\theta} - \theta_0)$.

---

## Mathematical Foundations

### 1. The Score Function $V$

The **Score $V$** is the gradient of the log-likelihood function with respect to parameter $\theta$:

$$V(X; \theta) = \frac{\partial}{\partial \theta} \ln f(X; \theta) = \frac{\frac{\partial}{\partial \theta} f(X; \theta)}{f(X; \theta)}$$

Under standard regularity conditions, the mean value of the Score vector is zero:

$$\mathbb{E}_\theta[V] = \int \frac{\frac{\partial}{\partial \theta} f(x; \theta)}{f(x; \theta)} f(x; \theta) \, dx = \frac{\partial}{\partial \theta} \int f(x; \theta) \, dx = \frac{\partial}{\partial \theta} (1) = 0$$

Because $\mathbb{E}[V] = 0$, the variance of the Score is equal to its second raw moment:

$$\text{Var}(V) = \mathbb{E}_\theta[V^2]$$

---

### 2. Theoretical Expected Fisher Information $J(\theta)$

The **Expected Fisher Information** $J(\theta)$ is defined as the variance of the Score function (Cover & Thomas, *Elements of Information Theory*, 2nd Ed., Section 11.10, Eq. 11.270):

$$J(\theta) = E_\theta \left[ \left( \frac{\partial}{\partial \theta} \ln f(X; \theta) \right)^2 \right] = -E_\theta \left[ \frac{\partial^2}{\partial \theta^2} \ln f(X; \theta) \right]$$

For an $i.i.d.$ sample $X_1, X_2, \dots, X_N \sim f(x; \theta)$, the total score is $V_n(\theta) = \sum_{i=1}^N V(X_i; \theta)$, and the total expected information is:

$$J_N(\theta) = N J(\theta)$$

#### Closed-Form Examples:
- **Gaussian $\mathcal{N}(\mu, \sigma^2)$**:
  $$J(\mu) = \frac{N}{\sigma^2}, \quad J(\sigma^2) = \frac{N}{2 \sigma^4}$$
- **Poisson $\text{Poisson}(\lambda)$**:
  $$J(\lambda) = \frac{N}{\lambda}$$
- **Exponential $\text{Exp}(\lambda)$**:
  $$J(\lambda) = \frac{N}{\lambda^2}$$

---

### 3. Empirical Observed Fisher Information $\mathcal{I}_n(\hat{\theta})$

For general complex distributions (such as multi-parameter ARMA, GARCH, deep probabilistic models, or non-Gaussian likelihoods), the expectation integral $\mathbb{E}_\theta [ V^2 ]$ **cannot be solved analytically in closed form**.

In practice, researchers compute the **Observed Fisher Information Matrix** evaluated at the maximum likelihood estimate $\hat{\theta}$:

$$\mathcal{I}_n(\hat{\theta}) = -\sum_{i=1}^N \left. \nabla_\theta^2 \ln f(x_i; \theta) \right|_{\theta = \hat{\theta}} = -H(\hat{\theta})$$

Alternatively, the **Outer Product of Gradients (OPG)** estimator uses empirical scores:

$$\mathcal{I}_{OPG}(\hat{\theta}) = \sum_{i=1}^N V(x_i; \hat{\theta}) V(x_i; \hat{\theta})^T$$

The asymptotic standard errors of the estimates $\hat{\theta}$ are extracted from the diagonal of the inverse Observed Fisher Information matrix:

$$\text{SE}(\hat{\theta}_j) = \sqrt{\left[ \mathcal{I}_n(\hat{\theta})^{-1} \right]_{jj}}$$

---

## Usage in MATLAB

```matlab
% 1. Compute Analytical Expected Fisher Information
N = 100;
distParams = struct('mu', 0, 'sigma2', 1.5);
resExpected = OceanMath.Information.FisherInformationMatrix.computeExpected('gaussian', distParams, N);
disp(resExpected.ExpectedInformation);

% 2. Compute Empirical Observed Fisher Information
data = randn(200, 1);
loglikFn = @(x, th) -0.5*log(2*pi*th(2)) - 0.5*((x - th(1)).^2)/th(2);
thetaHat = [mean(data), var(data, 1)];

resObserved = OceanMath.Information.FisherInformationMatrix.computeObserved(loglikFn, data, thetaHat);
disp("Observed Fisher Info:");
disp(resObserved.ObservedInformation);
disp("Standard Errors:");
disp(resObserved.StandardErrors);
```

---

## See Also
- [LikelihoodRatioTest Guide](likelihood_ratio_test_guide.md)
- `OceanMath.Information.FisherInformationMatrix`
- `OceanMath.Information.InformationEngine`
