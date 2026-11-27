# Likelihood Ratio Test Guide

## Overview

The Likelihood Ratio Test (LRT) is a fundamental hypothesis testing framework in statistical inference and time series analysis (Hamilton, *Time Series Analysis*, Chapter 5, §5.8). It tests $m$ linear constraints on a parameter vector $\theta$ by comparing the log-likelihood of an unrestricted model $\hat{\theta}$ to a restricted model $\tilde{\theta}$.

The test statistic $\lambda_{LRT}$ is computed as:

$$\lambda_{LRT} = 2 \left( \mathcal{L}(\hat{\theta}) - \mathcal{L}(\tilde{\theta}) \right)$$

Under the null hypothesis $H_0$, $\lambda_{LRT}$ asymptotically follows a Chi-Square distribution with $m$ degrees of freedom ($\chi^2(m)$):

$$\lambda_{LRT} \sim \chi^2(m)$$

## Strategy Architecture

The Ocean Library implements LRT via a decoupled dual-representation strategy pattern:

1. **`OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest`**: Core hypothesis test strategy inheriting from `IStatisticalTest`.
2. **`OceanPipeline.Transforms.LikelihoodRatioTestOperator`**: Decoupled pipeline operator transform with dual representation (`forward` for numeric computation, `expr_fn` for KaTeX visualization).

## Usage

```matlab
% 1. Instantiate the test strategy
lrt = OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest();

% 2. Run test with unrestricted log-likelihood, restricted log-likelihood, and degrees of freedom m
logL_unrestricted = 0.0;
logL_restricted = -0.857;
df = 1;

result = lrt.test(logL_unrestricted, logL_restricted, df, 'Alpha', 0.05);

% 3. Inspect immutable TestResult container
disp(result.Statistic);   % 1.714
disp(result.PValue);      % 0.19047
disp(result.Reject);      % false
```

## Examples

### Hamilton Example 5.8.5

In Hamilton Example 5.8.5, an unrestricted AR model has $\mathcal{L}(\hat{\theta}) = 0.0$, while a restricted model under $H_0$ (1 linear constraint, $m = 1$) has $\mathcal{L}(\tilde{\theta}) = -0.857$.

```matlab
lrt = OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest();
res = lrt.test(0.0, -0.857, 1);

assert(abs(res.Statistic - 1.714) < 1e-3);
assert(~res.Reject); % Fail to reject H0 at alpha = 0.05
```

### Pipeline Operator Integration

```matlab
op = OceanPipeline.Transforms.LikelihoodRatioTestOperator('DegreesOfFreedom', 1, 'Alpha', 0.05);

data = struct('L_unrestricted', 150.2, 'L_restricted', 142.1);
out = op.apply(data);

disp(out.TestResult.Statistic); % 16.2
disp(out.TestResult.Reject);    % true
```

## See Also

- `OceanStats.Inference.HypothesisTest.Base.IStatisticalTest`
- `OceanStats.Inference.HypothesisTest.Base.TestResult`
- `OceanPipeline.Transforms.LikelihoodRatioTestOperator`
- `OceanStats.Time.ARMALikelihood`
