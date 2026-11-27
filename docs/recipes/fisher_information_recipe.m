%% Fisher Information Matrix Copy-Paste Recipe
% Demonstrates Score V vector evaluation, Expected Fisher Information J(theta),
% and empirical Observed Fisher Information I_n(theta_hat) via Hessian and OPG.

clc; clear; close all;

%% 1. Generate Synthetic Data
rng(100);
N = 300;
trueMu = 2.5;
trueSigma2 = 1.44;
data = trueMu + sqrt(trueSigma2) * randn(N, 1);

%% 2. Standalone Score Function V(X; theta) & Regularity Check
loglikFn = @(x, th) -0.5 * log(2 * pi * th(2)) - 0.5 * ((x - th(1)).^2) / th(2);
thetaHat = [mean(data), var(data, 1)];

scoreRes = OceanMath.Information.ScoreFunction.compute(loglikFn, data, thetaHat);
fprintf('--- Standalone Score Function V (Cover & Thomas Eq. 11.268) ---\n');
fprintf('Mean Score E[V]: [%.4f, %.4f] (Passed Regularity E[V]≈0: %d)\n\n', scoreRes.MeanScore(1), scoreRes.MeanScore(2), scoreRes.PassedRegularity);

%% 3. Calculate Analytical Expected Fisher Information J(theta)
distParams = struct('mu', trueMu, 'sigma2', trueSigma2);
resExpected = OceanMath.Information.FisherInformationMatrix.computeExpected('gaussian', distParams, N);

fprintf('--- 1. Expected Fisher Information (Analytical) ---\n');
fprintf('Formula: %s\n', resExpected.Formula);
disp(resExpected.ExpectedInformation);

%% 4. Calculate Empirical Observed Fisher Information I_n(theta_hat)

resObserved = OceanMath.Information.FisherInformationMatrix.computeObserved(loglikFn, data, thetaHat);

fprintf('--- 2. Observed Fisher Information I_n(theta_hat) (Hessian) ---\n');
disp(resObserved.ObservedInformation);

fprintf('--- 3. Outer Product of Gradients (OPG) Estimator ---\n');
disp(resObserved.OPGInformation);

fprintf('--- 4. Asymptotic Standard Errors SE(theta_hat) ---\n');
fprintf('SE(mu_hat)    : %.4f (Theoretical: %.4f)\n', resObserved.StandardErrors(1), sqrt(trueSigma2 / N));
fprintf('SE(sigma2_hat): %.4f (Theoretical: %.4f)\n', resObserved.StandardErrors(2), sqrt(2 * trueSigma2^2 / N));

fprintf('\n--- Metadata Explanation ---\n');
fprintf('%s\n', resObserved.Metadata);
