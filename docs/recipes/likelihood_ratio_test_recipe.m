%% Likelihood Ratio Test (LRT) Copy-Paste Recipes
% Demonstrates statistical hypothesis testing using LikelihoodRatioTest
% and LikelihoodRatioTestOperator.

%% Recipe 1: Standard Likelihood Ratio Test (Hamilton Ex 5.8.5)
import OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest;

lrt = LikelihoodRatioTest();

% Log-likelihoods from unrestricted model (theta_hat) and restricted model (theta_tilde)
L_unrestricted = 0.0;
L_restricted = -0.857;
m = 1; % 1 constraint

result = lrt.test(L_unrestricted, L_restricted, m, 'Alpha', 0.05);

fprintf('LRT Statistic lambda : %.4f\n', result.Statistic);
fprintf('p-value             : %.4f\n', result.PValue);
fprintf('Reject Null (H0)    : %d\n', result.Reject);

%% Recipe 2: Significant Model Comparison (Rejecting H0)
L_unrestricted_2 = 254.5;
L_restricted_2 = 241.0; % Delta 2*13.5 = 27.0
m_2 = 2; % 2 constraints

result_2 = lrt.test(L_unrestricted_2, L_restricted_2, m_2);

fprintf('\n--- Test 2 ---\n');
fprintf('LRT Statistic lambda : %.4f\n', result_2.Statistic);
fprintf('p-value             : %.4e\n', result_2.PValue);
fprintf('Reject Null (H0)    : %d\n', result_2.Reject);

%% Recipe 3: Pipeline Operator Integration
import OceanPipeline.Transforms.LikelihoodRatioTestOperator;

op = LikelihoodRatioTestOperator('DegreesOfFreedom', 2, 'Alpha', 0.05);
dataStruct = struct('L_unrestricted', 254.5, 'L_restricted', 241.0);

out = op.apply(dataStruct);

fprintf('\n--- Pipeline Operator Result ---\n');
fprintf('LRT Statistic: %.4f, Reject: %d\n', ...
    out.TestResult.Statistic, out.TestResult.Reject);
