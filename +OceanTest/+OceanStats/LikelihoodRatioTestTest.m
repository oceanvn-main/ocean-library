classdef LikelihoodRatioTestTest < matlab.unittest.TestCase
    % LIKELIHOODRATIOTESTTEST  Unit tests for Likelihood Ratio Test and Pipeline Operator.
    %
    % NAMESPACE: OceanTest.OceanStats
    %
    % Author: Ocean Library System
    % Date: 2026-09-09

    methods (Test)
        % -------------------------------------------------------------- %
        function testHamiltonExample585(testCase)
            % Hamilton (1994) Chapter 5, Example 5.8.5 textbook test case
            % L(theta_hat) = 0.0, L(theta_tilde) = -0.857, m = 1
            % Expected: stat = 1.714, p-value ~ 0.1905, Reject = false

            engine = OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest();
            data = struct( ...
                'LogLikUnrestricted', 0.0, ...
                'LogLikRestricted', -0.857, ...
                'Restrictions', 1, ...
                'Alpha', 0.05 ...
            );

            result = engine.test(data);

            testCase.verifyEqual(result.Statistic, 1.714, 'AbsTol', 1e-3);
            testCase.verifyEqual(result.Df, 1);
            testCase.verifyTrue(result.PValue > 0.18 && result.PValue < 0.20);
            testCase.verifyFalse(result.Reject);
            testCase.verifyEqual(result.TestName, "Likelihood Ratio Test (LRT)");
        end

        % -------------------------------------------------------------- %
        function testRejectNullHypothesis(testCase)
            % Test case where restricted model significantly degrades log-likelihood
            % L_unrestricted = 120.5, L_restricted = 112.0, m = 2
            % stat = 2 * (120.5 - 112.0) = 17.0
            % p-value for Chi2(2) at 17.0 is ~0.000212 (Reject H0)

            engine = OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest();
            data = struct( ...
                'L_unrestricted', 120.5, ...
                'L_restricted', 112.0, ...
                'm', 2, ...
                'Alpha', 0.05 ...
            );

            result = engine.test(data);

            testCase.verifyEqual(result.Statistic, 17.0, 'AbsTol', 1e-4);
            testCase.verifyEqual(result.Df, 2);
            testCase.verifyTrue(result.PValue < 0.001);
            testCase.verifyTrue(result.Reject);
        end

        % -------------------------------------------------------------- %
        function testInvalidRestrictionsThrows(testCase)
            % Test invalid degrees of freedom (m < 1)
            engine = OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest();
            data = struct('LogLikUnrestricted', 10.0, 'LogLikRestricted', 5.0, 'Restrictions', 0);

            testCase.verifyError(@() engine.test(data), 'LikelihoodRatioTest:InvalidDf');
        end

        % -------------------------------------------------------------- %
        function testMissingFieldsThrows(testCase)
            % Test missing required log-likelihood field
            engine = OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest();
            data = struct('LogLikUnrestricted', 10.0, 'Restrictions', 1);

            testCase.verifyError(@() engine.test(data), 'LikelihoodRatioTest:MissingField');
        end

        % -------------------------------------------------------------- %
        function testPipelineOperatorNumericAndSymbolic(testCase)
            % Test LikelihoodRatioTestOperator prototype, fit, apply, and expr_fn
            op = OceanPipeline.Transforms.LikelihoodRatioTestOperator('IsSymbolic', true, 'Alpha', 0.05);
            proto = op.getPrototype();

            testCase.verifyEqual(proto.id, "op_tsa_lrt");
            testCase.verifyTrue(proto.is_symbolic);
            testCase.verifyEqual(numel(proto.inputs), 3);
            testCase.verifyEqual(numel(proto.outputs), 3);

            inputMatrix = [120.5, 112.0, 2];
            state = op.fit(inputMatrix);
            out = op.apply(inputMatrix, state);

            testCase.verifyEqual(out(1), 17.0, 'AbsTol', 1e-4);
            testCase.verifyTrue(out(2) < 0.001);
            testCase.verifyEqual(out(3), 1.0);

            expr = op.expr_fn(state);
            testCase.verifyTrue(expr.is_symbolic);
            testCase.verifyTrue(strlength(expr.symbolic) > 0);
        end

        % -------------------------------------------------------------- %
        function testPipelineOperatorHonestOpacity(testCase)
            % Test R7 Rule: IsSymbolic = false returns non-symbolic tag
            op = OceanPipeline.Transforms.LikelihoodRatioTestOperator('IsSymbolic', false);
            inputMatrix = [0.0, -0.857, 1];
            state = op.fit(inputMatrix);
            expr = op.expr_fn(state);

            testCase.verifyFalse(expr.is_symbolic);
            testCase.verifyEqual(expr.symbolic, "");
        end
    end
end
