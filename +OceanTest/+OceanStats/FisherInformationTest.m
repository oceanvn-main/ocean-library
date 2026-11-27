classdef FisherInformationTest < matlab.unittest.TestCase
    % FISHERINFORMATIONTEST Unit tests for FisherInformationMatrix
    %
    % Verifies Score V computation, Expected Fisher Information J(theta),
    % and empirical Observed Fisher Information I_n(theta_hat).

    methods (Test)
        function testGaussianExpectedInformation(testCase)
            N = 100;
            sigma2 = 2.0;
            params = struct('mu', 0, 'sigma2', sigma2);
            res = OceanMath.Information.FisherInformationMatrix.computeExpected('gaussian', params, N);

            testCase.assertTrue(res.IsAnalytic);
            testCase.assertEqual(res.ExpectedInformation(1, 1), N / sigma2, 'AbsTol', 1e-6);
            testCase.assertEqual(res.ExpectedInformation(2, 2), N / (2 * (sigma2^2)), 'AbsTol', 1e-6);
        end

        function testObservedFisherInformation(testCase)
            rng(42);
            N = 200;
            trueMu = 1.5;
            trueSigma2 = 1.0;
            data = trueMu + sqrt(trueSigma2) * randn(N, 1);

            % Gaussian Log-Likelihood Function: ln f(x; mu, sigma2)
            loglikFn = @(x, th) -0.5 * log(2 * pi * th(2)) - 0.5 * ((x - th(1)).^2) / th(2);

            thetaHat = [mean(data), var(data, 1)];
            res = OceanMath.Information.FisherInformationMatrix.computeObserved(loglikFn, data, thetaHat);

            % Observed Fisher Information J_mu = N / sigma2 = 200 / 1 = 200
            expectedI_mu = N / thetaHat(2);
            testCase.assertEqual(res.ObservedInformation(1, 1), expectedI_mu, 'RelTol', 0.05);

            % Standard errors should be positive
            testCase.assertTrue(all(res.StandardErrors > 0));
            testCase.assertFalse(res.IsAnalytic);
            testCase.assertTrue(contains(res.Metadata, "analytical"));
        end

        function testScoreMeanConvergence(testCase)
            rng(123);
            N = 500;
            trueLambda = 3.0;
            data = exprnd(1 / trueLambda, N, 1);

            % Exponential log-likelihood: ln f(x; lambda) = ln(lambda) - lambda * x
            loglikFn = @(x, th) log(th(1)) - th(1) * x;

            scoreRes = OceanMath.Information.FisherInformationMatrix.computeScore(loglikFn, data, trueLambda);

            % Mean score E[V] should be close to 0
            testCase.assertEqual(scoreRes.MeanScore, 0, 'AbsTol', 0.1);
        end
    end
end
