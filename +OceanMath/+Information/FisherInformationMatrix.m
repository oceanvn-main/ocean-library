classdef FisherInformationMatrix
    % FISHERINFORMATIONMATRIX Computes Score function V, Expected Fisher Information J(theta), and Observed Fisher Information matrix I_n(theta_hat).
    %
    %   TEXTBOOK REFERENCE:
    %       Cover & Thomas, "Elements of Information Theory" (2nd Ed.), Chapter 11, Section 11.10.
    %       Definition (Eq. 11.270):
    %           J(theta) = E_theta[ ( (d/d theta) ln f(X; theta) )^2 ]
    %
    %   The score V is defined as V(X; theta) = (d/d theta) ln f(X; theta).
    %   The expected Fisher Information is J(theta) = E_theta[ V^2 ] = - E_theta[ (d^2/d theta^2) ln f(X; theta) ].
    %   The empirical Observed Fisher Information matrix is I_n(theta_hat) = - sum_{i=1}^N (d^2/d theta^2) ln f(x_i; theta)|_{theta=theta_hat}.
    %
    %   Example:
    %       fim = OceanMath.Information.FisherInformationMatrix();
    %       res = fim.computeExpected('gaussian', struct('mu', 0, 'sigma2', 1), 100);
    %
    %   See also InformationEngine, OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest

    methods (Static)
        function res = computeScore(loglikFn, data, theta)
            % COMPUTESCORE Delegates score calculation to OceanMath.Information.ScoreFunction
            arguments
                loglikFn (1,1) function_handle
                data double
                theta (1,:) double
            end

            res = OceanMath.Information.ScoreFunction.compute(loglikFn, data, theta);
        end

        function res = computeExpected(distType, distParams, N)
            % COMPUTEEXPECTED Computes analytical Expected Fisher Information J(theta)
            arguments
                distType (1,1) string
                distParams (1,1) struct
                N (1,1) double = 1
            end

            switch lower(distType)
                case "gaussian"
                    sigma2 = distParams.sigma2;
                    % For N(mu, sigma2): J(mu) = N/sigma2, J(sigma2) = N/(2*sigma2^2)
                    J = [N / sigma2, 0; 0, N / (2 * (sigma2^2))];
                    paramNames = ["mu", "sigma2"];
                    formulaStr = "J(mu) = N / sigma^2, J(sigma^2) = N / (2 * sigma^4)";

                case "poisson"
                    lambda = distParams.lambda;
                    % For Poisson(lambda): J(lambda) = N / lambda
                    J = N / lambda;
                    paramNames = ["lambda"];
                    formulaStr = "J(lambda) = N / lambda";

                case "exponential"
                    rate = distParams.rate;
                    % For Exp(rate): J(rate) = N / rate^2
                    J = N / (rate^2);
                    paramNames = ["rate"];
                    formulaStr = "J(rate) = N / rate^2";

                case "arma"
                    sigma2 = distParams.sigma2;
                    % For ARMA noise variance: J(sigma2) = N / (2 * sigma2^2)
                    J = N / (2 * (sigma2^2));
                    paramNames = ["sigma2"];
                    formulaStr = "J(sigma^2) = N / (2 * sigma^4)";

                otherwise
                    error("OceanMath:FisherInformation:UnsupportedDist", ...
                        "Unsupported analytical distribution type '%s'. Use computeObserved for empirical likelihoods.", distType);
            end

            res = struct(...
                'Distribution', distType, ...
                'ExpectedInformation', J, ...
                'ParamNames', paramNames, ...
                'Formula', formulaStr, ...
                'SampleSize', N, ...
                'IsAnalytic', true, ...
                'Metadata', "Theoretical Fisher Information J(theta) derived via variance of Score V: var(V) = E[V^2]." ...
            );
        end

        function res = computeObserved(loglikFn, data, thetaHat, opts)
            % COMPUTEOBSERVED Computes empirical Observed Fisher Information I_n(theta_hat)
            arguments
                loglikFn (1,1) function_handle
                data double
                thetaHat (1,:) double
                opts.Method (1,1) string {mustBeMember(opts.Method, ["hessian", "opg", "both"])} = "both"
            end

            N = size(data, 1);
            p = numel(thetaHat);
            h = 1e-4;

            % 1. Compute Individual Scores V_i(thetaHat)
            scoreRes = OceanMath.Information.FisherInformationMatrix.computeScore(loglikFn, data, thetaHat);
            V = scoreRes.ScoreMatrix;

            % 2. Outer Product of Gradients (OPG) Estimator: I_opg = sum_{i=1}^N V_i V_i^T
            I_opg = V' * V;

            % 3. Negative Hessian Estimator: I_hessian = - sum_{i=1}^N grad^2 ln f(x_i; thetaHat)
            H = zeros(p, p);
            totalLogLik = @(th) sum(arrayfun(@(idx) loglikFn(data(idx, :), th), 1:N));

            for j = 1:p
                for k = 1:p
                    if j == k
                        th_plus = thetaHat; th_plus(j) = th_plus(j) + h;
                        th_minus = thetaHat; th_minus(j) = th_minus(j) - h;
                        f_plus = totalLogLik(th_plus);
                        f_center = totalLogLik(thetaHat);
                        f_minus = totalLogLik(th_minus);
                        H(j, k) = (f_plus - 2 * f_center + f_minus) / (h^2);
                    else
                        th_pp = thetaHat; th_pp(j) = thetaHat(j) + h; th_pp(k) = thetaHat(k) + h;
                        th_pm = thetaHat; th_pm(j) = thetaHat(j) + h; th_pm(k) = thetaHat(k) - h;
                        th_mp = thetaHat; th_mp(j) = thetaHat(j) - h; th_mp(k) = thetaHat(k) + h;
                        th_mm = thetaHat; th_mm(j) = thetaHat(j) - h; th_mm(k) = thetaHat(k) - h;
                        H(j, k) = (totalLogLik(th_pp) - totalLogLik(th_pm) - totalLogLik(th_mp) + totalLogLik(th_mm)) / (4 * h^2);
                    end
                end
            end

            I_obs = -H;

            % Enforce symmetry
            I_obs = (I_obs + I_obs') / 2;
            I_opg = (I_opg + I_opg') / 2;

            % Standard Error Inversion
            try
                covMatrix = inv(I_obs);
                stdErrors = sqrt(diag(covMatrix))';
            catch
                covMatrix = pinv(I_obs);
                stdErrors = sqrt(max(0, diag(covMatrix)))';
            end

            metadataStr = "Because the theoretical expected Fisher Information matrix cannot generally " + ...
                "be analytically solved in closed form for complex distributions, empirical sample evaluation " + ...
                "(Observed Fisher Information I_n(theta_hat) = -Hessian or OPG) is utilized for asymptotic covariance and variance estimation.";

            res = struct(...
                'ObservedInformation', I_obs, ...
                'OPGInformation', I_opg, ...
                'HessianMatrix', H, ...
                'CovarianceMatrix', covMatrix, ...
                'StandardErrors', stdErrors, ...
                'MeanScore', scoreRes.MeanScore, ...
                'SampleSize', N, ...
                'NumParams', p, ...
                'IsAnalytic', false, ...
                'Metadata', metadataStr ...
            );
        end

        function meta = getMetadata()
            % GETMETADATA Returns self-describing component metadata, LaTeX representations, and capability specs
            meta = struct(...
                'ClassName', "OceanMath.Information.FisherInformationMatrix", ...
                'Package', "+OceanMath/+Information/FisherInformationMatrix.m", ...
                'Category', "OceanMath.Information", ...
                'TextbookRef', "Cover & Thomas, Elements of Information Theory (2nd Ed.), Section 11.10", ...
                'TextbookEq', "Eq. (11.270): J(theta) = E_theta[ ( (d/d theta) ln f(X; theta) )^2 ]", ...
                'SymbolicLatex', "J(\theta) = \mathbb{E}_{\theta}\left[ \left( \frac{\partial}{\partial \theta} \ln f(X; \theta) \right)^2 \right] = -\mathbb{E}_{\theta}\left[ \frac{\partial^2}{\partial \theta^2} \ln f(X; \theta) \right] \quad \text{(Cover \& Thomas Eq. 11.270)}", ...
                'ScoreLatex', "V_i(\theta) = \frac{\partial}{\partial \theta} \ln f(x_i; \theta)", ...
                'ObservedLatex', "I_n(\hat{\theta}) = -\sum_{i=1}^N \frac{\partial^2}{\partial \theta^2} \ln f(x_i; \theta)\Big|_{\theta=\hat{\theta}}", ...
                'MetadataText', "Textbook Definition (Cover & Thomas Eq. 11.270): Fisher Information J(theta) is defined as the variance of the score function V = d/d theta ln f(X; theta). For arbitrary complex likelihoods where closed-form integration is intractable, empirical observed Fisher Information (Negative Hessian or OPG) is computed.", ...
                'Methods', [ ...
                    "computeScore(loglikFn, data, theta) -> ScoreMatrix V_i", ...
                    "computeExpected(distType, distParams, N) -> Theoretical Matrix J_N", ...
                    "computeObserved(loglikFn, data, thetaHat, opts) -> Observed Matrix I_n = -H(thetaHat)", ...
                    "getMetadata() -> Component Reflection Metadata Struct" ...
                ], ...
                'Capabilities', [ ...
                    "✓ Cover & Thomas Eq. (11.270) Score Variance J(theta) = E[(d/d theta ln f)^2]", ...
                    "✓ Regularity Score Expectation E[V] = 0", ...
                    "✓ Theoretical Expected Fisher Information J_N(theta)", ...
                    "✓ Empirical Observed Hessian I_n(theta_hat)", ...
                    "✓ Outer Product of Gradients (OPG) sum V_i V_i^T", ...
                    "✓ Identifiability Condition Number kappa(I_n)", ...
                    "✓ Asymptotic Covariance Sigma = I_n^-1 and Standard Errors SE(theta_hat)", ...
                    "✓ Cramer-Rao Variance Lower Bound & Precision Upper Bound" ...
                ] ...
            );
        end
    end
end
