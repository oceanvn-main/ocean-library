classdef LikelihoodRatioTest < OceanStats.Inference.HypothesisTest.Base.IStatisticalTest
    % LIKELIHOODRATIOTEST  Likelihood Ratio Test (LRT) for statistical hypothesis testing.
    %
    % NAMESPACE: OceanStats.Inference.HypothesisTest.Likelihood
    %
    % Strategy pattern: Inherits from IStatisticalTest.
    % Evaluates hypothesis restrictions on estimated parameter vectors theta.
    %
    % MATHEMATICAL DERIVATION:
    %   Under the null hypothesis H0 (which imposes m restrictions on theta),
    %   the test statistic lambda_LRT is defined as:
    %
    %     lambda_LRT = 2 * [ L(theta_hat) - L(theta_tilde) ]
    %
    %   where L(theta_hat)   = log-likelihood at unrestricted MLE
    %         L(theta_tilde) = log-likelihood at restricted MLE
    %         m              = number of parameter restrictions (degrees of freedom)
    %
    %   Asymptotically, lambda_LRT ~ Chi2(m) as N -> infinity.
    %
    % REQUIRED DATA FIELDS:
    %   data.LogLikUnrestricted (double) — Unrestricted log-likelihood L(theta_hat)
    %   data.LogLikRestricted   (double) — Restricted log-likelihood L(theta_tilde)
    %   data.Restrictions       (double) — Number of restrictions m (df >= 1)
    %   data.Alpha              (double) — Significance level (optional, default 0.05)
    %
    % USAGE:
    %   testEngine = OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest();
    %   data = struct('LogLikUnrestricted', 0.0, 'LogLikRestricted', -0.857, 'Restrictions', 1);
    %   result = testEngine.test(data);
    %   result.summary();
    %
    % Author: Ocean Library System
    % Date: 2026-09-09

    methods
        function result = test(obj, data)
            % TEST  Run Likelihood Ratio Test on provided log-likelihood data.
            %
            %   result = test(obj, data)
            %
            % Inputs:
            %   data - struct with required fields:
            %          LogLikUnrestricted (or L_unrestricted)
            %          LogLikRestricted (or L_restricted)
            %          Restrictions (or m)
            %          Alpha (optional, default 0.05)
            %
            % Outputs:
            %   result - TestResult value object
            
            arguments
                obj
                data (1,1) struct
            end

            % Extract Unrestricted Log-Likelihood
            if isfield(data, 'LogLikUnrestricted')
                lunrestricted = double(data.LogLikUnrestricted);
            elseif isfield(data, 'L_unrestricted')
                lunrestricted = double(data.L_unrestricted);
            else
                error('LikelihoodRatioTest:MissingField', ...
                    'Required field "LogLikUnrestricted" or "L_unrestricted" is missing.');
            end

            % Extract Restricted Log-Likelihood
            if isfield(data, 'LogLikRestricted')
                lrestricted = double(data.LogLikRestricted);
            elseif isfield(data, 'L_restricted')
                lrestricted = double(data.L_restricted);
            else
                error('LikelihoodRatioTest:MissingField', ...
                    'Required field "LogLikRestricted" or "L_restricted" is missing.');
            end

            % Extract Restrictions (Degrees of Freedom m)
            if isfield(data, 'Restrictions')
                m = double(data.Restrictions);
            elseif isfield(data, 'm')
                m = double(data.m);
            elseif isfield(data, 'Df')
                m = double(data.Df);
            else
                error('LikelihoodRatioTest:MissingField', ...
                    'Required field "Restrictions" or "m" is missing.');
            end

            if m < 1 || floor(m) ~= m
                error('LikelihoodRatioTest:InvalidDf', ...
                    'Degrees of freedom (m) must be a positive integer >= 1.');
            end

            % Extract Significance Level Alpha
            alpha = 0.05;
            if isfield(data, 'Alpha') && ~isempty(data.Alpha)
                alpha = double(data.Alpha);
            end

            % Compute Likelihood Ratio Test Statistic: 2 * [ L(theta_hat) - L(theta_tilde) ]
            stat = 2 * (lunrestricted - lrestricted);
            notes = "";

            % Monotonicity sanity check: L(unrestricted) >= L(restricted)
            if stat < 0
                notes = sprintf("Warning: Restricted likelihood exceeds unrestricted likelihood (stat = %.4f). Numerical optimization discrepancy suspected.", stat);
                stat = max(0, stat);
            end

            % Asymptotic Chi-Square p-value evaluation
            if exist('chi2cdf', 'file') == 2 || exist('chi2cdf', 'builtin') == 5
                pVal = chi2cdf(stat, m, 'upper');
            else
                % Fallback via incomplete gamma function if Statistics Toolbox is absent
                pVal = 1 - gammainc(stat / 2, m / 2, 'lower');
            end

            % Build immutable TestResult container
            result = OceanStats.Inference.HypothesisTest.Base.TestResult( ...
                'Statistic', stat, ...
                'PValue', pVal, ...
                'Df', m, ...
                'Alpha', alpha, ...
                'TestName', obj.name(), ...
                'Notes', notes ...
            );
        end

        function str = name(~)
            % NAME  Return human-readable name of hypothesis test.
            str = "Likelihood Ratio Test (LRT)";
        end

        function list = assumptions(~)
            % ASSUMPTIONS  Return theoretical assumptions required by LRT.
            list = { ...
                'H0 parameter space is nested strictly within H1 parameter space', ...
                'Likelihood functions satisfy standard regularity conditions for MLE consistency', ...
                'Test statistic 2*(L_unrestricted - L_restricted) is asymptotically Chi-Square distributed' ...
            };
        end
    end

    methods (Static)
        function meta = getMetadata()
            % GETMETADATA Returns self-describing component metadata, LaTeX representations, and capability specs
            meta = struct(...
                'ClassName', "OceanStats.Inference.HypothesisTest.Likelihood.LikelihoodRatioTest", ...
                'Package', "+OceanStats/+Inference/+HypothesisTest/+Likelihood/LikelihoodRatioTest.m", ...
                'Category', "OceanStats.Inference", ...
                'SymbolicLatex', "\Lambda = 2 \left[ \ln L(\hat{\theta}_{alt}) - \ln L(\hat{\theta}_{null}) \right] \sim \chi^2(df)", ...
                'MetadataText', "Likelihood Ratio Test (LRT) measures the log-likelihood ratio gap between nested model specifications. Under H0, Wilks theorem proves Lambda is asymptotically Chi-Square distributed.", ...
                'Methods', [ ...
                    "test(data) -> TestResult Object", ...
                    "assumptions() -> Array of Theoretical Requirements", ...
                    "getMetadata() -> Component Reflection Metadata Struct" ...
                ], ...
                'Capabilities', [ ...
                    "✓ Likelihood Ratio Deviance Lambda = 2*(ln L1 - ln L0)", ...
                    "✓ Asymptotic Chi2(df) Distribution & Critical Value", ...
                    "✓ Exact Asymptotic p-value P(Chi2 >= Lambda)", ...
                    "✓ Statistical Hypothesis Decision", ...
                    "✓ Confidence Upper Bound on Type I Error alpha" ...
                ] ...
            );
        end
    end
end
