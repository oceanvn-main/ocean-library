classdef ScoreFunction
    % SCOREFUNCTION Computes individual log-likelihood gradients (Score vector V), sample mean E[V], score covariance matrix, and tests standard regularity conditions E_theta[V] = 0.
    %
    %   TEXTBOOK REFERENCE:Cover & Thomas, "Elements of Information Theory" (2nd Ed.), Section 11.10.
    %       Definition (Eq. 11.268):
    %       
    %           V(X; theta) = (d/d theta) ln f(X; theta)
    %
    %       Regularity Identity (Eq. 11.269):
    %           E_theta[ V(X; theta) ] = integral ( (d/d theta f(x; theta)) / f(x; theta) ) * f(x; theta) dx = d/d theta (1) = 0
    %
    %   Example:
    %       loglik = @(x, th) -0.5*log(2*pi*th(2)) - 0.5*((x - th(1)).^2)/th(2);
    %       data = randn(100, 1);
    %       res = OceanMath.Information.ScoreFunction.compute(loglik, data, [0, 1]);
    %
    %   See also FisherInformationMatrix, InformationEngine

    methods (Static)
        function res = compute(loglikFn, data, theta)
            % COMPUTE Calculates individual score vectors V_i(theta) = d ln f(x_i; theta) / d theta
            arguments
                loglikFn (1,1) function_handle
                data double
                theta (1,:) double
            end

            N = size(data, 1);
            p = numel(theta);
            V = zeros(N, p);
            h = 1e-5;

            for i = 1:N
                x_i = data(i, :);
                for j = 1:p
                    theta_plus = theta;
                    theta_minus = theta;
                    theta_plus(j) = theta_plus(j) + h;
                    theta_minus(j) = theta_minus(j) - h;

                    ll_plus = loglikFn(x_i, theta_plus);
                    ll_minus = loglikFn(x_i, theta_minus);
                    V(i, j) = (ll_plus - ll_minus) / (2 * h);
                end
            end

            meanScore = mean(V, 1);
            scoreVar = cov(V);
            scoreNorm = norm(meanScore);
            passedRegularity = scoreNorm < 0.1;

            res = struct(...
                'ScoreMatrix', V, ...
                'MeanScore', meanScore, ...
                'ScoreNorm', scoreNorm, ...
                'PassedRegularity', passedRegularity, ...
                'ScoreCovariance', scoreVar, ...
                'SampleSize', N, ...
                'NumParams', p, ...
                'Metadata', "Score vector V = d/d theta ln f(X; theta). Under regularity conditions, E[V] = 0." ...
            );
        end

        function [passed, scoreNorm] = checkRegularity(V, tol)
            % CHECKREGULARITY Assesses if expected score E[V] is statistically close to 0 vector
            arguments
                V double
                tol (1,1) double = 0.1
            end
            meanScore = mean(V, 1);
            scoreNorm = norm(meanScore);
            passed = scoreNorm < tol;
        end

        function meta = getMetadata()
            % GETMETADATA Returns self-describing component metadata, LaTeX representations, and capability specs
            meta = struct(...
                'ClassName', "OceanMath.Information.ScoreFunction", ...
                'Package', "+OceanMath/+Information/ScoreFunction.m", ...
                'Category', "OceanMath.Information", ...
                'TextbookRef', "Cover & Thomas, Elements of Information Theory (2nd Ed.), Section 11.10", ...
                'TextbookEq', "Eq. (11.268): V(X; theta) = (d/d theta) ln f(X; theta)", ...
                'SymbolicLatex', "V(X; \\theta) = \\frac{\\partial}{\\partial \\theta} \\ln f(X; \\theta) = \\frac{\\nabla_{\\theta} f(X; \\theta)}{f(X; \\theta)} \\quad \\text{(Cover \\& Thomas Eq. 11.268)}", ...
                'RegularityLatex', "\\mathbb{E}_{\\theta}[V(X; \\theta)] = \\int \\frac{\\frac{\\partial}{\\partial \\theta} f(x; \\theta)}{f(x; \\theta)} f(x; \\theta) dx = \\frac{\\partial}{\\partial \\theta} (1) = 0 \\quad \\text{(Cover \\& Thomas Eq. 11.269)}", ...
                'MetadataText', "Textbook Definition (Cover & Thomas Eq. 11.268-11.269): Score function V is the gradient vector of the log-likelihood function with respect to parameter vector theta. Under standard regularity conditions, its expectation is identically 0.", ...
                'Methods', [ ...
                    "compute(loglikFn, data, theta) -> ScoreMatrix V_i & MeanScore E[V]", ...
                    "checkRegularity(V, tol) -> [passed, scoreNorm]", ...
                    "getMetadata() -> Component Reflection Metadata Struct" ...
                ], ...
                'Capabilities', [ ...
                    "✓ Cover & Thomas Eq. (11.268) Score Vector V_i = d/d theta ln f(x_i; theta)", ...
                    "✓ Cover & Thomas Eq. (11.269) Regularity Verification E_theta[V] = 0", ...
                    "✓ Individual Gradient Vectors V_i", ...
                    "✓ Empirical Score Covariance Matrix cov(V)", ...
                    "✓ Central Finite Difference Gradient Estimator" ...
                ] ...
            );
        end
    end
end
