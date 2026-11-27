classdef QMLECovariance
    % QMLECOVARIANCE Computes Quasi-Maximum Likelihood (QMLE) misspecification-robust standard errors and sandwich covariance matrix for Time Series Analysis.
    %
    %   TEXTBOOK REFERENCE:
    %       Hamilton, James D. "Time Series Analysis" (1994), Section 5.8, Eq. (5.8.7):
    %           E[(theta_hat - theta_0)(theta_hat - theta_0)'] = T^-1 { I_2D * I_OP^-1 * I_2D }^-1
    %                                                         = (1/T) * I_2D^-1 * I_OP * I_2D^-1
    %       White, Halbert. "Maximum Likelihood Estimation of Misspecified Models", Econometrica (1982).
    %
    %   DUAL REPRESENTATION CONTRACT:
    %       1. Symbolics  : LaTeX formulas for Hessian I_2D, Score Outer-Product I_OP, and Sandwich V_QMLE.
    %       2. Computation: Numerical forward pass estimating score vectors, Hessian, OPG, and robust SEs.
    %       3. Diagram    : Port signatures & pipeline block specifications (getPipelineSpec).
    %
    %   Example 1 (General Log-Likelihood):
    %       loglikFn = @(x, th) -0.5*log(2*pi*th(2)) - 0.5*((x - th(1)).^2)/th(2);
    %       data = randn(200, 1);
    %       res = OceanMath.TimeSeries.QMLECovariance.compute(loglikFn, data, [0, 1]);
    %
    %   Example 2 (AR(1) Time Series Model):
    %       y = filter(1, [1, -0.6], randn(300, 1));
    %       res = OceanMath.TimeSeries.QMLECovariance.fitAR(y, 1);
    %
    %   See also OceanMath.TimeSeries.Models.AR, OceanMath.Information.ScoreFunction, OceanMath.Information.FisherInformationMatrix

    methods (Static)
        function res = compute(loglikFn, data, thetaHat)
            % COMPUTE Calculates robust QMLE sandwich covariance matrix V_QMLE = (1/T) * I_2D^-1 * I_OP * I_2D^-1
            arguments
                loglikFn (1,1) function_handle
                data double
                thetaHat (1,:) double
            end

            T = size(data, 1);
            p = numel(thetaHat);
            h = 1e-4;

            % 1. Compute Individual Scores s_t(thetaHat) via finite differences
            S = zeros(T, p);
            for t = 1:T
                x_t = data(t, :);
                for j = 1:p
                    th_plus = thetaHat; th_plus(j) = th_plus(j) + h;
                    th_minus = thetaHat; th_minus(j) = th_minus(j) - h;
                    ll_plus = loglikFn(x_t, th_plus);
                    ll_minus = loglikFn(x_t, th_minus);
                    S(t, j) = (ll_plus - ll_minus) / (2 * h);
                end
            end

            % 2. Outer Product of Scores (OPG) Information Matrix: I_OP = (1/T) * sum_{t=1}^T s_t s_t'
            I_OP = (S' * S) / T;
            I_OP = (I_OP + I_OP') / 2; % Enforce symmetry

            % 3. Negative Second Derivative (Hessian) Information Matrix: I_2D = -(1/T) * sum_{t=1}^T grad^2 ln f(x_t; thetaHat)
            H = zeros(p, p);
            totalLogLik = @(th) sum(arrayfun(@(idx) loglikFn(data(idx, :), th), 1:T));

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

            I_2D = -H / T;
            I_2D = (I_2D + I_2D') / 2; % Enforce symmetry

            % 4. Naive vs Sandwich Covariance Inversion
            try
                inv_I2D = inv(I_2D);
            catch
                inv_I2D = pinv(I_2D);
            end

            try
                inv_IOP = inv(I_OP);
            catch
                inv_IOP = pinv(I_OP);
            end

            % Naive Covariance: (1/T) * I_2D^-1
            covNaive = inv_I2D / T;
            seNaive = sqrt(max(0, diag(covNaive)))';

            % Sandwich Covariance: (1/T) * I_2D^-1 * I_OP * I_2D^-1 (Hamilton [5.8.7])
            covQMLE = (inv_I2D * I_OP * inv_I2D) / T;
            covQMLE = (covQMLE + covQMLE') / 2;
            seQMLE = sqrt(max(0, diag(covQMLE)))';

            % 5. Misspecification Diagnostics
            inflationRatio = seQMLE ./ max(1e-12, seNaive);
            misspecificationDistance = norm(I_2D - I_OP, 'fro');
            isMisspecified = misspecificationDistance > 0.15 * norm(I_2D, 'fro');

            metadataStr = sprintf("QMLE Sandwich Covariance Matrix (Hamilton Eq. 5.8.7 / White 1982). " + ...
                "Misspecification distance ||I_2D - I_OP||_F = %.4f. Model misspecification detected: %s.", ...
                misspecificationDistance, string(isMisspecified));

            res = struct(...
                'QMLECovariance', covQMLE, ...
                'NaiveCovariance', covNaive, ...
                'QMLEStandardErrors', seQMLE, ...
                'NaiveStandardErrors', seNaive, ...
                'InflationRatio', inflationRatio, ...
                'HessianInformation', I_2D, ...
                'OPGInformation', I_OP, ...
                'MisspecificationDistance', misspecificationDistance, ...
                'IsMisspecified', isMisspecified, ...
                'SampleSize', T, ...
                'NumParams', p, ...
                'Metadata', metadataStr ...
            );
        end

        function res = fitAR(y, p)
            % FITAR Convenience method fitting an AR(p) model with both Naive and robust QMLE standard errors
            arguments
                y (:, 1) double
                p (1, 1) double {mustBePositive, mustBeInteger}
            end

            T = length(y);
            Y_dep = y(p+1:end);
            X_reg = zeros(T - p, p);
            for j = 1:p
                X_reg(:, j) = y(p+1-j:end-j);
            end

            % OLS estimate: phi = (X'X)^-1 X'y
            phiHat = (X_reg' * X_reg) \ (X_reg' * Y_dep);
            residuals = Y_dep - X_reg * phiHat;
            sigma2Hat = mean(residuals.^2);
            thetaHat = [phiHat', sigma2Hat];

            % Conditional AR log-likelihood per observation t
            arLogLik = @(row_t, th) -0.5*log(2*pi*th(end)) - 0.5 * ((row_t(1) - sum(th(1:end-1) .* row_t(2:end)))^2) / th(end);
            dataMat = [Y_dep, X_reg];

            res = OceanMath.TimeSeries.QMLECovariance.compute(arLogLik, dataMat, thetaHat);
            res.ARCoeffs = phiHat';
            res.ResidualVariance = sigma2Hat;
        end

        function meta = getMetadata()
            % GETMETADATA Returns self-describing component metadata, LaTeX representations, and capability specs
            meta = struct(...
                'ClassName', "OceanMath.TimeSeries.QMLECovariance", ...
                'Package', "+OceanMath/+TimeSeries/QMLECovariance.m", ...
                'Category', "OceanMath.TimeSeries", ...
                'Tier', 3, ...
                'TextbookRef', "Hamilton (1994) Section 5.8, Eq. (5.8.7); White (1982) Econometrica", ...
                'TextbookEq', "E[(theta_hat - theta_0)(theta_hat - theta_0)'] = T^-1 { I_2D * I_OP^-1 * I_2D }^-1", ...
                'SymbolicLatex', "\mathbf{V}_{\text{QMLE}} = T^{-1} \left\{ \hat{\mathcal{I}}_{2D} \hat{\mathcal{I}}_{OP}^{-1} \hat{\mathcal{I}}_{2D} \right\}^{-1} = \frac{1}{T} \hat{\mathcal{I}}_{2D}^{-1} \hat{\mathcal{I}}_{OP} \hat{\mathcal{I}}_{2D}^{-1} \quad \text{(Hamilton Eq. 5.8.7 / White 1982)}", ...
                'HessianLatex', "\hat{\mathcal{I}}_{2D} = -\frac{1}{T} \sum_{t=1}^T \frac{\partial^2 \ln f(y_t \mid x_t; \hat{\boldsymbol{\theta}})}{\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}'}", ...
                'OuterProductLatex', "\hat{\mathcal{I}}_{OP} = \frac{1}{T} \sum_{t=1}^T s_t(\hat{\boldsymbol{\theta}}) s_t(\hat{\boldsymbol{\theta}})'", ...
                'MetadataText', "Quasi-Maximum Likelihood (QMLE) Huber-White Sandwich Estimator for Time Series Analysis. Computes misspecification-robust variance-covariance matrix V_QMLE = (1/T) * I_2D^-1 * I_OP * I_2D^-1 when innovation density or likelihood function is misspecified.", ...
                'Methods', [ ...
                    "compute(loglikFn, data, thetaHat) -> QMLECovariance, StandardErrors, MisspecificationDistance", ...
                    "fitAR(y, p) -> Fit AR(p) model with both Naive and robust QMLE Standard Errors", ...
                    "getMetadata() -> Component Reflection Metadata Struct", ...
                    "getPipelineSpec() -> Generic OceanLearn Node Specification Struct" ...
                ], ...
                'Capabilities', [ ...
                    "✓ Hamilton Eq. (5.8.7) QMLE Robust Sandwich Covariance V_QMLE", ...
                    "✓ White (1982) Misspecification Distance ||I_2D - I_OP||_F", ...
                    "✓ Score Outer Product Information Matrix I_OP", ...
                    "✓ Observed Hessian Information Matrix I_2D", ...
                    "✓ Naive vs Robust Standard Error Inflation Ratio", ...
                    "✓ Automated AR(p) Misspecification Robust Inference" ...
                ] ...
            );
        end

        function pipeSpec = getPipelineSpec()
            % GETPIPELINESPEC Returns generic OceanLearn diagram pipeline node signature
            pipeSpec = struct(...
                'NodeID', "qmle_sandwich_covariance", ...
                'Label', "QMLE Sandwich Estimator (Hamilton §5.8)", ...
                'Domain', "time_series", ...
                'Tier', 3, ...
                'Inputs', struct(...
                    'loglikFn', struct('domain', "function", 'type', "handle"), ...
                    'data', struct('domain', "real_matrix", 'type', "double"), ...
                    'thetaHat', struct('domain', "parameter_vector", 'type', "double") ...
                ), ...
                'Outputs', struct(...
                    'QMLECovariance', struct('domain', "matrix", 'shape', "pxp"), ...
                    'QMLEStandardErrors', struct('domain', "vector", 'shape', "1xp"), ...
                    'MisspecificationDistance', struct('domain', "scalar", 'shape', "1x1") ...
                ), ...
                'IsSymbolic', true, ...
                'SymbolicLatex', "\mathbf{V}_{\text{QMLE}} = \frac{1}{T} \hat{\mathcal{I}}_{2D}^{-1} \hat{\mathcal{I}}_{OP} \hat{\mathcal{I}}_{2D}^{-1}" ...
            );
        end
    end
end
