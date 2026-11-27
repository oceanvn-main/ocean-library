classdef ReparameterizedLikelihood
    % REPARAMETERIZEDLIKELIHOOD Implements Hamilton (1994) Section 5.9 Parameter Transformations & Unconstrained MLE
    %
    %   TEXTBOOK REFERENCE:
    %       Hamilton, James D. "Time Series Analysis" (1994), Section 5.9:
    %           Eq. (5.9.1): phi = g(lambda) = lambda / (1 + |lambda|)   [Stationarity (-1, 1)]
    %           Eq. (5.9.2): Omega = P * P'                              [Covariance Matrix Positive-Definiteness]
    %
    %   DUAL REPRESENTATION CONTRACT:
    %       1. Symbolics  : LaTeX formulas for parameter mappings g(lambda), Cholesky Omega = P*P', and Jacobians.
    %       2. Computation: Numerical forward/inverse mapping, unconstrained MLE, and Delta Method SE propagation.
    %       3. Diagram    : Port signatures & pipeline block specifications (getPipelineSpec).
    %
    %   Example 1 (Scalar Bounds):
    %       loglikFn = @(x, th) -0.5*log(2*pi*th(2)) - 0.5*((x - th(1)).^2)/th(2);
    %       data = randn(250, 1);
    %       res = OceanMath.TimeSeries.ReparameterizedLikelihood.fit(loglikFn, data, [0.5, 2.0], {'stationarity', 'positive'});
    %
    %   Example 2 (Multivariate Covariance Matrix Omega = P * P'):
    %       Omega = [4.0, 1.2; 1.2, 3.0];
    %       lamVec = OceanMath.TimeSeries.ReparameterizedLikelihood.transformInverseCovariance(Omega);
    %       Omega_rec = OceanMath.TimeSeries.ReparameterizedLikelihood.transformCovarianceForward(lamVec, 2);
    %
    %   See also OceanMath.TimeSeries.QMLECovariance, OceanMath.Information.ScoreFunction, OceanMath.Information.FisherInformationMatrix

    methods (Static)
        function theta = transformForward(lambda, boundTypes)
            % TRANSFORMFORWARD Maps unconstrained parameters lambda in R^a to constrained parameters theta in Domain(g)
            %
            %   Eq. (5.9.1): phi = g(lambda) = lambda / (1 + |lambda|)  for stationarity (-1, 1)
            %   Positive   : sigma^2 = exp(lambda)                     for variances (0, Inf)
            arguments
                lambda (1,:) double
                boundTypes cell = {}
            end

            p = numel(lambda);
            theta = zeros(1, p);

            if isempty(boundTypes)
                boundTypes = repmat({'none'}, 1, p);
            end

            for j = 1:p
                bType = lower(boundTypes{j});
                lam = lambda(j);

                switch bType
                    case {'stationarity', 'unit_circle', 'eq_5_9_1'}
                        % Hamilton Eq [5.9.1]
                        theta(j) = lam / (1 + abs(lam));
                    case {'positive', 'variance', 'pos'}
                        theta(j) = exp(lam);
                    case {'none', 'unconstrained'}
                        theta(j) = lam;
                    otherwise
                        if isnumeric(bType) && numel(bType) == 2
                            L = bType(1); U = bType(2);
                            theta(j) = L + (U - L) / (1 + exp(-lam));
                        else
                            theta(j) = lam;
                        end
                end
            end
        end

        function lambda = transformInverse(theta, boundTypes)
            % TRANSFORMINVERSE Maps constrained parameters theta to unconstrained parameters lambda in R^a
            %
            %   Inverse Eq. (5.9.1): lambda = phi / (1 - |phi|)
            %   Inverse Positive   : lambda = log(sigma^2)
            arguments
                theta (1,:) double
                boundTypes cell = {}
            end

            p = numel(theta);
            lambda = zeros(1, p);

            if isempty(boundTypes)
                boundTypes = repmat({'none'}, 1, p);
            end

            for j = 1:p
                bType = lower(boundTypes{j});
                th = theta(j);

                switch bType
                    case {'stationarity', 'unit_circle', 'eq_5_9_1'}
                        % Clamp strictly inside (-1, 1) to avoid infinite division
                        th_clamped = max(-0.9999, min(0.9999, th));
                        lambda(j) = th_clamped / (1 - abs(th_clamped));
                    case {'positive', 'variance', 'pos'}
                        th_clamped = max(1e-12, th);
                        lambda(j) = log(th_clamped);
                    case {'none', 'unconstrained'}
                        lambda(j) = th;
                    otherwise
                        if isnumeric(bType) && numel(bType) == 2
                            L = bType(1); U = bType(2);
                            th_clamped = max(L + 1e-6, min(U - 1e-6, th));
                            lambda(j) = -log((U - L) / (th_clamped - L) - 1);
                        else
                            lambda(j) = th;
                        end
                end
            end
        end

        function Omega = transformCovarianceForward(lambdaVector, n)
            % TRANSFORMCOVARIANCEFORWARD Reconstructs symmetric positive-definite covariance matrix Omega = P * P' (Hamilton Eq. 5.9.2)
            %   where P is lower-triangular constructed from unconstrained 1D vector lambdaVector of length n*(n+1)/2.
            arguments
                lambdaVector (1, :) double
                n (1, 1) double {mustBePositive, mustBeInteger}
            end

            m = n * (n + 1) / 2;
            if numel(lambdaVector) ~= m
                error('ReparameterizedLikelihood:VectorSizeMismatch', 'Expected vector of length %d for n=%d matrix.', m, n);
            end

            P = zeros(n, n);
            idx = 1;
            for j = 1:n
                for i = j:n
                    P(i, j) = lambdaVector(idx);
                    idx = idx + 1;
                end
            end

            Omega = P * P';
        end

        function lambdaVector = transformInverseCovariance(Omega)
            % TRANSFORMINVERSECOVARIANCE Extracts unconstrained parameter vector from covariance matrix Omega via MATLAB built-in chol
            arguments
                Omega (:, :) double
            end

            [n, m_dim] = size(Omega);
            if n ~= m_dim
                error('ReparameterizedLikelihood:MatrixNotSquare', 'Input covariance matrix must be square.');
            end

            % Built-in MATLAB Cholesky decomposition: P is lower-triangular (P * P' = Omega)
            P = chol((Omega + Omega') / 2, 'lower');

            m = n * (n + 1) / 2;
            lambdaVector = zeros(1, m);
            idx = 1;
            for j = 1:n
                for i = j:n
                    lambdaVector(idx) = P(i, j);
                    idx = idx + 1;
                end
            end
        end

        function J = computeJacobian(lambda, boundTypes)
            % COMPUTEJACOBIAN Calculates diagonal Jacobian matrix J_g = d(g(lambda))/d(lambda)
            arguments
                lambda (1,:) double
                boundTypes cell = {}
            end

            p = numel(lambda);
            J = zeros(p, p);

            if isempty(boundTypes)
                boundTypes = repmat({'none'}, 1, p);
            end

            for j = 1:p
                bType = lower(boundTypes{j});
                lam = lambda(j);

                switch bType
                    case {'stationarity', 'unit_circle', 'eq_5_9_1'}
                        % d/d_lam [ lam / (1 + |lam|) ] = 1 / (1 + |lam|)^2
                        J(j, j) = 1 / ((1 + abs(lam))^2);
                    case {'positive', 'variance', 'pos'}
                        J(j, j) = exp(lam);
                    case {'none', 'unconstrained'}
                        J(j, j) = 1.0;
                    otherwise
                        if isnumeric(bType) && numel(bType) == 2
                            L = bType(1); U = bType(2);
                            sigm = 1 / (1 + exp(-lam));
                            J(j, j) = (U - L) * sigm * (1 - sigm);
                        else
                            J(j, j) = 1.0;
                        end
                end
            end
        end

        function val = evalLikelihood(loglikFn, data, lambda, boundTypes)
            % EVALLIKELIHOOD Evaluates log-likelihood at reparameterized parameter vector theta = g(lambda)
            arguments
                loglikFn (1,1) function_handle
                data double
                lambda (1,:) double
                boundTypes cell = {}
            end

            theta = OceanMath.TimeSeries.ReparameterizedLikelihood.transformForward(lambda, boundTypes);
            T = size(data, 1);
            val = sum(arrayfun(@(t) loglikFn(data(t, :), theta), 1:T));
        end

        function res = fit(loglikFn, data, initialTheta, boundTypes)
            % FIT Performs unconstrained Maximum Likelihood Search over lambda in R^a using fminsearch/fminunc
            arguments
                loglikFn (1,1) function_handle
                data double
                initialTheta (1,:) double
                boundTypes cell = {}
            end

            p = numel(initialTheta);
            if isempty(boundTypes)
                boundTypes = repmat({'none'}, 1, p);
            end

            % 1. Map initial theta guess to unconstrained lambda
            initialLambda = OceanMath.TimeSeries.ReparameterizedLikelihood.transformInverse(initialTheta, boundTypes);

            % 2. Objective function for minimization: -LogLikelihood(g(lambda))
            objFn = @(lam) -OceanMath.TimeSeries.ReparameterizedLikelihood.evalLikelihood(loglikFn, data, lam, boundTypes);

            % 3. Unconstrained numerical optimization over R^a
            options = optimset('Display', 'off', 'MaxFunEvals', 2000, 'MaxIter', 1000, 'TolX', 1e-6, 'TolFun', 1e-6);
            [lambdaHat, negLL, exitflag, output] = fminsearch(objFn, initialLambda, options);

            % 4. Transform back to constrained parameter space thetaHat = g(lambdaHat)
            thetaHat = OceanMath.TimeSeries.ReparameterizedLikelihood.transformForward(lambdaHat, boundTypes);

            % 5. Standard Error Calculations: Delta Method vs Direct Hessian Evaluation (Hamilton p. 146)
            T = size(data, 1);
            h = 1e-4;

            % Hessian with respect to unconstrained lambda
            H_lambda = zeros(p, p);
            for j = 1:p
                for k = 1:p
                    if j == k
                        lp = lambdaHat; lp(j) = lp(j) + h;
                        lm = lambdaHat; lm(j) = lm(j) - h;
                        f_p = objFn(lp); f_c = objFn(lambdaHat); f_m = objFn(lm);
                        H_lambda(j, k) = (f_p - 2*f_c + f_m) / (h^2);
                    else
                        l_pp = lambdaHat; l_pp(j) = lambdaHat(j) + h; l_pp(k) = lambdaHat(k) + h;
                        l_pm = lambdaHat; l_pm(j) = lambdaHat(j) + h; l_pm(k) = lambdaHat(k) - h;
                        l_mp = lambdaHat; l_mp(j) = lambdaHat(j) - h; l_mp(k) = lambdaHat(k) + h;
                        l_mm = lambdaHat; l_mm(j) = lambdaHat(j) - h; l_mm(k) = lambdaHat(k) - h;
                        H_lambda(j, k) = (objFn(l_pp) - objFn(l_pm) - objFn(l_mp) + objFn(l_mm)) / (4 * h^2);
                    end
                end
            end

            try
                varLambda = inv(H_lambda);
            catch
                varLambda = pinv(H_lambda);
            end

            % Delta Method SE: Var(thetaHat) = J_g * Var(lambdaHat) * J_g'
            J_g = OceanMath.TimeSeries.ReparameterizedLikelihood.computeJacobian(lambdaHat, boundTypes);
            varThetaDelta = J_g * varLambda * J_g';
            seDelta = sqrt(max(0, diag(varThetaDelta)))';

            % Direct Hessian Evaluation at thetaHat (Hamilton p. 146 method 1)
            objFnTheta = @(th) -sum(arrayfun(@(t) loglikFn(data(t, :), th), 1:T));
            H_theta = zeros(p, p);
            for j = 1:p
                for k = 1:p
                    if j == k
                        tp = thetaHat; tp(j) = tp(j) + h;
                        tm = thetaHat; tm(j) = tm(j) - h;
                        f_p = objFnTheta(tp); f_c = objFnTheta(thetaHat); f_m = objFnTheta(tm);
                        H_theta(j, k) = (f_p - 2*f_c + f_m) / (h^2);
                    else
                        t_pp = thetaHat; t_pp(j) = thetaHat(j) + h; t_pp(k) = thetaHat(k) + h;
                        t_pm = thetaHat; t_pm(j) = thetaHat(j) + h; t_pm(k) = thetaHat(k) - h;
                        t_mp = thetaHat; t_mp(j) = thetaHat(j) - h; t_mp(k) = thetaHat(k) + h;
                        t_mm = thetaHat; t_mm(j) = thetaHat(j) - h; t_mm(k) = thetaHat(k) - h;
                        H_theta(j, k) = (objFnTheta(t_pp) - objFnTheta(t_pm) - objFnTheta(t_mp) + objFnTheta(t_mm)) / (4 * h^2);
                    end
                end
            end

            try
                varThetaDirect = inv(H_theta);
            catch
                varThetaDirect = pinv(H_theta);
            end
            seDirect = sqrt(max(0, diag(varThetaDirect)))';

            metadataStr = sprintf("Likelihood Reparameterization (Hamilton §5.9 / Eq. 5.9.1 & 5.9.2). " + ...
                "Optimization completed in unconstrained space R^%d with exitflag %d.", p, exitflag);

            res = struct(...
                'ThetaHat', thetaHat, ...
                'LambdaHat', lambdaHat, ...
                'LogLikelihood', -negLL, ...
                'DeltaStandardErrors', seDelta, ...
                'DirectStandardErrors', seDirect, ...
                'VarThetaDelta', varThetaDelta, ...
                'VarThetaDirect', varThetaDirect, ...
                'VarLambda', varLambda, ...
                'Jacobian', J_g, ...
                'BoundTypes', {boundTypes}, ...
                'ExitFlag', exitflag, ...
                'Metadata', metadataStr ...
            );
        end

        function meta = getMetadata()
            % GETMETADATA Self-describing component metadata for system registration
            meta = struct(...
                'ClassName', 'OceanMath.TimeSeries.ReparameterizedLikelihood', ...
                'TextbookRef', 'Hamilton, James D. "Time Series Analysis" (1994), Section 5.9, Eq. (5.9.1) & (5.9.2)', ...
                'SymbolicLatex', '\theta = g(\lambda), \quad \phi = \frac{\lambda}{1 + |\lambda|}, \quad \Omega = P P''', ...
                'Capabilities', {{'Parameter Transformation', 'Multivariate Covariance Parameterization', 'Unconstrained Optimization', 'Delta Method SE Propagation'}} ...
            );
        end

        function spec = getPipelineSpec()
            % GETPIPELINESPEC Exposes block diagram input/output port specifications for studio visualization
            spec = struct(...
                'ComponentName', 'ReparameterizedLikelihood', ...
                'Category', 'Time Series Analysis', ...
                'Description', 'Reparameterizes likelihood function theta = g(lambda) and Omega = P*P'' via unconstrained optimization', ...
                'Inputs', struct('name', {'unconstrainedLogLik', 'data', 'initialTheta'}, 'type', {'function_handle', 'double', 'double'}, 'shape', {'1x1', 'N x K', '1 x p'}), ...
                'Outputs', struct('name', {'thetaHat', 'seDelta', 'res'}, 'type', {'double', 'double', 'struct'}, 'shape', {'1 x p', '1 x p', 'scalar'}) ...
            );
        end

        function str = getLaTeXRepresentation()
            % GETLATEXREPRESENTATION Returns LaTeX mathematical representation of the reparameterization
            str = sprintf(['\\begin{aligned}\n' ...
                '\\text{Eq. (5.9.1): } & \\phi = g(\\lambda) = \\frac{\\lambda}{1 + |\\lambda|}, \\quad \\phi \\in (-1, 1) \\\\\n' ...
                '\\text{Eq. (5.9.2): } & \\Omega = P P'', \\quad P \\text{ lower-triangular via chol(Omega, ''lower'')}\n' ...
                '\\end{aligned}']);
        end
    end
end
