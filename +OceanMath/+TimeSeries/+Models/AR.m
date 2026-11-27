classdef AR < OceanMath.TimeSeries.Models.ITimeSeriesModel
    % AR Concrete class representing an Autoregressive (AR) process.
    %
    % Represents the model:
    %   Y(t) = Constant + phi_1*Y(t-1) + ... + phi_p*Y(t-p) + e(t)
    % where e(t) is a Gaussian white noise sequence with variance NoiseStd^2.
    %
    % See also: OceanMath.TimeSeries.Models.ITimeSeriesModel
    
    properties
        ARCoeffs (1, :) double = []
        Constant (1, 1) double = 0
        NoiseStd (1, 1) double = 1.0
    end
    
    methods
        function obj = AR(opts)
            % AR Constructor for the AR process model.
            %
            % Options:
            %   ARCoeffs - (1, :) double, autoregressive parameters (phi_1, ..., phi_p)
            %   Constant - (1, 1) double, constant/intercept term (default 0)
            %   NoiseStd - (1, 1) double, noise standard deviation (default 1.0)
            arguments
                opts.ARCoeffs (1, :) double = []
                opts.Constant (1, 1) double = 0
                opts.NoiseStd (1, 1) double = 1.0
            end
            obj.ARCoeffs = opts.ARCoeffs;
            obj.Constant = opts.Constant;
            obj.NoiseStd = opts.NoiseStd;
        end
        
        function tf = isStationary(obj)
            % ISSTATIONARY True if the process is stationary.
            %
            % Evaluates the roots of the AR characteristic polynomial:
            %   z^p - phi_1*z^(p-1) - ... - phi_p = 0
            % The process is stationary if all roots lie inside the unit circle.
            if isempty(obj.ARCoeffs)
                tf = true;
                return;
            end
            poly = [1, -obj.ARCoeffs];
            r = roots(poly);
            tf = all(abs(r) < 1);
        end
        
        function tf = isInvertible(~)
            % ISINVERTIBLE Autoregressive processes are always invertible.
            tf = true;
        end
        
        function y = simulate(obj, N, sigma)
            % SIMULATE Generate a realization of the AR process.
            %
            % INPUTS:
            %   N     - (1,1) double, length of simulated series
            %   sigma - (1,1) double, standard deviation of white noise (optional)
            %
            % OUTPUTS:
            %   y     - (N x 1) double, simulated time series
            arguments
                obj
                N     (1, 1) double {mustBePositive, mustBeInteger}
                sigma (1, 1) double = -1
            end
            
            sig = sigma;
            if sig == -1
                sig = obj.NoiseStd;
            end
            
            % Use burn-in steps to wash out initialization transients
            burnIn = 100;
            totalN = N + burnIn;
            
            % Generate Gaussian white noise
            e = sig * randn(totalN, 1);
            
            % Setup filter transfer function coefficients
            % Y_t = Constant + sum(phi * Y_prev) + e_t
            a = [1, -obj.ARCoeffs];
            
            % Compute filter
            rhs = obj.Constant + e;
            yFull = filter(1, a, rhs);
            
            % Discard the burn-in period
            y = yFull(burnIn + 1 : end);
        end
        
        function rho = theoreticalACF(obj, lags)
            % THEORETICALACF Compute the theoretical ACF up to a maximum lag.
            %
            % Solves the Yule-Walker equations exactly for AR(p) processes.
            %
            % INPUTS:
            %   lags - (1,1) double, maximum lag (lags >= 0)
            %
            % OUTPUTS:
            %   rho  - (1 x lags+1) double, ACF values for lags 0 to lags
            arguments
                obj
                lags (1, 1) double {mustBeNonnegative, mustBeInteger}
            end
            
            if ~obj.isStationary()
                warning('AR:NonStationary', 'Theoretical ACF is undefined for non-stationary processes.');
                rho = [1, nan(1, lags)];
                return;
            end
            
            % Analytical shortcut for White Noise
            if isempty(obj.ARCoeffs)
                rho = [1, zeros(1, lags)];
                return;
            end
            
            p = length(obj.ARCoeffs);
            
            % Setup Yule-Walker system: M * rho_{1..p} = b
            M = eye(p);
            for k = 1:p
                for j = 1:p
                    % terms for rho_j in equation k:
                    % coefficient of rho_j in sum_{i=1}^p phi_i * rho_{|k-i|}
                    if (k-j) >= 1 && (k-j) <= p
                        M(k, j) = M(k, j) - obj.ARCoeffs(k-j);
                    end
                    if (k+j) >= 1 && (k+j) <= p
                        M(k, j) = M(k, j) - obj.ARCoeffs(k+j);
                    end
                end
            end
            
            b = obj.ARCoeffs(:);
            rho_p = M \ b;
            
            % Combine: rho_0 = 1, then rho_1 ... rho_p
            rho = [1, rho_p'];
            
            % Compute higher lags recursively
            if lags > p
                for k = p+1:lags
                    val = 0;
                    for i = 1:p
                        val = val + obj.ARCoeffs(i) * rho(k - i + 1);
                    end
                    rho(k+1) = val;
                end
            end
            
            % Format to exactly lags + 1 elements
            if length(rho) > lags + 1
                rho = rho(1:lags+1);
            end
        end
        
        function phi = theoreticalPACF(obj, lags)
            % THEORETICALPACF Compute the theoretical PACF up to a maximum lag.
            %
            % Uses the Durbin-Levinson recursion on the theoretical ACF.
            %
            % INPUTS:
            %   lags - (1,1) double, maximum lag (lags >= 1)
            %
            % OUTPUTS:
            %   phi  - (1 x lags) double, PACF values for lags 1 to lags
            arguments
                obj
                lags (1, 1) double {mustBePositive, mustBeInteger}
            end
            
            if ~obj.isStationary()
                warning('AR:NonStationary', 'Theoretical PACF is undefined for non-stationary processes.');
                phi = nan(1, lags);
                return;
            end
            
            % Get theoretical ACF up to lags
            rho = obj.theoreticalACF(lags);
            
            % Solve Durbin-Levinson recursion
            phiMatrix = zeros(lags, lags);
            phiMatrix(1, 1) = rho(2);
            
            for k = 2:lags
                num = rho(k+1) - sum(phiMatrix(k-1, 1:k-1) .* rho(k:-1:2));
                den = 1 - sum(phiMatrix(k-1, 1:k-1) .* rho(2:k));
                phiMatrix(k, k) = num / den;
                
                for j = 1:k-1
                    phiMatrix(k, j) = phiMatrix(k-1, j) - phiMatrix(k, k) * phiMatrix(k-1, k-j);
                end
            end
            
            % PACF is the diagonal of the Durbin-Levinson matrix
            phi = diag(phiMatrix)';
        end
        
        function [forecastMean, confidenceBands] = forecast(obj, yHistory, horizon)
            % FORECAST Generate multi-step forecasts with 95% confidence intervals.
            %
            % INPUTS:
            %   yHistory - (N x 1) or (1 x N) double, historical sequence
            %   horizon  - (1,1) double, forecast steps
            %
            % OUTPUTS:
            %   forecastMean    - (horizon x 1) double, predicted future values
            %   confidenceBands - (horizon x 2) double, [lower, upper] limits
            arguments
                obj
                yHistory        (:, :) double
                horizon         (1, 1) double {mustBePositive, mustBeInteger}
            end
            
            if ~isvector(yHistory)
                error('AR:forecast:InvalidHistory', 'yHistory must be a vector.');
            end
            
            yHistory = yHistory(:);
            p = length(obj.ARCoeffs);
            N = length(yHistory);
            
            if N < p
                error('AR:forecast:HistoryTooShort', ...
                    'yHistory must have length at least equal to AR order %d.', p);
            end
            
            forecastMean = zeros(horizon, 1);
            history = yHistory;
            
            % 1. Recursive point forecasts
            for h = 1:horizon
                past = history(end - p + 1 : end);
                % Dot product of ARCoeffs with flipped past values
                val = obj.Constant + sum(obj.ARCoeffs .* flip(past(:)'));
                forecastMean(h) = val;
                history = [history; val];
            end
            
            % 2. Calculate forecast uncertainty variance
            % Compute impulse response (psi coefficients) of the model
            b = 1;
            a = [1, -obj.ARCoeffs];
            imp = [1; zeros(horizon - 1, 1)];
            psi = filter(b, a, imp);
            
            confidenceBands = zeros(horizon, 2);
            for h = 1:horizon
                % Standard error for h-step ahead prediction
                stdErr = obj.NoiseStd * sqrt(sum(psi(1:h).^2));
                confidenceBands(h, 1) = forecastMean(h) - 1.96 * stdErr;
                confidenceBands(h, 2) = forecastMean(h) + 1.96 * stdErr;
            end
        end
        
        function str = getEquation(obj)
            % GETEQUATION Return a string representation of the model equation.
            strParts = "Y(t) = ";
            if obj.Constant ~= 0
                strParts = strParts + sprintf("%.2f", obj.Constant);
            end
            
            % AR coefficients
            for i = 1:length(obj.ARCoeffs)
                val = obj.ARCoeffs(i);
                if val >= 0
                    signStr = " + ";
                else
                    signStr = " - ";
                    val = -val;
                end
                if i == 1 && obj.Constant == 0 && signStr == " + "
                    signStr = "";
                end
                strParts = strParts + sprintf("%s%.2f*Y(t-%d)", signStr, val, i);
            end
            
            % Add noise term
            if isempty(obj.ARCoeffs) && obj.Constant == 0
                strParts = strParts + "e(t)";
            else
                strParts = strParts + " + e(t)";
            end
            
            str = string(strParts);
        end
    end
    
    methods (Static)
        function model = fit(y, options)
            % FIT Estimate AR parameters from a series via Yule-Walker.
            %
            % Options:
            %   Order      - (1,1) double, specific AR order. If empty, performs order selection.
            %   MaxOrder   - (1,1) double, maximum AR order for AIC/BIC sweeps (default 10)
            %   Criterion  - (1,1) string, select order by "AIC" or "BIC" (default "BIC")
            arguments
                y           (:, :) double
                options.Order       double = []
                options.MaxOrder    (1, 1) double {mustBePositive, mustBeInteger} = 10
                options.Criterion   (1, 1) string {mustBeMember(options.Criterion, ["AIC", "BIC"])} = "BIC"
            end
            
            if ~isvector(y)
                error('AR:fit:InvalidInput', 'y must be a vector.');
            end
            
            y = y(:);
            N = length(y);
            
            % Check if specific order is requested
            if ~isempty(options.Order)
                p = options.Order;
                if p < 0 || p >= N
                    error('AR:fit:InvalidOrder', 'Order must be non-negative and less than data length.');
                end
                [phi, c, noiseStd] = OceanMath.TimeSeries.Models.AR.fitOrder_(y, p);
            else
                % Sweep and select order
                maxP = min(options.MaxOrder, N - 3);
                bestMetric = Inf;
                bestP = 1;
                
                for p_sweep = 1:maxP
                    [~, ~, noiseStd_sweep] = OceanMath.TimeSeries.Models.AR.fitOrder_(y, p_sweep);
                    sigma_sq = noiseStd_sweep^2;
                    
                    % Calculate selection metric
                    if options.Criterion == "AIC"
                        metric = N * log(sigma_sq) + 2 * (p_sweep + 1);
                    else
                        metric = N * log(sigma_sq) + (p_sweep + 1) * log(N);
                    end
                    
                    if metric < bestMetric
                        bestMetric = metric;
                        bestP = p_sweep;
                    end
                end
                
                p = bestP;
                [phi, c, noiseStd] = OceanMath.TimeSeries.Models.AR.fitOrder_(y, p);
            end
            
            % Create fitted model instance
            model = OceanMath.TimeSeries.Models.AR(...
                'ARCoeffs', phi', ...
                'Constant', c, ...
                'NoiseStd', noiseStd ...
            );
        end
    end
    
    methods (Static, Access = private)
        function [phi, c, noiseStd] = fitOrder_(y, p)
            % FITORDER_ Helper to estimate Yule-Walker parameters for order p
            N = length(y);
            
            if p == 0
                phi = [];
                c = mean(y);
                noiseStd = std(y);
                return;
            end
            
            % Center the series
            yMean = mean(y);
            yDev = y - yMean;
            
            % Compute sample autocovariances up to lag p
            c_cov = zeros(p + 1, 1);
            c_cov(1) = sum(yDev.^2) / N;
            for k = 1:p
                c_cov(k+1) = sum(yDev(k+1:end) .* yDev(1:end-k)) / N;
            end
            
            % Solve Yule-Walker: Gamma * phi = gamma
            Gamma = toeplitz(c_cov(1:p));
            gamma = c_cov(2:p+1);
            
            % Solve system (with numerical fallback)
            try
                phi = Gamma \ gamma;
            catch
                % Fallback to pseudo-inverse if singular
                phi = pinv(Gamma) * gamma;
            end
            
            % Estimate residual variance
            sigma_sq = c_cov(1) - sum(phi .* c_cov(2:p+1));
            if sigma_sq <= 0
                sigma_sq = 1e-12; % Prevent zero/negative variance
            end
            
            noiseStd = sqrt(sigma_sq);
            
            % Constant intercept estimate
            c = yMean * (1 - sum(phi));
        end
    end
end
