classdef ARMA < OceanMath.TimeSeries.Models.ITimeSeriesModel
    % ARMA Concrete class representing an Autoregressive Moving Average process.
    %
    % Represents the model:
    %   Y(t) = Constant + phi_1*Y(t-1) + ... + phi_p*Y(t-p) + e(t) + theta_1*e(t-1) + ... + theta_q*e(t-q)
    % where e(t) is a Gaussian white noise sequence.
    %
    % See also: OceanMath.TimeSeries.Models.ITimeSeriesModel
    
    properties
        ARCoeffs (1, :) double = []
        MACoeffs (1, :) double = []
        Constant (1, 1) double = 0
    end
    
    methods
        function obj = ARMA(opts)
            % ARMA Constructor for the ARMA process model.
            %
            % Options:
            %   ARCoeffs - (1, :) double, autoregressive parameters (phi_1, ..., phi_p)
            %   MACoeffs - (1, :) double, moving average parameters (theta_1, ..., theta_q)
            %   Constant - (1, 1) double, constant/drift term (default 0)
            arguments
                opts.ARCoeffs (1, :) double = []
                opts.MACoeffs (1, :) double = []
                opts.Constant (1, 1) double = 0
            end
            obj.ARCoeffs = opts.ARCoeffs;
            obj.MACoeffs = opts.MACoeffs;
            obj.Constant = opts.Constant;
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
            p = length(obj.ARCoeffs);
            poly = [1, -obj.ARCoeffs];
            r = roots(poly);
            tf = all(abs(r) < 1);
        end
        
        function tf = isInvertible(obj)
            % ISINVERTIBLE True if the process is invertible.
            %
            % Evaluates the roots of the MA characteristic polynomial:
            %   z^q + theta_1*z^(q-1) + ... + theta_q = 0
            % The process is invertible if all roots lie inside the unit circle.
            if isempty(obj.MACoeffs)
                tf = true;
                return;
            end
            q = length(obj.MACoeffs);
            poly = [1, obj.MACoeffs];
            r = roots(poly);
            tf = all(abs(r) < 1);
        end
        
        function y = simulate(obj, N, sigma)
            % SIMULATE Generate a realization of the ARMA process.
            %
            % INPUTS:
            %   N     - (1,1) double, length of simulated series
            %   sigma - (1,1) double, standard deviation of white noise
            %
            % OUTPUTS:
            %   y     - (N x 1) double, simulated time series
            arguments
                obj
                N     (1, 1) double {mustBePositive, mustBeInteger}
                sigma (1, 1) double {mustBePositive}
            end
            
            % Use burn-in steps to wash out initialization transients
            burnIn = 100;
            totalN = N + burnIn;
            
            % Generate Gaussian white noise
            e = sigma * randn(totalN, 1);
            
            % Setup filter transfer function coefficients
            % Y_t = Constant + sum(phi * Y_prev) + e_t + sum(theta * e_prev)
            b = [1, obj.MACoeffs];
            a = [1, -obj.ARCoeffs];
            
            % First, compute the numerator filtering (moving average parts)
            rhs = obj.Constant + filter(b, 1, e);
            
            % Next, compute the denominator filtering (autoregressive parts)
            yFull = filter(1, a, rhs);
            
            % Discard the burn-in period
            y = yFull(burnIn + 1 : end);
        end
        
        function rho = theoreticalACF(obj, lags)
            % THEORETICALACF Compute the theoretical ACF up to a maximum lag.
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
                warning('ARMA:NonStationary', 'Theoretical ACF is undefined for non-stationary processes.');
                rho = [1, nan(1, lags)];
                return;
            end
            
            % Analytical shortcut for White Noise
            if isempty(obj.ARCoeffs) && isempty(obj.MACoeffs)
                rho = [1, zeros(1, lags)];
                return;
            end
            
            % Analytical shortcut for AR(1)
            if length(obj.ARCoeffs) == 1 && isempty(obj.MACoeffs)
                phi = obj.ARCoeffs(1);
                rho = phi .^ (0:lags);
                return;
            end
            
            % Analytical shortcut for MA(1)
            if isempty(obj.ARCoeffs) && length(obj.MACoeffs) == 1
                theta = obj.MACoeffs(1);
                rho_1 = theta / (1 + theta^2);
                rho = zeros(1, lags + 1);
                rho(1) = 1;
                if lags >= 1
                    rho(2) = rho_1;
                end
                return;
            end
            
            % General ARMA(p, q) via MA(infinity) expansion
            b = [1, obj.MACoeffs];
            a = [1, -obj.ARCoeffs];
            
            % Generate impulse response coefficients up to a large horizon
            M = max(5000, 10 * lags);
            impulse = [1; zeros(M, 1)];
            psi = filter(b, a, impulse);
            
            % Compute autocovariances
            gamma = zeros(1, lags + 1);
            for k = 0:lags
                gamma(k+1) = sum(psi(1 : M-k+1) .* psi(k+1 : M+1));
            end
            
            % Normalize to get ACF
            rho = gamma / gamma(1);
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
                warning('ARMA:NonStationary', 'Theoretical PACF is undefined for non-stationary processes.');
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
        
        function str = getEquation(obj)
            % GETEQUATION Return a string representation of the model equation.
            strParts = "Y(t) = ";
            if obj.Constant ~= 0
                strParts = strParts + sprintf("%.2f", obj.Constant);
            end
            
            % AR part
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
            
            % Noise
            if isempty(obj.ARCoeffs) && obj.Constant == 0
                strParts = strParts + "e(t)";
            else
                strParts = strParts + " + e(t)";
            end
            
            % MA part
            for j = 1:length(obj.MACoeffs)
                val = obj.MACoeffs(j);
                if val >= 0
                    signStr = " + ";
                else
                    signStr = " - ";
                    val = -val;
                end
                strParts = strParts + sprintf("%s%.2f*e(t-%d)", signStr, val, j);
            end
            
            str = string(strParts);
        end
    end
end
