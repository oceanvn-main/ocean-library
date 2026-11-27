classdef ARMAProcess
    % ARMAPROCESS Synthetic ARMA(1,1) time series generator using MATLAB built-ins.
    %
    % NAMESPACE: OceanStats.Time
    %
    % Generates synthetic realizations of an ARMA(1,1) stochastic process:
    % y(t) = phi * y(t-1) + e(t) + theta * e(t-1), where e(t) ~ N(0, sigma^2).
    %
    % Uses Econometrics Toolbox arima() / simulate() if available, otherwise
    % uses MATLAB built-in digital filter engine filter([1, theta], [1, -phi], e).
    %
    % Author: Ocean Library System
    % Date: 2026-09-03
    
    methods (Static)
        function y = simulate(phi, theta, sigma2, T, seed)
            % SIMULATE Generates a dynamic time series realization.
            %
            % Args:
            %   phi (double): Autoregressive parameter |phi| < 1.
            %   theta (double): Moving average parameter |theta| < 1.
            %   sigma2 (double): Innovation variance sigma^2 > 0.
            %   T (int): Time steps / length of series (default 500).
            %   seed (int): Optional random seed for reproducibility.
            %
            % Returns:
            %   y (double): Synthesized [T x 1] time series array.
            
            arguments
                phi (1,1) double
                theta (1,1) double
                sigma2 (1,1) double = 1.0
                T (1,1) double = 500
                seed = []
            end
            
            if ~isempty(seed)
                rng(seed, 'twister');
            end
            
            % Check if Econometrics Toolbox arima/simulate is available
            hasArima = license('test', 'Econometrics_Toolbox') && exist('arima', 'class') == 8;
            
            if hasArima
                Mdl = arima('Constant', 0, 'AR', {phi}, 'MA', {theta}, 'Variance', sigma2);
                y = simulate(Mdl, T);
            else
                % Built-in MATLAB filter engine fallback
                sigma = sqrt(sigma2);
                e = sigma * randn(T, 1);
                b = [1, theta];
                a = [1, -phi];
                y = filter(b, a, e);
            end
        end
    end
end
