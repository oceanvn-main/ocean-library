classdef ARMALikelihood
    % ARMALIKELIHOOD Evaluates conditional log-likelihood terrain for ARMA(1,1).
    %
    % NAMESPACE: OceanStats.Time
    %
    % Computes exact Box-Jenkins conditional Gaussian log-likelihood using MATLAB
    % built-in digital filter engine filter([1, -phi], [1, theta], y):
    % e(t) = y(t) - phi * y(t-1) - theta * e(t-1)
    % L(phi, theta | y) = - (T-1)/2 * log(2*pi*sigma^2) - 1/(2*sigma^2) * sum(e(t)^2)
    %
    % Author: Ocean Library System
    % Date: 2026-09-03

    methods (Static)
        function L = logLikelihood(params, y, sigma2)
            % LOGLIKELIHOOD Computes exact conditional Gaussian log-likelihood.
            %
            % Args:
            %   params (double): [phi, theta] parameter vector.
            %   y (double): [T x 1] observed time series.
            %   sigma2 (double): Innovation variance (default 1.0).
            %
            % Returns:
            %   L (double): Scalar log-likelihood score.
            
            arguments
                params (1,2) double
                y (:,1) double
                sigma2 (1,1) double = 1.0
            end
            
            phi = params(1);
            theta = params(2);
            
            % Stationarity and invertibility bounds penalty
            if abs(phi) >= 0.999 || abs(theta) >= 0.999
                L = -1e10;
                return;
            end
            
            % MATLAB built-in digital filter engine for residual computation
            % Inverse filter: (1 - phi*B) y_t = (1 + theta*B) e_t  =>  e_t = filter([1, -phi], [1, theta], y)
            b = [1, -phi];
            a = [1, theta];
            e = filter(b, a, y);
            
            % Truncate first transient observation e(1)
            e_sub = e(2:end);
            N = length(e_sub);
            
            L = -0.5 * N * log(2 * pi * sigma2) - (1 / (2 * sigma2)) * sum(e_sub.^2);
        end
        
        function [PhiGrid, ThetaGrid, LGrid] = gridSweep(y, sigma2, phiRange, thetaRange, res)
            % GRIDSWEEP Sweeps 2D parameter bounds to construct topographical map.
            %
            % Args:
            %   y (double): Time series data.
            %   sigma2 (double): Innovation variance.
            %   phiRange (double): [min, max] range for phi.
            %   thetaRange (double): [min, max] range for theta.
            %   res (int): Grid resolution (default 100).
            %
            % Returns:
            %   PhiGrid (double): Meshgrid of phi values [res x res].
            %   ThetaGrid (double): Meshgrid of theta values [res x res].
            %   LGrid (double): Log-likelihood surface values [res x res].
            
            arguments
                y (:,1) double
                sigma2 (1,1) double = 1.0
                phiRange (1,2) double = [-0.9, 0.9]
                thetaRange (1,2) double = [-0.9, 0.9]
                res (1,1) double = 100
            end
            
            phi_vec = linspace(phiRange(1), phiRange(2), res);
            theta_vec = linspace(thetaRange(1), thetaRange(2), res);
            [PhiGrid, ThetaGrid] = meshgrid(phi_vec, theta_vec);
            
            LGrid = zeros(size(PhiGrid));
            for i = 1:numel(PhiGrid)
                p = [PhiGrid(i), ThetaGrid(i)];
                LGrid(i) = OceanStats.Time.ARMALikelihood.logLikelihood(p, y, sigma2);
            end
        end
    end
end
