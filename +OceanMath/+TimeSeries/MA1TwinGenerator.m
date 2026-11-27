classdef MA1TwinGenerator < handle
    % MA1TwinGenerator Class for Invertible and Non-Invertible MA(1) Processes
    %
    % This class manages the parameters, simulation, and moment validation
    % for observationally equivalent MA(1) process twins:
    %
    %   Y_t = eps_t + theta * eps_{t-1},   eps_t ~ i.i.d. N(0, sigma^2)
    %
    % Mathematical Background:
    % ------------------------
    % Any MA(1) process has an observationally equivalent representation obtained
    % by reflecting the root of the MA polynomial (1 + theta * z = 0) through 
    % the unit circle. The parameters of the twin representation are related by:
    %
    %   theta_tilde  = 1 / theta
    %   sigma_tilde^2 = sigma^2 * theta^2
    %
    % Only the representation with |theta| < 1 is invertible, allowing the
    % underlying shocks (eps_t) to be recovered from the history of Y_t via
    % a stable AR(inf) filter.
    
    properties (SetAccess = private)
        % Invertible twin parameters (|theta_inv| < 1)
        theta_inv  (1,1) double
        sigma2_inv (1,1) double
        
        % Non-invertible twin parameters (|theta_non| > 1)
        theta_non  (1,1) double
        sigma2_non (1,1) double
        
        % Theoretical moments shared by both processes
        theory_var  (1,1) double  % Var(Y_t) = sigma^2 * (1 + theta^2)
        theory_cov1 (1,1) double  % Cov(Y_t, Y_{t-1}) = sigma^2 * theta
        theory_cov2 (1,1) double  % Cov(Y_t, Y_{t-k}) = 0 (k >= 2)
    end
    
    methods
        function obj = MA1TwinGenerator(theta, sigma2)
            % Constructor: Instantiates the twins from a single parameter set.
            %
            % Usage:
            %   generator = MA1TwinGenerator(theta, sigma2);
            %
            % The class automatically identifies if the input is invertible
            % or non-invertible, and calculates the companion twin.
            
            if abs(theta) == 1
                error('Boundary case |theta| = 1 is non-invertible and has no unique root-flipped twin.');
            end
            
            if abs(theta) < 1
                % Input is invertible
                obj.theta_inv  = theta;
                obj.sigma2_inv = sigma2;
                
                % Compute non-invertible twin
                obj.theta_non  = 1.0 / theta;
                obj.sigma2_non = sigma2 * (theta^2);
            else
                % Input is non-invertible
                obj.theta_non  = theta;
                obj.sigma2_non = sigma2;
                
                % Compute invertible twin
                obj.theta_inv  = 1.0 / theta;
                obj.sigma2_inv = sigma2 * (theta^2);
            end
            
            % Compute shared theoretical moments using invertible twin values
            obj.theory_var  = obj.sigma2_inv * (1 + obj.theta_inv^2);
            obj.theory_cov1 = obj.sigma2_inv * obj.theta_inv;
            obj.theory_cov2 = 0.0;
        end
        
        function [Y, eps] = simulateInvertible(obj, T, seed)
            % Simulates a path of length T for the Invertible MA(1) process.
            %
            % Outputs:
            %   Y   : Simulated output sequence (T x 1)
            %   eps : Underlying i.i.d. noise sequence (T x 1)
            
            if nargin >= 3, rng(seed); end
            
            sigma = sqrt(obj.sigma2_inv);
            eps = randn(T, 1) * sigma;
            Y = zeros(T, 1);
            
            for t = 2:T
                Y(t) = eps(t) + obj.theta_inv * eps(t-1);
            end
        end
        
        function [Y, eps] = simulateNonInvertible(obj, T, seed)
            % Simulates a path of length T for the Non-Invertible MA(1) process.
            %
            % Outputs:
            %   Y   : Simulated output sequence (T x 1)
            %   eps : Underlying i.i.d. noise sequence (T x 1)
            
            if nargin >= 3, rng(seed); end
            
            sigma = sqrt(obj.sigma2_non);
            eps = randn(T, 1) * sigma;
            Y = zeros(T, 1);
            
            for t = 2:T
                Y(t) = eps(t) + obj.theta_non * eps(t-1);
            end
        end
        
        function acov = computeSampleAutocovariance(~, Y, max_lag)
            % Utility function to compute the sample autocovariance of a series Y.
            %
            % Inputs:
            %   Y       : Column vector of observations (T x 1)
            %   max_lag : Maximum lag to evaluate (scalar)
            % Outputs:
            %   acov    : Vector containing autocovariances from lag 0 to max_lag
            
            T = length(Y);
            Y_centered = Y - mean(Y);
            acov = zeros(max_lag + 1, 1);
            
            for k = 0:max_lag
                acov(k+1) = Y_centered(1:T-k)' * Y_centered(k+1:T) / T;
            end
        end
    end
end
