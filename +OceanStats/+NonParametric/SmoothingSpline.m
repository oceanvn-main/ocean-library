classdef SmoothingSpline < OceanStats.Base.ProbabilisticModel
    % SMOOTHINGSPLINE A reusable Smoothing Spline model.
    %
    % This class encapsulates the logic for fitting and predicting with 
    % cubic B-splines using Ridge regression. It provides helper methods
    % explicitly designed to interface with OceanStats.Data.BootstrapSampler.
    %
    % == USAGE ==
    %   spline = OceanStats.NonParametric.SmoothingSpline(...
    %       'Knots', 15, 'Lambda', 0.1, 'Domain', [0, 1]);
    %   
    %   [params] = spline.fit(data);
    %   y_pred = spline.predict(x_new, params);
    %
    % Author: Ocean Library System
    % Date: 2026-02-11

    properties
        Knots   (1,1) double {mustBePositive, mustBeInteger} = 15 % Number of Internal Knots
        Degree  (1,1) double {mustBePositive, mustBeInteger} = 3  % Spline Degree (Cubic=3)
        Lambda  (1,1) double {mustBeNonnegative} = 0.1            % Regularization Strength
        Domain  (1,2) double = [0, 1]                             % Domain [min, max]
        
        % Stored Fitted Parameters (optional state)
        LastParams struct = struct()
    end

    % methods
    %     function obj = SmoothingSpline(options)
    %         % Constructor
    %         arguments
    %             options.Knots  = 15
    %             options.Degree = 3
    %             options.Lambda = 0.1
    %             options.Domain = [0, 1]
    %         end
    %         obj.Knots  = options.Knots;
    %         obj.Degree = options.Degree;
    %         obj.Lambda = options.Lambda;
    %         obj.Domain = options.Domain;
    %     end
    methods 
        function obj = SmoothingSpline(options)
            %Constructor
            arguments
                options.Knots = 15
                options.Degree = 3
                options.Lambda = 0.1
                options.Domain = [0, 1]

            end

        end

        function [params, H] = fit(obj, data)
            % FIT Fits the smoothing spline to data (X, Y).
            % 
            % Inputs:
            %   data - (N x 2) matrix. Col 1 = X, Col 2 = Y.
            %
            % Outputs:
            %   params - Struct with fields .beta and .sigma
            %   H      - Basis matrix (optional)
            
            x = data(:,1);
            y = data(:,2);

            % 1. Generate Basis Matrix H (N x p)
            % Each column corresponds to a B-spline basis function evaluation.
            import OceanStats.NonParametric.generateBSplineBasis;
            H = generateBSplineBasis(x, obj.Domain, obj.Knots, obj.Degree);

            % 2. Setup Ridge Regression
            % Normal Equations: (H'H + Lambda*Omega) * beta = H'y
            % Here we use standard Ridge (Omega = I). 
            % For P-Splines, Omega would be a difference penalty matrix (D'D).
            [~, p] = size(H);
            Gram = H' * H;          % (p x p)
            Omega = eye(p);         % Identity Penalty Matrix
            
            % Compute Precision Matrix (Inverse Covariance of Beta)
            Precision = Gram + obj.Lambda * Omega;
            
            % 3. Solve for Beta
            % beta = Precision \ (H' * y)
            beta = Precision \ (H' * y);

            % 4. Estimate Noise Variance (Sigma^2)
            % Based on Residual Sum of Squares (RSS)
            y_hat = H * beta;
            residuals = y - y_hat;
            
            % Degrees of Freedom correction
            % A simple approximation for p parameters is (N - p).
            % For penalized splines, effective DOF (trace(S)) is better, 
            % but (N-p) is sufficient for a basic variance estimator here.
            dof = max(1, length(y) - p); 
            sigma = sqrt(sum(residuals.^2) / dof);
            
            params = struct('beta', beta, 'sigma', sigma);
            obj.LastParams = params;
        end

        function y_pred = predict(obj, x_new, params)
            % PREDICT Evaluates the spline at x_new.
            % Inputs:
            %   x_new  - inputs
            %   params - (Optional) struct with .beta. Uses obj.LastParams if missed.
            
            if nargin < 3
                if isempty(fieldnames(obj.LastParams))
                    error('Model not fitted and no params provided.');
                end
                params = obj.LastParams;
            end
            
            import OceanStats.NonParametric.generateBSplineBasis;
            H_new = generateBSplineBasis(x_new, obj.Domain, obj.Knots, obj.Degree);
            y_pred = H_new * params.beta;
        end

        % ---------------------------------------------------------
        % ProbabilisticModel Interface / Bootstrap
        % ---------------------------------------------------------
        
        function data_new = fit_and_generate(obj, params, n)
             % Compatibility alias if needed
             % But ProbabilisticModel requires boot_gen or simulate?
             % Abstract definition was boot_gen(obj, params, n)
             error('Not implemented directly. Use boot_gen with context.');
        end

        function data_new = boot_gen(obj, params, n, x_context)
            % BOOT_GEN Generates synthetic data (Parametric Bootstrap).
            % 
            % Note on Fixed-X vs Random-X:
            % This model is conditional on X. To generate Y*, we need X*.
            % For Fixed-X/Parametric bootstrap, we typically reuse the original X.
            %
            % Procedure:
            % 1. Predict mean response E[y|x] using fitted params (beta).
            % 2. Add Gaussian noise with fitted std dev (sigma).
            %
            % Inputs:
            %   params    - fitted params (beta, sigma)
            %   n         - sample size (unused if x_context determines size)
            %   x_context - (Required) The X locations to generate at. 
            
            if nargin < 4
                 error('SmoothingSpline requires X context for generation (Fixed-X design). Pass x_context.');
            end
            
            % Generate Mean Response
            y_mean = obj.predict(x_context, params);
            
            % Add Noise
            epsilon = params.sigma * randn(size(x_context));
            y_new = y_mean + epsilon;
            
            data_new = [x_context, y_new];
        end
        
        % Legacy / Bootstrap Helper aliases
        function params = boot_fit(obj, data)
            params = obj.fit(data);
        end
        
        function stat = boot_stat(obj, data, x_grid)
            params = obj.fit(data);
            s = obj.predict(x_grid, params);
            stat = s(:)';
        end
    end
end
