classdef InterpolationError
    % INTERPOLATIONERROR  Value object for scattered interpolation error metrics.
    %
    %   Computes RMSE, MAE, MaxErr, and R^2 between predictions and ground truth
    %   evaluated at a regular grid or arbitrary test points.
    %
    %   Usage:
    %     err = OceanStats.Interpolation.InterpolationError.fromFunc( ...
    %               model, f_true, 'Grid', true, 'NGrid', 100, 'Dim', 2, ...
    %               'DomainLow', 0, 'DomainHigh', 1);
    %
    %     err = OceanStats.Interpolation.InterpolationError.fromPoints( ...
    %               model, X_test, Y_test);
    %
    %   Properties:
    %     RMSE   - Root Mean Square Error
    %     MAE    - Mean Absolute Error
    %     MaxErr - Maximum absolute error
    %     R2     - Coefficient of determination (1 = perfect, 0 = mean-baseline)
    %     N      - Number of evaluation points
    %     Xeval  - Evaluation coordinates (M x D)
    %     Yhat   - Model predictions (M x 1)
    %     Ytrue  - Ground truth values (M x 1)
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    properties (SetAccess = private)
        RMSE   (1,1) double
        MAE    (1,1) double
        MaxErr (1,1) double
        R2     (1,1) double
        N      (1,1) double
        Xeval  double   % (M x D)
        Yhat   double   % (M x 1)
        Ytrue  double   % (M x 1)
    end

    methods (Static)

        function obj = fromFunc(model, f_true, opts)
            % FROMFUNC  Evaluate model against a ground-truth function handle.
            %
            %   obj = InterpolationError.fromFunc(model, f_true, Name, Value)
            %
            %   Options:
            %     Grid       (logical)  true = regular grid, false = random points  [true]
            %     NGrid      (integer)  points per dimension when Grid=true         [100]
            %     NRand      (integer)  random test points when Grid=false          [500]
            %     Dim        (integer)  input dimensionality                        [1]
            %     DomainLow  (double)   lower domain bound                          [0]
            %     DomainHigh (double)   upper domain bound                          [1]

            arguments
                model   OceanMath.Interpolation.IScatteredInterpolant
                f_true  function_handle
                opts.Grid       (1,1) logical = true
                opts.NGrid      (1,1) double  = 100
                opts.NRand      (1,1) double  = 500
                opts.Dim        (1,1) double  = 1
                opts.DomainLow  (1,1) double  = 0
                opts.DomainHigh (1,1) double  = 1
            end

            L = opts.DomainHigh - opts.DomainLow;

            if opts.Grid
                if opts.Dim == 1
                    Xeval = linspace(opts.DomainLow, opts.DomainHigh, opts.NGrid)';
                else
                    % 2D grid — NGrid^2 points
                    t = linspace(opts.DomainLow, opts.DomainHigh, opts.NGrid);
                    [Gx, Gy] = meshgrid(t, t);
                    Xeval = [Gx(:), Gy(:)];
                end
            else
                Xeval = opts.DomainLow + L * rand(opts.NRand, opts.Dim);
            end

            Ytrue = f_true(Xeval);
            Yhat  = model.predict(Xeval);

            obj = OceanStats.Interpolation.InterpolationError( ...
                      Xeval, Yhat, Ytrue);
        end

        function obj = fromPoints(model, X_test, Y_test)
            % FROMPOINTS  Evaluate model at given test points with known labels.
            %
            %   obj = InterpolationError.fromPoints(model, X_test, Y_test)

            arguments
                model   OceanMath.Interpolation.IScatteredInterpolant
                X_test  (:,:) double
                Y_test  (:,1) double
            end

            Yhat = model.predict(X_test);
            obj  = OceanStats.Interpolation.InterpolationError( ...
                       X_test, Yhat, Y_test);
        end

    end

    methods

        function obj = InterpolationError(Xeval, Yhat, Ytrue)
            % Private constructor — use static factories.
            arguments
                Xeval (:,:) double
                Yhat  (:,1) double
                Ytrue (:,1) double
            end

            resid = Yhat - Ytrue;
            ss_res = sum(resid.^2);
            ss_tot = sum((Ytrue - mean(Ytrue)).^2);

            obj.Xeval  = Xeval;
            obj.Yhat   = Yhat;
            obj.Ytrue  = Ytrue;
            obj.N      = numel(Ytrue);
            obj.RMSE   = sqrt(mean(resid.^2));
            obj.MAE    = mean(abs(resid));
            obj.MaxErr = max(abs(resid));
            obj.R2     = 1 - ss_res / (ss_tot + eps);
        end

        function printSummary(obj, label)
            % PRINTSUMMARY  Print a formatted error summary table.
            arguments
                obj
                label (1,1) string = "InterpolationError"
            end
            fprintf('\n── %s (N=%d eval pts) ──\n', label, obj.N);
            fprintf('  RMSE    : %.6f\n', obj.RMSE);
            fprintf('  MAE     : %.6f\n', obj.MAE);
            fprintf('  MaxErr  : %.6f\n', obj.MaxErr);
            fprintf('  R^2     : %.6f\n', obj.R2);
        end

    end
end
