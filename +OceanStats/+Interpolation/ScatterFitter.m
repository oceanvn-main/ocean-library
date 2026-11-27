classdef ScatterFitter < handle
    % SCATTERFITTER  Orchestrator for scattered data fitting experiments.
    %
    %   Manages the full workflow:
    %     1. Accepts training data (X, y) and ground-truth function
    %     2. Fits one or more IScatteredInterpolant models
    %     3. Computes InterpolationError for each model
    %     4. Stores results for downstream comparison and visualization
    %
    %   Usage:
    %     fitter = OceanStats.Interpolation.ScatterFitter( ...
    %                  'DomainLow', 0, 'DomainHigh', 1, 'Dim', 1);
    %
    %     fitter.addModel('rbf-tps',  InterpolationFactory.create('rbf'));
    %     fitter.addModel('knn-5',    InterpolationFactory.create('knn','K',5));
    %     fitter.addModel('natural',  InterpolationFactory.create('natural'));
    %
    %     fitter.fitAll(X_train, Y_train);
    %     fitter.evaluateAll(gen.MetaData.f_true);
    %     fitter.printSummary();
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-18

    properties
        DomainLow  (1,1) double = 0.0
        DomainHigh (1,1) double = 1.0
        Dim        (1,1) double {mustBeMember(Dim, [1,2])} = 1
        NGrid      (1,1) double {mustBePositive} = 100  % eval grid density
    end

    properties (SetAccess = private)
        Models_  containers.Map   % name → IScatteredInterpolant
        Errors_  containers.Map   % name → InterpolationError
        Names_   cell             % ordered list of model names
    end

    methods
        function obj = ScatterFitter(opts)
            % SCATTERFITTER Constructor.
            arguments
                opts.DomainLow  (1,1) double = 0.0
                opts.DomainHigh (1,1) double = 1.0
                opts.Dim        (1,1) double = 1
                opts.NGrid      (1,1) double = 100
            end
            obj.DomainLow  = opts.DomainLow;
            obj.DomainHigh = opts.DomainHigh;
            obj.Dim        = opts.Dim;
            obj.NGrid      = opts.NGrid;
            obj.Models_    = containers.Map();
            obj.Errors_    = containers.Map();
            obj.Names_     = {};
        end

        function addModel(obj, name, model)
            % ADDMODEL  Register an IScatteredInterpolant with a label.
            %
            %   fitter.addModel('rbf-tps', InterpolationFactory.create('rbf'))

            arguments
                obj
                name  (1,1) string
                model OceanMath.Interpolation.IScatteredInterpolant
            end
            obj.Models_(char(name)) = model;
            if ~ismember(char(name), obj.Names_)
                obj.Names_{end+1} = char(name);
            end
        end

        function fitAll(obj, X_train, Y_train)
            % FITALL  Fit every registered model on (X_train, Y_train).
            arguments
                obj
                X_train (:,:) double
                Y_train (:,1) double
            end
            for i = 1:numel(obj.Names_)
                nm = obj.Names_{i};
                model = obj.Models_(nm);
                model.fit(X_train, Y_train);
            end
        end

        function evaluateAll(obj, f_true, opts)
            % EVALUATEALL  Compute InterpolationError for each model against
            %   a ground-truth function handle (regular grid evaluation).
            %
            %   fitter.evaluateAll(f_true)
            %   fitter.evaluateAll(f_true, 'Grid', false, 'NRand', 1000)

            arguments
                obj
                f_true function_handle
                opts.Grid  (1,1) logical = true
                opts.NRand (1,1) double  = 500
            end

            import OceanStats.Interpolation.InterpolationError

            for i = 1:numel(obj.Names_)
                nm  = obj.Names_{i};
                mdl = obj.Models_(nm);
                err = InterpolationError.fromFunc(mdl, f_true, ...
                    'Grid',       opts.Grid, ...
                    'NGrid',      obj.NGrid, ...
                    'NRand',      opts.NRand, ...
                    'Dim',        obj.Dim, ...
                    'DomainLow',  obj.DomainLow, ...
                    'DomainHigh', obj.DomainHigh);
                obj.Errors_(nm) = err;
            end
        end

        function evaluateAllOnPoints(obj, X_test, Y_test)
            % EVALUATEALLLONPOINTS  Evaluate on provided test points.

            arguments
                obj
                X_test (:,:) double
                Y_test (:,1) double
            end

            import OceanStats.Interpolation.InterpolationError

            for i = 1:numel(obj.Names_)
                nm  = obj.Names_{i};
                mdl = obj.Models_(nm);
                err = InterpolationError.fromPoints(mdl, X_test, Y_test);
                obj.Errors_(nm) = err;
            end
        end

        function printSummary(obj)
            % PRINTSUMMARY  Print a comparison table for all models.
            fprintf('\n%s\n', repmat('─', 1, 55));
            fprintf('%-20s  %8s  %8s  %8s  %6s\n', ...
                'Model', 'RMSE', 'MAE', 'MaxErr', 'R^2');
            fprintf('%s\n', repmat('─', 1, 55));
            for i = 1:numel(obj.Names_)
                nm = obj.Names_{i};
                if obj.Errors_.isKey(nm)
                    e = obj.Errors_(nm);
                    fprintf('%-20s  %8.5f  %8.5f  %8.5f  %6.4f\n', ...
                        nm, e.RMSE, e.MAE, e.MaxErr, e.R2);
                else
                    fprintf('%-20s  (not evaluated)\n', nm);
                end
            end
            fprintf('%s\n', repmat('─', 1, 55));
        end

        function model = getModel(obj, name)
            % GETMODEL  Retrieve a fitted model by name.
            model = obj.Models_(char(name));
        end

        function err = getError(obj, name)
            % GETERROR  Retrieve the InterpolationError struct by model name.
            err = obj.Errors_(char(name));
        end

        function names = getNames(obj)
            names = obj.Names_;
        end
    end
end
