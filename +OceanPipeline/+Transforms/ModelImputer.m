classdef ModelImputer < OceanPipeline.Transforms.IDataTransform
    % MODELIMPUTER  Adapter: wraps OceanStats.Data.PredictiveImputer as a
    %               standard IDataTransform for use in the Pipeline layer.
    %
    %   This class contains NO algorithm. It is a thin bridge that:
    %     1. Implements the IDataTransform fit/apply contract
    %     2. Delegates all computation to OceanStats.Data.PredictiveImputer
    %
    %   DEPENDENCY DIRECTION:
    %     +OceanPipeline --> +OceanStats    (one way only)
    %     +OceanStats knows nothing about Pipeline
    %
    %   USAGE:
    %     imp   = OceanPipeline.Transforms.ModelImputer('ModelType','CART');
    %     state = imp.fit(X_train);
    %     X_train_clean = imp.apply(X_train, state);
    %     X_test_clean  = imp.apply(X_test,  state);   % uses train models only
    %
    %   Constructor options are forwarded to PredictiveImputer.
    %   See also: OceanStats.Data.PredictiveImputer
    %             OceanPipeline.Transforms.IDataTransform

    properties (SetAccess = private)
        Core   OceanStats.Data.PredictiveImputer   % the real implementation
    end

    methods
        function obj = ModelImputer(opts)
            arguments
                opts.ModelType  (1,1) string {mustBeMember(opts.ModelType,{'OLS','CART'})} = "OLS"
                opts.Stochastic (1,1) logical = true
                opts.MaxIters   (1,1) double  = 10
                opts.K          (1,1) double  = 5
                opts.Tol        (1,1) double  = 1e-4
            end
            % Forward all options to the Stats-layer implementation
            obj.Core = OceanStats.Data.PredictiveImputer( ...
                'ModelType',  opts.ModelType,  ...
                'Stochastic', opts.Stochastic, ...
                'MaxIters',   opts.MaxIters,   ...
                'K',          opts.K,          ...
                'Tol',        opts.Tol);
        end

        % -------------------------------------------------------------- %
        function state = fit(obj, data)
            % FIT  Delegate to PredictiveImputer.fit (MICE + PMM).
            state = obj.Core.fit(data);
        end

        % -------------------------------------------------------------- %
        function output = apply(obj, data, state)
            % APPLY  Delegate to PredictiveImputer.apply.
            output = obj.Core.apply(data, state);
        end

        % -------------------------------------------------------------- %
        function describe(obj)
            fprintf('[ModelImputer] type=%s  stochastic=%d  K=%d  maxIters=%d\n', ...
                obj.Core.ModelType, obj.Core.Stochastic, ...
                obj.Core.K, obj.Core.MaxIters);
        end
    end
end
