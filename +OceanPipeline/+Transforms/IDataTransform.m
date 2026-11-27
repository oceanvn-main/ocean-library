classdef IDataTransform < handle
    % IDATATRANSFORM  Abstract interface for all data quality transforms.
    %
    %   ANTI-LEAKAGE PROTOCOL (mandatory):
    %     state = transform.fit(data_train)      % learn ONLY from training data
    %     clean = transform.apply(data_any, state) % apply to train, test, prod
    %
    %   The state struct carries everything learned during fit().
    %   apply() is a pure function of (data, state) — it never mutates state.
    %
    %   CHAINING:
    %     Transforms can be chained in a cell array and applied sequentially.
    %     See OceanPipeline.Orchestration.DataPipeline (DA-6).
    %
    %   USAGE:
    %     imp   = OceanPipeline.Transforms.MissingValueImputer('Strategy','mean');
    %     state = imp.fit(X_train);
    %     X_clean_train = imp.apply(X_train, state);
    %     X_clean_test  = imp.apply(X_test,  state);  % same state — no leakage
    %
    %   See also: OceanPipeline.Transforms.MissingValueImputer
    %             OceanPipeline.Transforms.OutlierClipper
    %             OceanPipeline.Transforms.DataValidator
    %             OceanPipeline.Transforms.DataMasker

    methods (Abstract)
        state  = fit(obj, data)
        %   FIT  Learn transform parameters from training data.
        %   data  — (N × p) double matrix
        %   state — struct capturing learned parameters (transform-specific)

        output = apply(obj, data, state)
        %   APPLY  Transform data using pre-learned state.
        %   data   — (N × p) double matrix
        %   state  — struct returned by fit()
        %   output — (N' × p') double matrix (shape may change for 'drop')
    end

    methods
        function describe(obj)
            % DESCRIBE  Print a human-readable summary of this transform.
            fprintf('[%s]\n', class(obj));
        end

        function [out_train, out_test, state] = fitApply(obj, train, test)
            % FITAPPLY  Convenience: fit on train, apply on both sets.
            %   Guaranteed no leakage — fit() only sees train.
            state     = obj.fit(train);
            out_train = obj.apply(train, state);
            out_test  = obj.apply(test,  state);
        end
    end
end
