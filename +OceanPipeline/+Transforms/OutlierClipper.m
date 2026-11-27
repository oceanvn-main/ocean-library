classdef OutlierClipper < OceanPipeline.Transforms.IDataTransform
    % OUTLIERCLIPPEER  Clip extreme values to learned training-data bounds.
    %
    %   MODES:
    %     'sigma' — bounds = mean ± N * std   (default, N=3)
    %     'iqr'   — bounds = [Q1-1.5*IQR, Q3+1.5*IQR]
    %
    %   Values outside bounds are clipped (not dropped).
    %   Bounds are learned from training data only (anti-leakage).
    %
    %   USAGE:
    %     clip  = OceanPipeline.Transforms.OutlierClipper('Mode','sigma','N',3);
    %     state = clip.fit(X_train);
    %     X_train_clean = clip.apply(X_train, state);
    %     X_test_clean  = clip.apply(X_test,  state);
    %
    %   See also: OceanPipeline.Transforms.IDataTransform

    properties (SetAccess = private)
        Mode (1,1) string  = "sigma"
        N    (1,1) double  = 3        % sigma multiplier (sigma mode only)
    end

    methods
        function obj = OutlierClipper(opts)
            arguments
                opts.Mode (1,1) string {mustBeMember(opts.Mode,{'sigma','iqr'})} = "sigma"
                opts.N    (1,1) double = 3
            end
            obj.Mode = opts.Mode;
            obj.N    = opts.N;
        end

        % -------------------------------------------------------------- %
        function state = fit(obj, data)
            % FIT  Compute per-column clipping bounds from training data.
            % state.lower — (1 × p) lower bounds
            % state.upper — (1 × p) upper bounds
            % state.nCols — p
            arguments
                obj
                data double
            end
            p          = size(data, 2);
            state.nCols = p;
            state.mode  = char(obj.Mode);

            switch obj.Mode
                %Notice the mean ($\mu$) and standard deviation ($\sigma$) are themselves highly sensitive to outliers
                case 'sigma'
                    mu          = mean(data, 1, 'omitnan');
                    sg          = std(data,  0, 1, 'omitnan');
                    state.lower = mu - obj.N * sg;
                    state.upper = mu + obj.N * sg;

                case 'iqr'
                    %Using percentiles to avoid the influence of outliers
                    Q1          = prctile(data, 25, 1);
                    Q3          = prctile(data, 75, 1);
                    IQR_        = Q3 - Q1;
                    state.lower = Q1 - 1.5 * IQR_;
                    state.upper = Q3 + 1.5 * IQR_;
            end
        end

        % -------------------------------------------------------------- %
        function output = apply(~, data, state)
            % APPLY  Clip values to [lower, upper] learned during fit().
            arguments
                ~
                data  double
                state struct
            end
            if size(data, 2) ~= state.nCols
                error('OceanPipeline:Transforms:OutlierClipper:DimMismatch', ...
                    'Expected %d columns, got %d.', state.nCols, size(data,2));
            end
            output = max(data, state.lower);   % clip from below
            output = min(output, state.upper); % clip from above
        end

        % -------------------------------------------------------------- %
        function describe(obj)
            fprintf('[OutlierClipper] mode="%s"  N=%.1f\n', obj.Mode, obj.N);
        end
    end
end
