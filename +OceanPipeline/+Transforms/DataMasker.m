classdef DataMasker < OceanPipeline.Transforms.IDataTransform
    % DATAMASKER  Anonymize or suppress columns using masking operations.
    %
    %   THREE OPERATIONS (can be combined):
    %     'drop'  — Remove named columns entirely
    %     'noise' — Add Gaussian noise N(0, NoiseStd) to named columns
    %     'hash'  — SHA-256 hash a column's values (for PII — not applicable to
    %               numeric matrices; reserved for future string support)
    %
    %   USAGE:
    %     % Drop columns 1 and 3 (by index)
    %     mask  = OceanPipeline.Transforms.DataMasker('Drop', [1, 3]);
    %     state = mask.fit(X_train);
    %     X_clean = mask.apply(X_train, state);
    %
    %     % Add noise to column 2
    %     mask  = OceanPipeline.Transforms.DataMasker('Noise', [2], 'NoiseStd', 0.1);
    %     state = mask.fit(X_train);
    %     X_noisy = mask.apply(X_train, state);
    %
    %     % Both: drop col 1, noise col 2
    %     mask  = OceanPipeline.Transforms.DataMasker('Drop',[1],'Noise',[2],'NoiseStd',0.05);
    %
    %   See also: OceanPipeline.Transforms.IDataTransform

    properties (SetAccess = private)
        DropCols  double   % column indices to drop
        NoiseCols double   % column indices to add noise to
        NoiseStd  (1,1) double = 0.01
    end

    methods
        function obj = DataMasker(opts)
            arguments
                opts.Drop     double = []
                opts.Noise    double = []
                opts.NoiseStd (1,1) double = 0.01
            end
            obj.DropCols = opts.Drop;
            obj.NoiseCols = opts.Noise;
            obj.NoiseStd = opts.NoiseStd;
        end

        % -------------------------------------------------------------- %
        function state = fit(obj, data)
            % FIT  Record the learned masking plan.
            % state.nCols    — p (for apply validation)
            % state.dropCols — column indices to remove
            % state.noiseCols— column indices to perturb
            % state.noiseStd — std of Gaussian noise
            % state.keepCols — indices of columns that survive after drop
            arguments
                obj
                data double
            end
            p = size(data, 2);

            % Validate indices
            if any(obj.DropCols > p) || any(obj.DropCols < 1)
                error('OceanPipeline:Transforms:DataMasker:BadIndex', ...
                    'Drop indices must be in [1, %d].', p);
            end
            if any(obj.NoiseCols > p) || any(obj.NoiseCols < 1)
                error('OceanPipeline:Transforms:DataMasker:BadIndex', ...
                    'Noise indices must be in [1, %d].', p);
            end

            state.nCols    = p;
            state.dropCols = obj.DropCols(:)';
            state.noiseCols= obj.NoiseCols(:)';
            state.noiseStd = obj.NoiseStd;
            state.keepCols = setdiff(1:p, state.dropCols);
        end

        % -------------------------------------------------------------- %
        function output = apply(~, data, state)
            % APPLY  Mask data using pre-learned plan from fit().
            arguments
                ~
                data  double
                state struct
            end
            if size(data, 2) ~= state.nCols
                error('OceanPipeline:Transforms:DataMasker:DimMismatch', ...
                    'Expected %d columns, got %d.', state.nCols, size(data,2));
            end

            output = data;

            % Step 1: Add noise (before drop, so indices still valid)
            if ~isempty(state.noiseCols)
                noise = state.noiseStd * randn(size(output,1), numel(state.noiseCols));
                output(:, state.noiseCols) = output(:, state.noiseCols) + noise;
            end

            % Step 2: Drop columns
            if ~isempty(state.dropCols)
                output = output(:, state.keepCols);
            end
        end

        % -------------------------------------------------------------- %
        function describe(obj)
            fprintf('[DataMasker] drop=%s  noise=%s  noiseStd=%.4g\n', ...
                mat2str(obj.DropCols), mat2str(obj.NoiseCols), obj.NoiseStd);
        end
    end
end
