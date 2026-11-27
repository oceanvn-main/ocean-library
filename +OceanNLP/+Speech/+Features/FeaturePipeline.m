classdef FeaturePipeline < OceanNLP.Speech.Core.IFeatureTransform
    % FEATUREPIPELINE  Ordered composition of feature transforms (itself a transform).
    %
    %   p  = OceanNLP.Speech.Features.FeaturePipeline({LogMel(8000), DeltaFeatures()})
    %   fs = p.forward(waveform)     % t_K( ... t_2( t_1(waveform) ) )
    %   cfg = p.config()             % struct: Type "FeaturePipeline", Stages (1 x K struct
    %                                %   array, fields Index, Type, Config) in application order
    %   d  = p.outputDim(inputDim)   % t_K.outputDim( ... t_1.outputDim(inputDim))
    %
    %   Stage 1 receives the Waveform; every later stage receives the previous
    %   stage's FeatureSequence (each stage enforces its own input type). The
    %   result keeps the last stage's values, axes, units and FrequencyHz, and
    %   its Config is the pipeline config (which embeds every stage config).

    properties (SetAccess = immutable)
        Stages (1,:) cell
    end

    methods
        function obj = FeaturePipeline(stages)
            arguments
                stages (1,:) cell {mustBeNonempty}
            end
            for k = 1:numel(stages)
                if ~isa(stages{k}, 'OceanNLP.Speech.Core.IFeatureTransform')
                    error('OceanNLP:Speech:Features:FeaturePipeline:StageType', ...
                        'Stage %d is a %s, not an OceanNLP.Speech.Core.IFeatureTransform.', k, class(stages{k}));
                end
            end
            obj.Stages = stages;
        end

        function fs = forward(obj, input)
            arguments
                obj
                input
            end
            if ~isa(input, 'OceanNLP.Speech.Core.Waveform')
                error('OceanNLP:Speech:Features:FeaturePipeline:InputType', ...
                    'FeaturePipeline.forward expects an OceanNLP.Speech.Core.Waveform, got %s.', class(input));
            end
            cur = input;
            history = input.History;
            for k = 1:numel(obj.Stages)
                % Runtime type progression is intentional: stage 1 normally maps
                % Waveform -> FeatureSequence; later stages map FeatureSequence ->
                % FeatureSequence.  Each stage owns its input-type validation.
                cur = obj.Stages{k}.forward(cur);
                if k == 1 && isfield(cur.Config, 'InputHistory')
                    history = cur.Config.InputHistory;   % e.g. records a mono mixdown
                end
            end
            cfg = obj.config();
            cfg.InputHistory = history;
            % Preserve the last stage's data semantics, but replace its top-level
            % Config with the complete ordered pipeline so a checkpoint can rebuild
            % the whole chain rather than only the final transform.
            fs = OceanNLP.Speech.Core.FeatureSequence(cur.Values, 'FrameTimes', cur.FrameTimes, ...
                'FrequencyHz', cur.FrequencyHz, 'AxisNames', cur.AxisNames, 'Units', cur.Units, ...
                'Config', cfg, 'Source', cur.Source);
        end

        function cfg = config(obj)
            K = numel(obj.Stages);
            st = struct('Index', cell(1, K), 'Type', cell(1, K), 'Config', cell(1, K));
            for k = 1:K
                c = obj.Stages{k}.config();
                st(k).Index = k;
                st(k).Type = string(c.Type);
                st(k).Config = c;
            end
            cfg = struct('Type', "FeaturePipeline", 'Stages', st);
        end

        function d = outputDim(obj, inputDim)
            arguments
                obj
                inputDim double = []
            end
            d = inputDim;
            for k = 1:numel(obj.Stages)
                % Dimension planning mirrors forward() without processing audio.
                % Example: LogMel []->40, then delta order 2 maps 40->120.
                d = obj.Stages{k}.outputDim(d);
            end
        end
    end

    methods (Static)
        function f = fromConfig(cfg, opts)
            % FROMCONFIG  Inverse of config(). Each stage is rebuilt by
            %   'StageBuilder' (default: OceanNLP.Speech.Features.<Type>.fromConfig);
            %   Models.RecognizerFactory passes its own builder so runtime-registered
            %   feature types work inside pipelines too.
            arguments
                cfg (1,1) struct
                opts.StageBuilder = []
            end
            st = cfg.Stages;
            if iscell(st), st = [st{:}]; end
            stages = cell(1, numel(st));
            for k = 1:numel(st)
                c = st(k).Config;
                if isempty(opts.StageBuilder)
                    stages{k} = feval("OceanNLP.Speech.Features." + string(c.Type) + ".fromConfig", c);
                else
                    stages{k} = opts.StageBuilder(c);
                end
            end
            f = OceanNLP.Speech.Features.FeaturePipeline(stages);
        end
    end
end
