classdef StageRegistry
    % STAGEREGISTRY  The Speech Pipeline Studio stages, in stepper order. Adding a stage =
    %   writing one IStage class and listing it here (or register() at run time).

    methods (Static)
        function stages = create()
            % CREATE  New instances of every stage (cell array of IStage).
            S = 'OceanViz.App.NLP.SpeechStudio.Stages.';
            names = {'InputStage', 'FramesStage', 'SpectrumStage', 'MelStage', 'FeaturesStage', ...
                'EncoderStage', 'ProbabilitiesStage', 'DecodingStage', 'TranscriptStage'};
            ctors = [cellfun(@(n) str2func([S n]), names, 'UniformOutput', false), ...
                OceanViz.App.NLP.SpeechStudio.Stages.StageRegistry.extra()];
            stages = cellfun(@(c) c(), ctors, 'UniformOutput', false);
        end

        function register(ctor)
            OceanViz.App.NLP.SpeechStudio.Stages.StageRegistry.extra(ctor);
        end
    end

    methods (Static, Access = private)
        function list = extra(add)
            persistent registered
            if isempty(registered), registered = {}; end
            if nargin > 0, registered{end+1} = add; end
            list = registered;
        end
    end
end
