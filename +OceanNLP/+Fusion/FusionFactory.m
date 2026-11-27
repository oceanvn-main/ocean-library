classdef FusionFactory
    % FUSIONFACTORY  Factory class for instantiating rank fusion strategies.
    %
    % PURPOSE:
    %   Constructs rank fusion strategy instances (RRF, Borda) given a strategy name string.
    %
    % SUPPORTED STRATEGIES:
    %   'rrf'   - ReciprocalRankFusion (options: K)
    %   'borda' - BordaCountFusion
    %
    % EXAMPLES:
    %   fusion = OceanNLP.Fusion.FusionFactory.create('rrf', 'K', 60);
    %   fusion = OceanNLP.Fusion.FusionFactory.create('borda');
    %
    % See also: OceanNLP.Fusion.IFusionStrategy, OceanNLP.Fusion.ReciprocalRankFusion, OceanNLP.Fusion.BordaCountFusion

    methods (Static)
        function f = create(name, options)
            arguments
                name (1,1) string
                options.K (1,1) double {mustBeNonnegative} = 60
            end

            switch lower(name)
                case "rrf"
                    f = OceanNLP.Fusion.ReciprocalRankFusion('K', options.K);
                case "borda"
                    f = OceanNLP.Fusion.BordaCountFusion();
                otherwise
                    error('OceanNLP:Fusion:UnknownStrategy', ...
                        'Unknown fusion strategy "%s". Valid choices: "rrf", "borda".', name);
            end
        end
    end
end
