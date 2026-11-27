classdef SimilarityFactory
    % SIMILARITYFACTORY  Create ISimilarity objects by name.
    %
    %   Centralizes similarity-strategy construction so callers never
    %   hard-code class names. Follows the Factory pattern (mirrors
    %   OceanDL.Activation.ActivationFactory).
    %
    %   Supported names (case-insensitive):
    %     'cosine' — CosineSimilarity
    %     'dot'    — DotProductSimilarity
    %
    %   Usage:
    %     sim = OceanStats.Similarity.SimilarityFactory.create('cosine');
    %     s   = sim.forward(A, B);

    methods (Static)
        function sim = create(name)
            % CREATE  Return an ISimilarity object for the given name.
            %
            %   sim = SimilarityFactory.create('cosine')
            import OceanStats.Similarity.*

            switch lower(strtrim(name))
                case 'cosine'
                    sim = CosineSimilarity();
                case 'dot'
                    sim = DotProductSimilarity();
                otherwise
                    error('OceanStats:Similarity:SimilarityFactory:Unknown', ...
                        'Unknown similarity "%s". Valid: cosine, dot.', name);
            end
        end

        function names = validNames()
            % VALIDNAMES  Return cell of all recognised similarity name strings.
            names = {'cosine', 'dot'};
        end
    end
end
