classdef WeightingFactory
    % WEIGHTINGFACTORY  Create IWeighting objects by name.
    %
    %   Centralizes weighting-strategy construction so callers never
    %   hard-code class names. Follows the Factory pattern (mirrors
    %   OceanDL.Activation.ActivationFactory,
    %   OceanStats.Similarity.SimilarityFactory).
    %
    %   Supported names (case-insensitive):
    %     'tfidf' — TfidfWeighting
    %     'bm25'  — BM25Weighting
    %
    %   Usage:
    %     w = OceanStats.Data.Weighting.WeightingFactory.create('bm25');
    %     w.fit(counts);
    %     W = w.forward(counts);
    %
    %   See also: OceanStats.Data.Weighting.IWeighting,
    %             OceanStats.Data.Weighting.TfidfWeighting,
    %             OceanStats.Data.Weighting.BM25Weighting

    methods (Static)
        function w = create(name)
            % CREATE  Return an IWeighting object for the given name.
            %
            %   w = WeightingFactory.create('tfidf')
            import OceanStats.Data.Weighting.*

            switch lower(strtrim(name))
                case 'tfidf'
                    w = TfidfWeighting();
                case 'bm25'
                    w = BM25Weighting();
                otherwise
                    error('OceanStats:Data:Weighting:WeightingFactory:Unknown', ...
                        'Unknown weighting "%s". Valid: tfidf, bm25.', name);
            end
        end

        function names = validNames()
            % VALIDNAMES  Return cell of all recognised weighting name strings.
            names = {'tfidf', 'bm25'};
        end
    end
end
