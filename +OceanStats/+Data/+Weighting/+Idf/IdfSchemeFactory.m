classdef IdfSchemeFactory
    % IDFSCHEMEFACTORY  Create IIdfScheme objects by name.
    %
    %   Centralizes idf-scheme construction so callers (and stateDict
    %   round-trips — see TfidfWeighting/BM25Weighting.loadStateDict) never
    %   hard-code class names. Follows the Factory pattern (mirrors
    %   OceanDL.Activation.ActivationFactory,
    %   OceanStats.Similarity.SimilarityFactory,
    %   OceanStats.Data.Weighting.WeightingFactory).
    %
    %   Supported names (case-insensitive):
    %     'smooth'  — SmoothIdf  (scikit-learn / Lucene smooth_idf; TfidfWeighting default)
    %     'classic' — ClassicIdf (Sparck Jones 1972, un-smoothed textbook formula)
    %     'bm25'    — BM25Idf    (Elasticsearch/Lucene BM25 non-negative idf; BM25Weighting default)
    %
    %   Usage:
    %     scheme = OceanStats.Data.Weighting.Idf.IdfSchemeFactory.create('classic');
    %     tfidf  = OceanStats.Data.Weighting.TfidfWeighting('IdfScheme', scheme);
    %
    %   See also: OceanStats.Data.Weighting.Idf.IIdfScheme

    methods (Static)
        function scheme = create(name)
            % CREATE  Return an IIdfScheme object for the given name.
            import OceanStats.Data.Weighting.Idf.*

            switch lower(strtrim(name))
                case 'smooth'
                    scheme = SmoothIdf();
                case 'classic'
                    scheme = ClassicIdf();
                case 'bm25'
                    scheme = BM25Idf();
                otherwise
                    error('OceanStats:Data:Weighting:Idf:IdfSchemeFactory:Unknown', ...
                        'Unknown idf scheme "%s". Valid: smooth, classic, bm25.', name);
            end
        end

        function names = validNames()
            % VALIDNAMES  Return cell of all recognised idf scheme name strings.
            names = {'smooth', 'classic', 'bm25'};
        end
    end
end
