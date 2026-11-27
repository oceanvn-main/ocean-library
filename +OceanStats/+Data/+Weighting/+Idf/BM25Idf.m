classdef BM25Idf < OceanStats.Data.Weighting.Idf.IIdfScheme
    % BM25IDF  Non-negative idf variant used by Elasticsearch/Lucene's BM25
    %   similarity (Robertson & Sparck Jones, 1994, patched to avoid the
    %   negative-idf case the original Robertson formula allows).
    %
    %   idf_t = ln( (N - df_t + 0.5) / (df_t + 0.5) + 1 )
    %
    %   The default idf scheme for BM25Weighting. Validated bit-for-bit
    %   against a direct reproduction of the Elasticsearch/Lucene formula on
    %   real data.
    %
    %   See also: OceanStats.Data.Weighting.Idf.IIdfScheme
    %             OceanStats.Data.Weighting.BM25Weighting

    methods
        function idf = compute(~, df, N)
            arguments
                ~
                df (1,:) double
                N  (1,1) double
            end
            idf = log((N - df + 0.5) ./ (df + 0.5) + 1);
        end

        function s = name(~)
            s = "bm25";
        end

        function latex = symbolicLatex(~)
            latex = "\mathrm{idf}_t = \ln\!\left(\dfrac{N-\mathrm{df}_t+0.5}{\mathrm{df}_t+0.5}+1\right)";
        end
    end
end
