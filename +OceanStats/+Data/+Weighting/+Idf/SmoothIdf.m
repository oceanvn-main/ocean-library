classdef SmoothIdf < OceanStats.Data.Weighting.Idf.IIdfScheme
    % SMOOTHIDF  Smoothed idf (scikit-learn / Lucene "smooth_idf" convention).
    %
    %   idf_t = ln( (1+N) / (1+df_t) ) + 1
    %
    %   The default idf scheme for TfidfWeighting. Smoothing (+1 on both N
    %   and df_t) avoids division by zero and guarantees idf_t > 0 even for
    %   a term that occurs in every document (the un-smoothed classic
    %   formula gives exactly 0 in that case). Validated bit-for-bit against
    %   scikit-learn's TfidfTransformer(smooth_idf=True) on real data.
    %
    %   See also: OceanStats.Data.Weighting.Idf.IIdfScheme
    %             OceanStats.Data.Weighting.Idf.ClassicIdf

    methods
        function idf = compute(~, df, N)
            arguments
                ~
                df (1,:) double
                N  (1,1) double
            end
            idf = log((1 + N) ./ (1 + df)) + 1;
        end

        function s = name(~)
            s = "smooth";
        end

        function latex = symbolicLatex(~)
            latex = "\mathrm{idf}_t = \ln\!\left(\dfrac{1+N}{1+\mathrm{df}_t}\right) + 1";
        end
    end
end
