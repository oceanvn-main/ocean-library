classdef ClassicIdf < OceanStats.Data.Weighting.Idf.IIdfScheme
    % CLASSICIDF  Original unsmoothed idf (Sparck Jones, 1972).
    %
    %   idf_t = log10( N / df_t )
    %
    %   The textbook definition (Manning, Raghavan & Schutze, "Introduction
    %   to Information Retrieval", eq. 11.5). Two differences from
    %   OceanStats.Data.Weighting.Idf.SmoothIdf, both consequences of having
    %   no smoothing term:
    %     - idf_t = 0 exactly for a term that occurs in every document
    %       (N/df_t = 1, log10(1) = 0) — such a term contributes nothing to
    %       any TF-IDF weight under this scheme.
    %     - undefined (division by zero) if df_t = 0. Not reachable when df
    %       comes from a vocabulary built over the same corpus being scored
    %       (every in-vocabulary term has df_t >= 1 by construction), but a
    %       real constraint if this scheme is ever fed a df vector from a
    %       different source.
    %
    %   See also: OceanStats.Data.Weighting.Idf.IIdfScheme
    %             OceanStats.Data.Weighting.Idf.SmoothIdf

    methods
        function idf = compute(~, df, N)
            arguments
                ~
                df (1,:) double {mustBePositive}
                N  (1,1) double
            end
            idf = log10(N ./ df);
        end

        function s = name(~)
            s = "classic";
        end

        function latex = symbolicLatex(~)
            latex = "\mathrm{idf}_t = \log_{10}\!\left(\dfrac{N}{\mathrm{df}_t}\right)";
        end
    end
end
