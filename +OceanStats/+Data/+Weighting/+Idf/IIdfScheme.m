classdef (Abstract) IIdfScheme < handle
    % IIDFSCHEME  Abstract interface for inverse-document-frequency formulas.
    %
    %   Strategy Pattern: isolates the idf_t = f(df_t, N) formula from the
    %   weighting scheme that consumes it (TfidfWeighting, BM25Weighting).
    %   A new idf formula is added by implementing this interface — no
    %   existing weighting class's source is touched, and any weighting
    %   class that accepts an IIdfScheme via its constructor can use it
    %   immediately (Open/Closed: open for extension, closed for
    %   modification).
    %
    %   Objects are stateless strategies (compare IActivation) — they hold
    %   no fitted data of their own; TfidfWeighting/BM25Weighting still own
    %   the fitted IDF buffer, they just delegate ITS FORMULA to whichever
    %   IIdfScheme they were constructed with.
    %
    %   CONVENTION:
    %     df  — (1 x V) document frequency per term (count of docs the term
    %           occurs in at least once)
    %     N   — (1,1) total number of documents in the fitted corpus
    %     idf — (1 x V) inverse document frequency per term
    %
    %   USAGE:
    %     scheme = OceanStats.Data.Weighting.Idf.IdfSchemeFactory.create('smooth');
    %     idf    = scheme.compute(df, N);
    %
    %   See also: OceanStats.Data.Weighting.Idf.IdfSchemeFactory
    %             OceanStats.Data.Weighting.TfidfWeighting
    %             OceanStats.Data.Weighting.BM25Weighting

    methods (Abstract)
        idf = compute(obj, df, N)
        % COMPUTE  Elementwise idf_t = f(df_t, N).
        %   df  — (1 x V) double, document frequency per term
        %   N   — (1,1) double, total document count
        %   idf — (1 x V) double

        s = name(obj)
        % NAME  Lowercase registry key used by IdfSchemeFactory.

        latex = symbolicLatex(obj)
        % SYMBOLICLATEX  Single-line LaTeX for this scheme's idf formula —
        %   composed into the owning weighting class's own getMetadata()
        %   so the reported formula stays honest (R7) no matter which
        %   scheme is actually plugged in, instead of the owning class
        %   hardcoding one specific formula's LaTeX.
    end
end
