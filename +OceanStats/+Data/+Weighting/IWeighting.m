classdef (Abstract) IWeighting < handle
    % IWEIGHTING  Abstract interface for corpus-level count reweighting.
    %
    %   PyTorch-style contract: fit() mutates buffers on self (like
    %   BatchNorm running stats); forward() reads them back. Operates on
    %   ANY (N docs x V terms) sparse count matrix — domain-agnostic
    %   (bag-of-words, bag-of-visual-words, etc. all fit this shape).
    %
    %   USAGE:
    %     w = TfidfWeighting();
    %     w.fit(counts_train);          % learns df/idf from TRAIN ONLY
    %     Xtr = w.forward(counts_train);
    %     Xte = w.forward(counts_test); % reuses fitted buffers — no leakage
    %
    %   See also: OceanStats.Data.Weighting.TfidfWeighting
    %             OceanStats.Data.Weighting.BM25Weighting

    methods (Abstract)
        fit(obj, counts)
        % FIT  Learn corpus-level statistics (idf, avg length, ...) from counts.
        %   counts — (N x V) sparse non-negative count matrix. Mutates obj.

        W = forward(obj, counts)
        % FORWARD  Apply the fitted weighting to a (possibly new) count matrix.
        %   counts — (N x V) sparse count matrix, same V as fit().
        %   W      — (N x V) weighted matrix.

        tf = isFitted(obj)
        s  = stateDict(obj)
        loadStateDict(obj, s)
    end
end
