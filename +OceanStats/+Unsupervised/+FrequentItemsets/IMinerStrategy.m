classdef (Abstract) IMinerStrategy < handle
    % IMINERSTRATEGY  Abstract interface for frequent itemset mining algorithms.
    %
    %   All mining algorithms (Apriori, FP-Growth, etc.) must implement this
    %   interface, making them hot-swappable in the pipeline.
    %
    %   USAGE:
    %     miner = OceanStats.Unsupervised.FrequentItemsets.AprioriMiner();
    %     result = miner.mine(X, 0.01);
    %
    %   See also: AprioriMiner, FPGrowthMiner, TransactionEncoder
    
    methods (Abstract)
        result = mine(obj, X, minSupport, opts)
        %MINE  Find all frequent itemsets above minimum support threshold.
        %
        %   result = miner.mine(X, minSupport)
        %   result = miner.mine(X, minSupport, opts)
        %
        %   INPUTS:
        %     X          — sparse(N x p) logical matrix from TransactionEncoder
        %     minSupport — minimum support threshold (0..1, e.g., 0.01 = 1%)
        %     opts       — (optional) struct with algorithm-specific options
        %
        %   OUTPUT:
        %     result — struct with fields:
        %       .itemsets   — cell array of uint32 vectors (column indices)
        %       .support    — double vector (support value for each itemset)
        %       .stats      — struct with timing, pruning, k-level breakdown
    end
end
