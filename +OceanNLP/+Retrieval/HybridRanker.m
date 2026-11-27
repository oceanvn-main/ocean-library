classdef HybridRanker < OceanNLP.Retrieval.IRanker
    % HYBRIDRANKER  Hybrid Search / Multi-Retriever Rank Fusion.
    %
    % PURPOSE:
    %   Executes multiple distinct IRanker implementations in parallel for a query
    %   (e.g., keyword BM25 + dense vector embeddings) and merges their candidate
    %   rankings into a single unified result using a Fusion Strategy (e.g., RRF).
    %
    % ROLES:
    %   - Rankers : Cell array of distinct IRanker objects (e.g., {bm25, dense})
    %   - Fusion  : IFusionStrategy instance (e.g. ReciprocalRankFusion)
    %
    % See also: OceanNLP.Retrieval.IRanker,
    %           OceanNLP.Retrieval.MultiQueryRanker,
    %           OceanNLP.Fusion.ReciprocalRankFusion

    properties (SetAccess = private)
        Rankers cell = {}   % Cell array of IRanker objects
        Fusion              % IFusionStrategy object
        NumDocs (1,1) double = 0
    end

    methods
        function obj = HybridRanker(options)
            arguments
                options.Rankers (1,:) cell
                options.Fusion (1,1) OceanNLP.Fusion.IFusionStrategy
            end
            for i = 1:numel(options.Rankers)
                if ~isa(options.Rankers{i}, 'OceanNLP.Retrieval.IRanker')
                    error('OceanNLP:HybridRanker:BadRanker', 'Every ranker must implement IRanker.');
                end
            end
            obj.Rankers = options.Rankers;
            obj.Fusion  = options.Fusion;
        end

        function fit(obj, corpus)
            % FIT  Fit every underlying ranker on the document corpus.
            arguments
                obj
                corpus string
            end
            for i = 1:numel(obj.Rankers)
                obj.Rankers{i}.fit(corpus);
            end
            obj.NumDocs = numel(corpus);
        end

        function [ids, scores] = rank(obj, query)
            % RANK  Query all rankers in parallel and fuse their ranked results.
            arguments
                obj
                query string
            end
            if ~obj.isFitted()
                error('OceanNLP:HybridRanker:NotFitted', 'Call fit(corpus) before rank().');
            end

            % 1. Run each ranker independently on the query
            ranked = cell(1, numel(obj.Rankers));
            for i = 1:numel(obj.Rankers)
                [ranked{i}, ~] = obj.Rankers{i}.rank(query);
            end

            % 2. Merge (fuse) all ranked outputs using the fusion strategy
            [ids, scores] = obj.Fusion.fuse(ranked);
        end

        function tf = isFitted(obj)
            tf = obj.NumDocs > 0 && all(cellfun(@(r) r.isFitted(), obj.Rankers));
        end

        function s = name(obj)
            s = "hybrid-" + string(class(obj.Fusion));
        end
    end
end

