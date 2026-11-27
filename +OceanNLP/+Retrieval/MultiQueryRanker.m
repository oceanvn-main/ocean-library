classdef MultiQueryRanker < OceanNLP.Retrieval.IRanker
    % MULTIQUERYRANKER  Query Expansion / Multi-Query Retrieval.
    %
    % PURPOSE:
    %   Expands a single user query into multiple query variations (synonyms,
    %   rephrasings, or expansions), runs the same BaseRanker against each
    %   variation, and merges (fuses) the resulting ranked lists using Rank Fusion.
    %
    % ROLES:
    %   - BaseRanker : Single underlying retriever (e.g. BM25Ranker, DenseRanker)
    %   - Expander   : Generates N expanded variations of the user query
    %   - Fuser      : Merges the N ranked candidate lists into 1 final ranking list
    %
    % See also: OceanNLP.Retrieval.IRanker,
    %           OceanNLP.Retrieval.HybridRanker,
    %           OceanNLP.Fusion.ReciprocalRankFusion

    properties (SetAccess = private)
        BaseRanker  % Single underlying IRanker object (e.g., BM25Ranker)
        Expander    % QueryExpander instance for generating query variations
        Fuser       % IFusionStrategy (defaults to ReciprocalRankFusion)
        NumDocs (1,1) double = 0
    end

    methods
        function obj = MultiQueryRanker(options)
            arguments
                options.BaseRanker (1,1) OceanNLP.Retrieval.IRanker
                options.Expander (1,1) OceanNLP.Retrieval.QueryExpander
                options.Fuser (1,1) OceanNLP.Fusion.ReciprocalRankFusion = OceanNLP.Fusion.ReciprocalRankFusion()
            end
            obj.BaseRanker = options.BaseRanker;
            obj.Expander   = options.Expander;
            obj.Fuser      = options.Fuser;
        end

        function fit(obj, corpus)
            % FIT  Fit the underlying BaseRanker on the corpus.
            arguments
                obj
                corpus string
            end
            obj.BaseRanker.fit(corpus);
            obj.NumDocs = numel(corpus);
        end

        function [ids, scores] = rank(obj, query)
            % RANK  Expand query, run BaseRanker for each variation, and fuse results.
            arguments
                obj
                query string
            end
            if ~obj.isFitted()
                error('OceanNLP:MultiQueryRanker:NotFitted', 'Call fit(corpus) before rank().');
            end

            % 1. Expand single query into multiple variations
            qs = obj.Expander.expand(query);

            % 2. Retrieve ranked candidate lists for each query variation
            ranked = cell(1, numel(qs));
            for i = 1:numel(qs)
                [ranked{i}, ~] = obj.BaseRanker.rank(qs(i));
            end

            % 3. Fuse candidate lists into one final ranking
            [ids, scores] = obj.Fuser.fuse(ranked);
        end

        function tf = isFitted(obj)
            tf = obj.NumDocs > 0 && obj.BaseRanker.isFitted();
        end

        function s = name(obj)
            s = "multi-query-" + obj.BaseRanker.name();
        end
    end
end

