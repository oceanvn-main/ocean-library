classdef ANNRanker < OceanNLP.Retrieval.IRanker
    % ANNRANKER  Approximate Nearest Neighbor (ANN) Vector Search Retrieval.
    %
    % PURPOSE:
    %   Performs fast semantic vector retrieval by encoding text into dense embeddings
    %   and indexing them in an Approximate Nearest Neighbor (ANN) structure (e.g. HNSW, IVF).
    %
    % ROLES:
    %   - Encoder : IDenseEncoder object (encodes text -> dense embedding vector)
    %   - Index   : Vector index object (defaults to OceanAlgo.Search.HNSWIndex)
    %
    % See also: OceanNLP.Retrieval.IRanker,
    %           OceanNLP.Retrieval.DenseRanker,
    %           OceanAlgo.Search.HNSWIndex

    properties (SetAccess = private)
        Encoder     % IDenseEncoder instance for computing embeddings
        Index       % Vector index instance (e.g., HNSWIndex)
        NumDocs (1,1) double = 0
    end

    methods
        function obj = ANNRanker(options)
            arguments
                options.Encoder (1,1) OceanNLP.Embedding.IDenseEncoder
                options.Index (1,1) = OceanAlgo.Search.HNSWIndex()
            end
            obj.Encoder = options.Encoder;
            obj.Index   = options.Index;
        end
        function fit(obj, corpus)
            % FIT  Encode corpus to dense embeddings and build the ANN index.
            arguments
                obj
                corpus string
            end
            % 1. Encode corpus into dense embeddings (N x D)
            docEmbeddings = obj.Encoder.forward(corpus);

            % 2. Fit vector index on embeddings
            obj.Index.fit(docEmbeddings);
            obj.NumDocs = numel(corpus);
        end

        function [rankedIdx, scores] = rank(obj, query)
            % RANK  Encode query and perform approximate vector nearest-neighbor search.
            arguments
                obj
                query string
            end

            % 1. Encode query to dense vector (1 x D)
            q = obj.Encoder.forward(query);

            % 2. Query the approximate nearest-neighbor index
            [rankedIdx, scores] = obj.Index.search(q, obj.NumDocs);

            % 3. Extract top query row result
            rankedIdx = rankedIdx(1, :);
            scores    = scores(1, :);
        end

        function tf = isFitted(obj)
            tf = obj.NumDocs > 0 && obj.Index.IsFitted;
        end

        function s = name(obj)
            s = "ann-" + string(class(obj.Index));
        end

        function meta = getMetadata(obj)
            meta = struct('Name', obj.name(), 'IsApproximate', true, 'Index', obj.Index.getMetadata());
        end

        function spec = getPipelineSpec(obj)
            spec = struct( ...
                'Inputs', "text", ...
                'Outputs', ["ids", "scores"], ...
                'ComposesFrom', {class(obj.Encoder), class(obj.Index)});
        end
    end
end

