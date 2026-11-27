classdef ColBERTRanker < OceanNLP.Retrieval.IRanker
    % COLBERTRANKER  Late-interaction retrieval: per-token embeddings for
    % query and documents, score = MaxSim (sum over query tokens of the best
    % document-token match). EXACT full scan: no PLAID/centroid pruning, no
    % approximate index.
    %
    % fit(corpus) encodes every document to a (nd x dim) matrix, stored as
    % SINGLE precision (matches the reference; double would double memory for
    % no accuracy gain). rank(query) encodes the query once and scores every
    % stored document matrix.
    % ColBERT DON'T ALLOW query and doc enter the Transformer together
    %
    %
    % USAGE:
    %   enc = OceanNLP.Embedding.ColBERTEncoder('Weights', W, 'VocabFile', v);
    %   r = OceanNLP.Retrieval.ColBERTRanker('Encoder', enc);
    %   r.fit(corpus);  [idx, s] = r.rank("query");
    %
    % See also: OceanNLP.Embedding.ColBERTEncoder, OceanStats.Similarity.LateInteraction

    properties (SetAccess = private)
        Encoder
        Interaction = OceanStats.Similarity.LateInteraction()
        DocMatrices (1,:) cell = {}      % buffer: N cells of (nd x dim) single
        NumDocs (1,1) double = 0
    end

    methods
        function obj = ColBERTRanker(options)
            arguments
                options.Encoder (1,1) OceanNLP.Embedding.ColBERTEncoder
            end
            obj.Encoder = options.Encoder;
        end

        function fit(obj, corpus)
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            corpus = string(corpus);
            docs = cell(1, numel(corpus));
            for i = 1:numel(corpus)
                docs{i} = single(obj.Encoder.encodeDocument(corpus(i)));
            end
            obj.DocMatrices = docs;
            obj.NumDocs = numel(docs);
        end

        function [rankedIdx, scores] = rank(obj, query)
            arguments
                obj
                query (1,1) string
            end
            if ~obj.isFitted()
                error('OceanNLP:Retrieval:ColBERTRanker:NotFitted', 'Call fit(corpus) before rank().');
            end
            Q = obj.Encoder.encodeQuery(query);
            [scores, rankedIdx] = sort(obj.Interaction.score(Q, obj.DocMatrices), 'descend');
        end

        function tf = isFitted(obj)
            tf = obj.NumDocs > 0;
        end

        function s = name(obj)
            s = "late-interaction-" + obj.Encoder.name();
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = { ...
                'VIRTUAL composition of ColBERTEncoder + LateInteraction; no encoding or MaxSim math implemented here', ...
                'the encoder is IsSymbolic=false (tokenization has no closed form); MaxSim is closed-form via obj.Interaction.getMetadata()', ...
                'exact full-scan MaxSim over all documents; document token matrices stored as single precision'};
            meta.IsSymbolic = false;
            meta.Reference = "VIRTUAL composite: inspect obj.Interaction.getMetadata() for the closed-form MaxSim (Khattab & Zaharia 2020)";
            liMeta = obj.Interaction.getMetadata();
            assert(strlength(liMeta.SymbolicLatex) > 0, 'OceanNLP:Retrieval:ColBERTRanker:InteractionMetadataUnresolved', ...
                'LateInteraction metadata did not resolve.');
            meta.ComposedPartReference = struct('MaxSimSymbolicLatex', liMeta.SymbolicLatex);
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "corpus", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x N (fit)"), ...
                struct('Name', "query", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1 (rank)")];
            spec.Outputs = [ ...
                struct('Name', "rankedIdx", 'Domain', "ranked-result", 'Dtype', "double", 'Shape', "1 x N"), ...
                struct('Name', "scores", 'Domain', "ranked-result", 'Dtype', "double", 'Shape', "1 x N")];
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "colbert rank");
            spec.ComposesFrom = {'OceanNLP.Embedding.ColBERTEncoder', 'OceanStats.Similarity.LateInteraction'};
        end
    end
end
