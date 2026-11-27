classdef BiEncoderRanker < OceanNLP.Retrieval.IRanker
    % BIENCODERRANKER  Bi-encoder retrieval: query and document are encoded
    % INDEPENDENTLY into one vector each; score = similarity of the vectors.
    %
    % Differs from DenseRanker in three ways: separate QueryEncoder /
    % DocEncoder (DPR-style; pass the same object twice for a shared "siamese"
    % SBERT encoder, or omit QueryEncoder), any ISimilarity (default dot
    % product, the natural pairing for L2-normalizing encoders), and the
    % encoders are IDenseEncoder objects, so BertEncoder('NormalizeOutput',true)
    % gives the all-MiniLM-L6-v2 pipeline (mean pool + normalize).
    %
    % fit(corpus) encodes every document once (offline step); rank(query)
    % costs one query encoding plus one exact full-scan similarity.
    %
    % USAGE:
    %   enc = OceanNLP.Embedding.BertEncoder('Weights', W, 'VocabFile', v, 'NormalizeOutput', true);
    %   r = OceanNLP.Retrieval.BiEncoderRanker('DocEncoder', enc);
    %   r.fit(corpus);  [idx, s] = r.rank("query");
    %
    % See also: OceanNLP.Retrieval.DenseRanker, OceanStats.Similarity.DotProductSimilarity

    properties (SetAccess = private)
        QueryEncoder
        DocEncoder
        Similarity
        CorpusMatrix double = []     % (N x D) buffer
        NumDocs (1,1) double = 0
    end

    methods
        function obj = BiEncoderRanker(options)
            arguments
                options.DocEncoder (1,1) OceanNLP.Embedding.IDenseEncoder
                options.QueryEncoder (1,1) OceanNLP.Embedding.IDenseEncoder
                options.Similarity (1,1) OceanStats.Similarity.ISimilarity = OceanStats.Similarity.DotProductSimilarity()
            end
            if ~isfield(options, 'DocEncoder')
                error('OceanNLP:Retrieval:BiEncoderRanker:MissingOption', '"DocEncoder" is required.');
            end
            obj.DocEncoder = options.DocEncoder;
            if isfield(options, 'QueryEncoder')
                obj.QueryEncoder = options.QueryEncoder;
            else
                obj.QueryEncoder = options.DocEncoder;     % shared ("siamese") encoder
            end
            obj.Similarity = options.Similarity;
            if obj.QueryEncoder.embeddingDim() ~= obj.DocEncoder.embeddingDim()
                error('OceanNLP:Retrieval:BiEncoderRanker:DimMismatch', ...
                    'Query encoder dim %d and document encoder dim %d must match.', ...
                    obj.QueryEncoder.embeddingDim(), obj.DocEncoder.embeddingDim());
            end
        end

        function fit(obj, corpus)
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            obj.DocEncoder.fit(corpus);
            if ~obj.QueryEncoder.isFitted()
                obj.QueryEncoder.fit(corpus);
            end
            obj.CorpusMatrix = obj.DocEncoder.forward(corpus);
            obj.NumDocs = size(obj.CorpusMatrix, 1);
        end

        function [rankedIdx, scores] = rank(obj, query)
            arguments
                obj
                query (1,1) string
            end
            if ~obj.isFitted()
                error('OceanNLP:Retrieval:BiEncoderRanker:NotFitted', 'Call fit(corpus) before rank().');
            end
            q = obj.QueryEncoder.forward(query);                       % (1 x D)
            P = obj.Similarity.pairwise(q, obj.CorpusMatrix);          % (1 x N)
            [scores, rankedIdx] = sort(P(1, :), 'descend');
        end

        function tf = isFitted(obj)
            tf = obj.NumDocs > 0 && obj.DocEncoder.isFitted();
        end

        function s = name(obj)
            s = "bi-encoder-" + obj.DocEncoder.name() + "-" + obj.Similarity.name();
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = { ...
                'VIRTUAL composition of query/document IDenseEncoders + an ISimilarity; no encoding or similarity math is implemented here', ...
                sprintf('encoders (%s) are IsSymbolic=false (tokenization has no closed form), so no end-to-end formula is claimed', class(obj.DocEncoder)), ...
                'the closed-form similarity is available via obj.Similarity.getMetadata().SymbolicLatex', ...
                'exact full-scan search over all documents (no approximate nearest-neighbour index)'};
            meta.IsSymbolic = false;
            meta.Reference = "VIRTUAL composite: inspect obj.Similarity.getMetadata(); Reimers & Gurevych 2019 (SBERT), Karpukhin et al. 2020 (DPR)";
            simMeta = obj.Similarity.getMetadata();
            assert(strlength(simMeta.SymbolicLatex) > 0, 'OceanNLP:Retrieval:BiEncoderRanker:SimilarityMetadataUnresolved', ...
                'Similarity metadata did not resolve.');
            meta.ComposedPartReference = struct('SimilaritySymbolicLatex', simMeta.SymbolicLatex);
        end

        function spec = getPipelineSpec(obj)
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "corpus", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x N (fit)"), ...
                struct('Name', "query", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1 (rank)")];
            spec.Outputs = [ ...
                struct('Name', "rankedIdx", 'Domain', "ranked-result", 'Dtype', "double", 'Shape', "1 x N"), ...
                struct('Name', "scores", 'Domain', "ranked-result", 'Dtype', "double", 'Shape', "1 x N")];
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "bi-encoder rank");
            spec.ComposesFrom = {class(obj.DocEncoder), class(obj.Similarity)};
        end
    end
end
