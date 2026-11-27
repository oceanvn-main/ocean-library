classdef CrossEncoderReranker < OceanNLP.Retrieval.IRanker
    % CROSSENCODERRERANKER  Two-stage retrieval: a first-stage IRanker
    % proposes a shortlist, the CrossEncoder re-scores its top-K pairs.
    %
    % rank(query) returns a FULL permutation of 1:N so RetrievalEvaluator and
    % every metric work unchanged:
    %   positions 1..K  : the first stage's top-K, re-ordered by cross-encoder score
    %   positions K+1.. : the rest in first-stage order, scores = -Inf
    %                     ("not re-scored"; the two score scales are not
    %                     comparable, and -Inf keeps scores non-increasing)
    % K is required (no silent default). If K >= N all documents are re-scored.
    %
    % USAGE:
    %   rr = OceanNLP.Retrieval.CrossEncoderReranker('FirstStage', BM25Ranker(), ...
    %            'CrossEncoder', ce, 'K', 20);
    %   rr.fit(corpus);  [idx, s] = rr.rank("query");
    %
    % See also: OceanNLP.Embedding.CrossEncoder, OceanNLP.Retrieval.IRanker

    properties (SetAccess = private)
        FirstStage
        CrossEncoder
        K (1,1) double = 0
        Corpus (1,:) string = strings(1, 0)
    end

    methods
        function obj = CrossEncoderReranker(options)
            arguments
                options.FirstStage (1,1) OceanNLP.Retrieval.IRanker
                options.CrossEncoder (1,1) OceanNLP.Embedding.CrossEncoder
                options.K (1,1) double {mustBeInteger, mustBePositive}
            end
            for f = ["FirstStage" "CrossEncoder" "K"]
                if ~isfield(options, f)
                    error('OceanNLP:Retrieval:CrossEncoderReranker:MissingOption', ...
                        '"%s" is required (no silent default).', f);
                end
            end
            obj.FirstStage = options.FirstStage;
            obj.CrossEncoder = options.CrossEncoder;
            obj.K = options.K;
        end

        function fit(obj, corpus)
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            obj.FirstStage.fit(corpus);
            obj.Corpus = string(corpus);
        end

        function [rankedIdx, scores] = rank(obj, query)
            arguments
                obj
                query (1,1) string
            end
            if ~obj.isFitted()
                error('OceanNLP:Retrieval:CrossEncoderReranker:NotFitted', 'Call fit(corpus) before rank().');
            end
            %Pick K
            idx = obj.FirstStage.rank(query);
            N = numel(idx);
            k = min(obj.K, N);
            top = idx(1:k);

            %Re-score
            ceScores = obj.CrossEncoder.scoreDocs(query, obj.Corpus(top));
            [sortedScores, order] = sort(ceScores, 'descend');
            %Re-order by Cross-Encoder
            rankedIdx = [top(order), idx(k+1:end)];
            scores = [sortedScores, -Inf(1, N - k)];
        end

        function tf = isFitted(obj)
            tf = ~isempty(obj.Corpus) && obj.FirstStage.isFitted();
        end

        function s = name(obj)
            s = obj.FirstStage.name() + "+" + obj.CrossEncoder.name() + sprintf("@%d", obj.K);
        end

        function meta = getMetadata(obj)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = { ...
                'VIRTUAL composition of a first-stage IRanker + CrossEncoder; no scoring math implemented here', ...
                sprintf('only the first-stage top-K=%d are re-scored; the rest keep first-stage order with score -Inf', obj.K), ...
                'closed forms live one level down: obj.CrossEncoder.getMetadata() (heads) and the first stage'};
            meta.IsSymbolic = false;
            meta.Reference = "VIRTUAL composite: retrieve-then-rerank (Nogueira & Cho 2019); inspect obj.CrossEncoder.getMetadata()";
            ceMeta = obj.CrossEncoder.getMetadata();
            assert(strlength(ceMeta.ComposedPartReference.PoolerSymbolicLatex) > 0 && ...
                   strlength(ceMeta.ComposedPartReference.ClassifierSymbolicLatex) > 0, ...
                'OceanNLP:Retrieval:CrossEncoderReranker:CrossEncoderMetadataUnresolved', ...
                'CrossEncoder head formulas did not resolve.');
            firstMeta = obj.FirstStage.getMetadata();
            assert(isfield(firstMeta, 'Reference') && strlength(firstMeta.Reference) > 0, ...
                'OceanNLP:Retrieval:CrossEncoderReranker:FirstStageMetadataUnresolved', ...
                'First-stage metadata Reference is empty.');
            meta.ComposedPartReference = struct('CrossEncoderReference', ceMeta.Reference, ...
                'PoolerSymbolicLatex', ceMeta.ComposedPartReference.PoolerSymbolicLatex, ...
                'ClassifierSymbolicLatex', ceMeta.ComposedPartReference.ClassifierSymbolicLatex, ...
                'FirstStageReference', firstMeta.Reference);
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
            spec.DiagramSymbol = struct('Type', "default", 'Label', "rerank");
            spec.ComposesFrom = {class(obj.FirstStage), 'OceanNLP.Embedding.CrossEncoder'};
        end
    end
end
