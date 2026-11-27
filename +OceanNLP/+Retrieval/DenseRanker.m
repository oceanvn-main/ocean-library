classdef DenseRanker < OceanNLP.Retrieval.IRanker
    % DENSERANKER  Dense-embedding + cosine similarity retrieval: fit(corpus), rank(query).
    %
    % THEORY:
    %   Pure composition, ZERO duplicated encoding or cosine math: wraps an
    %   already-built OceanNLP.Embedding.IDenseEncoder (raw text -> a fixed
    %   (N x D) DENSE embedding matrix) and an
    %   OceanStats.Similarity.CosineSimilarity (all-vs-all row similarity).
    %   fit() encodes the whole corpus into embedding space exactly once and
    %   caches the resulting DENSE matrix as a buffer on self; rank() never
    %   re-encodes the corpus, only the (single, cheap) query. Structurally
    %   identical to SemanticRanker (TfidfVectorizer + CosineSimilarity) —
    %   same shape, a dense encoder standing in for the sparse vectorizer
    %   (Retrieval System Master Plan §3.4).
    %
    % CONSTRUCTOR / NO-DEFAULT-ENCODER NOTE:
    %   Unlike SemanticRanker, which can default-construct a bare
    %   TfidfVectorizer(), DenseRanker CANNOT default-construct an
    %   IDenseEncoder: IDenseEncoder is Abstract (never directly
    %   constructible), and its one concrete implementation, BertEncoder,
    %   REQUIRES a real 'Weights' struct + 'VocabFile' at construction —
    %   there is no meaningful zero-arg default. Encoder is therefore a
    %   REQUIRED name-value constructor argument (no default value in the
    %   `arguments` block; MATLAB rejects construction if it is omitted):
    %     ranker = OceanNLP.Retrieval.DenseRanker('Encoder', encoderInstance);
    %   'Similarity' stays optional, defaulting to a fresh CosineSimilarity()
    %   — mirroring SemanticRanker's own Similarity-defaulting pattern
    %   exactly, since CosineSimilarity (unlike IDenseEncoder) DOES have a
    %   sensible zero-arg default.
    %
    %   PROPERTY-TYPING NOTE (carried forward from BertEncoder's own
    %   "MATLAB CONSTRAINT WORKED AROUND", master plan §7 Execution Log): a
    %   property typed to a class with no public zero-argument constructor
    %   fails at CLASS-LOAD time, because MATLAB tries to default-construct
    %   it to validate the property declaration itself. This applies to
    %   Abstract classes too (they can never be constructed at all), so the
    %   `Encoder` property below is intentionally declared WITHOUT a
    %   class-name type restriction — the constructor's own `arguments`
    %   block (`options.Encoder (1,1) OceanNLP.Embedding.IDenseEncoder`) is
    %   where the actual type guarantee lives, exactly as BertEncoder does
    %   for its own untyped Tokenizer/TokEmbed/Blocks properties.
    %
    % POLICY:
    %   - Query encoding reuses the SAME fitted IDenseEncoder as
    %     fit(corpus) (same loaded weights, same tokenizer/vocab) —
    %     rank() treats query as a single-document "corpus" and calls
    %     Encoder.forward(query), never re-fitting on it. A (1,1) string
    %     query satisfies IDenseEncoder.forward()'s `corpus (1,:) {mustBeText}`
    %     contract directly with N=1 — no cell-wrapping required (confirmed
    %     against BertEncoder.forward()'s actual argument validation).
    %   - fit() before rank(): rank() throws
    %     OceanNLP:Retrieval:DenseRanker:NotFitted if called before fit().
    %   - CorpusMatrix / query embeddings are DENSE throughout (per
    %     IDenseEncoder.forward()'s contract) — never treated as sparse,
    %     unlike SemanticRanker's TF-IDF matrix.
    %   - rank() return shape is the OceanNLP.Retrieval.IRanker contract:
    %     [rankedIdx, scores] = rank(query), rankedIdx/scores (1 x N),
    %     rankedIdx a permutation of 1:N (original corpus positions) sorted
    %     by DESCENDING score (MATLAB's sort() is stable, so ties keep
    %     their original corpus order); scores(k) is the score of document
    %     rankedIdx(k). Identical shape/sort convention to
    %     BM25Ranker.rank()/SemanticRanker.rank() by design — any IRanker
    %     may be swapped in behind this interface (load-bearing for
    %     RetrievalEvaluator).
    %
    % USAGE:
    %   enc    = OceanNLP.Embedding.BertEncoder('Weights', W, 'VocabFile', "vocab.txt");
    %   dense  = OceanNLP.Retrieval.DenseRanker('Encoder', enc);
    %   dense.fit(corpus);                       % index once (embedding matrix)
    %   [idx, scores] = dense.rank("cat food");   % query many times
    %   topDocIdx = idx(1);                       % most relevant corpus position
    %
    % See also: OceanNLP.Retrieval.IRanker, OceanNLP.Retrieval.SemanticRanker,
    %           OceanNLP.Retrieval.BM25Ranker, OceanNLP.Embedding.IDenseEncoder,
    %           OceanNLP.Embedding.BertEncoder, OceanStats.Similarity.CosineSimilarity

    properties (SetAccess = private)
        % See class PROPERTY-TYPING NOTE: intentionally untyped at the
        % property level (IDenseEncoder is Abstract; no zero-arg default is
        % possible). Actual type is OceanNLP.Embedding.IDenseEncoder,
        % enforced by the constructor's `arguments` block.
        Encoder
        Similarity   (1,1) OceanStats.Similarity.CosineSimilarity = OceanStats.Similarity.CosineSimilarity()
        CorpusMatrix double = []   % (N x D) DENSE fitted embedding matrix (buffer)
        NumDocs      (1,1) double = 0
    end

    methods
        function obj = DenseRanker(options)
            % DENSERANKER  Construct with a REQUIRED, already-built
            % IDenseEncoder (see class CONSTRUCTOR note — no default is
            % possible) plus an optional Similarity strategy (defaults to a
            % fresh CosineSimilarity(), mirroring SemanticRanker's own
            % Similarity-defaulting pattern).
            %   ranker = DenseRanker('Encoder', encoderInstance)
            %   ranker = DenseRanker('Encoder', encoderInstance, 'Similarity', sim)
            arguments
                options.Encoder (1,1) OceanNLP.Embedding.IDenseEncoder
                options.Similarity (1,1) OceanStats.Similarity.CosineSimilarity = OceanStats.Similarity.CosineSimilarity()
            end
            obj.Encoder    = options.Encoder;
            obj.Similarity = options.Similarity;
        end

        function fit(obj, corpus)
            % FIT  Encode the whole corpus into embedding space once; cache on self.
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            obj.Encoder.fit(corpus);
            obj.CorpusMatrix = obj.Encoder.forward(corpus);   % (N x D) DENSE
            obj.NumDocs      = size(obj.CorpusMatrix, 1);
        end

        function [rankedIdx, scores] = rank(obj, query)
            % RANK  Score every fitted corpus document against query, sorted descending.
            %   [rankedIdx, scores] = rank(query) — see class/IRanker help for
            %   the exact return-shape contract.
            arguments
                obj
                query (1,1) string
            end
            if ~obj.isFitted()
                error('OceanNLP:Retrieval:DenseRanker:NotFitted', ...
                    'Call fit(corpus) before rank().');
            end

            % Single-document "corpus" through the SAME fitted encoder — a
            % (1,1) string satisfies IDenseEncoder.forward()'s (1,:)
            % corpus contract directly (N=1), no cell-wrapping needed.
            queryVec = obj.Encoder.forward(query);   % (1 x D) DENSE

            P = obj.Similarity.pairwise(queryVec, obj.CorpusMatrix);   % (1 x N)
            scoresByDoc = P(1, :);   % already dense (§3.3: never sparse)

            [sortedScores, order] = sort(scoresByDoc, 'descend');
            rankedIdx = order;
            scores    = sortedScores;
        end

        function tf = isFitted(obj)
            tf = obj.Encoder.isFitted();
        end

        function s = name(obj)
            s = "dense-" + obj.Encoder.name() + "-cosine";
        end

        function meta = getMetadata(obj)
            % GETMETADATA  Pillar 1 (symbolic) — honest-opacity stub (R7),
            % VIRTUAL-composite case (addendum §2.2), same pattern as
            % SemanticRanker/BM25Ranker.
            %   DenseRanker = IDenseEncoder implementation (itself
            %   non-symbolic — see e.g. BertEncoder.getMetadata(), whose
            %   tokenization half has no closed form) composed with
            %   CosineSimilarity (all-vs-all scoring, HAS a closed form).
            %   Because the encoding half has no end-to-end formula, the
            %   composite as a whole has no single valid end-to-end phi
            %   either — that formula is NOT fabricated or copied up here.
            %   The real closed-form cosine similarity formula IS
            %   inspectable, one level down, via
            %   obj.Similarity.getMetadata().SymbolicLatex (obj.Similarity
            %   holds the wrapped CosineSimilarity instance). obj.Encoder
            %   (the wrapped IDenseEncoder) is NOT cited here as a source
            %   of a formula — it is itself IsSymbolic=false, so it has
            %   none to point to (same discrimination SemanticRanker
            %   already makes between its symbolic Similarity and
            %   non-symbolic Vectorizer).
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = { ...
                'this class is a VIRTUAL composition of an IDenseEncoder implementation + CosineSimilarity; no encoding or similarity math is implemented here (see class THEORY note)', ...
                sprintf('the encoding half (%s) is itself IsSymbolic=false (its tokenization step has no closed form), so no single end-to-end formula is claimed at this level', class(obj.Encoder)), ...
                'the real closed-form cosine similarity formula is available one level down via obj.Similarity.getMetadata().SymbolicLatex'};
            meta.IsSymbolic = false;
            meta.Reference = "VIRTUAL composite (addendum §2.2 honesty rule): inspect obj.Similarity.getMetadata() for the real closed-form cosine similarity formula";
        end

        function spec = getPipelineSpec(obj)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata. VIRTUAL: pure grouping of two already-registered
            % operators (the wrapped IDenseEncoder implementation,
            % CosineSimilarity).
            %
            %   ComposesFrom NOTE (deliberate deviation from SemanticRanker/
            %   BM25Ranker's fixed-string ComposesFrom): those wrap a single
            %   CONCRETE class by property type (TfidfVectorizer,
            %   CountVectorizer). Encoder here is typed to the ABSTRACT
            %   IDenseEncoder interface (see class PROPERTY-TYPING note) and
            %   is genuinely polymorphic — a DenseRanker could wrap
            %   BertEncoder today or any future IDenseEncoder implementation
            %   tomorrow — so the composition target is read dynamically
            %   from the actual instance (class(obj.Encoder)) rather than
            %   hardcoded to "OceanNLP.Embedding.BertEncoder".
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "corpus", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x N (fit)"), ...
                struct('Name', "query", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1 (rank)")];
            spec.Outputs = [ ...
                struct('Name', "rankedIdx", 'Domain', "ranked-result", 'Dtype', "double", 'Shape', "1 x N"), ...
                struct('Name', "scores", 'Domain', "ranked-result", 'Dtype', "double", 'Shape', "1 x N")];
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "dense rank");
            spec.ComposesFrom = {class(obj.Encoder), 'OceanStats.Similarity.CosineSimilarity'};
        end
    end
end
