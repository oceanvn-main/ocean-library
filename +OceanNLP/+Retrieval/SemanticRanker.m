classdef SemanticRanker < OceanNLP.Retrieval.IRanker
    % SEMANTICRANKER  TF-IDF + cosine similarity retrieval: fit(corpus), rank(query).
    %
    % THEORY:
    %   Pure composition, ZERO duplicated TF-IDF or cosine math: internally
    %   wraps an OceanNLP.FeatureExtraction.TfidfVectorizer (raw text -> a
    %   fixed (N x V) TF-IDF matrix, itself a composition of
    %   CountVectorizer + TfidfWeighting) and an
    %   OceanStats.Similarity.CosineSimilarity (all-vs-all row similarity).
    %   fit() vectorizes the whole corpus into TF-IDF space exactly once
    %   and caches the resulting matrix as a buffer on self; rank() never
    %   re-vectorizes the corpus, only the (single, cheap) query.
    %
    %   "Semantic" here means TF-IDF cosine overlap (classic vector-space
    %   retrieval), not a learned dense embedding — despite the name, no
    %   neural encoder is involved; the name distinguishes this ranker's
    %   continuous, normalized cosine scoring from BM25Ranker's lexical
    %   count-weighted scoring, matching the plan's "semantic vs lexical"
    %   two-ranker split (Master Plan §5, Agent E).
    %
    % POLICY:
    %   - Query vectorization reuses the SAME fitted TfidfVectorizer as
    %     fit(corpus) (same Vocabulary, same fitted IDF) — rank() treats
    %     query as a single-document corpus and calls
    %     Vectorizer.forward(query), never re-fitting on it (no leakage,
    %     matches TfidfVectorizer's own train/test reuse contract).
    %   - OUT-OF-VOCABULARY query terms: Vocabulary.encode() (invoked
    %     inside the wrapped CountVectorizer) is OOV-safe — unseen tokens
    %     collapse to the reserved OOV column, which carries zero TF-IDF
    %     weight for every fit-time document, so OOV terms simply do not
    %     contribute to the cosine score rather than erroring.
    %   - EMPTY OR ALL-OOV QUERY (TF-IDF row is all-zero): cosine
    %     similarity is 0 against every document (CosineSimilarity's Eps
    %     guard avoids a 0/0 division) — rank() still returns cleanly, with
    %     all documents tied at score 0 in original corpus order.
    %   - fit() before rank(): rank() throws
    %     OceanNLP:Retrieval:SemanticRanker:NotFitted if called before fit().
    %   - rank() return shape is the OceanNLP.Retrieval.IRanker contract:
    %     [rankedIdx, scores] = rank(query), rankedIdx/scores (1 x N),
    %     rankedIdx a permutation of 1:N (original corpus positions) sorted
    %     by DESCENDING score; scores(k) is the score of document
    %     rankedIdx(k). Identical shape to BM25Ranker.rank() by design —
    %     either ranker may be swapped in behind IRanker.
    %
    % USAGE:
    %   sem = OceanNLP.Retrieval.SemanticRanker();
    %   sem.fit(corpus);                       % index once (TF-IDF matrix)
    %   [idx, scores] = sem.rank("cat food");   % query many times
    %   topDocIdx = idx(1);                     % most relevant corpus position
    %
    %   sem2 = OceanNLP.Retrieval.SemanticRanker('SublinearTf', true, ...
    %       'Tokenizer', OceanNLP.Tokenize.RegexTokenizer('Pattern', "[\w']+"));
    %
    % See also: OceanNLP.Retrieval.IRanker, OceanNLP.Retrieval.BM25Ranker,
    %           OceanNLP.FeatureExtraction.TfidfVectorizer,
    %           OceanStats.Similarity.CosineSimilarity

    properties (SetAccess = private)
        Vectorizer   (1,1) OceanNLP.FeatureExtraction.TfidfVectorizer = OceanNLP.FeatureExtraction.TfidfVectorizer()
        Similarity   (1,1) OceanStats.Similarity.CosineSimilarity     = OceanStats.Similarity.CosineSimilarity()
        CorpusMatrix double = []   % (N x V) sparse fitted TF-IDF matrix (buffer)
        NumDocs      (1,1) double = 0
    end

    properties (Dependent, SetAccess = private)
        Vocabulary   % passthrough to Vectorizer.Vocabulary, for introspection
    end

    methods
        function obj = SemanticRanker(options)
            % SEMANTICRANKER  Construct, forwarding options to the wrapped
            % TfidfVectorizer (Tokenizer, SublinearTf, Norm) and
            % CosineSimilarity (Eps).
            arguments
                options.Tokenizer   (1,1) OceanNLP.Tokenize.ITokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
                options.SublinearTf (1,1) logical = false
                options.Norm        (1,1) string  = "l2"
                options.Eps         (1,1) double {mustBePositive} = 1e-8
            end
            obj.Vectorizer = OceanNLP.FeatureExtraction.TfidfVectorizer( ...
                'Tokenizer', options.Tokenizer, 'SublinearTf', options.SublinearTf, ...
                'Norm', options.Norm);
            obj.Similarity = OceanStats.Similarity.CosineSimilarity('Eps', options.Eps);
        end

        function fit(obj, corpus)
            % FIT  Vectorize the whole corpus into TF-IDF space once; cache on self.
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            obj.Vectorizer.fit(corpus);
            obj.CorpusMatrix = obj.Vectorizer.forward(corpus);
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
                error('OceanNLP:Retrieval:SemanticRanker:NotFitted', ...
                    'Call fit(corpus) before rank().');
            end

            % Single-document "corpus" through the SAME fitted vectorizer —
            % same Vocabulary, same fitted IDF, no re-fitting on the query.
            queryVec = obj.Vectorizer.forward(query);      % (1 x V) sparse

            P = obj.Similarity.pairwise(queryVec, obj.CorpusMatrix);  % (1 x N)
            scoresByDoc = full(P(1, :));

            [sortedScores, order] = sort(scoresByDoc, 'descend');
            rankedIdx = order;
            scores    = sortedScores;
        end

        function tf = isFitted(obj)
            tf = obj.Vectorizer.isFitted();
        end

        function s = name(~)
            s = "semantic-tfidf-cosine";
        end

        function v = get.Vocabulary(obj)
            v = obj.Vectorizer.Vocabulary;
        end

        function meta = getMetadata(~)
            % GETMETADATA  Pillar 1 (symbolic) — honest-opacity stub (R7),
            % VIRTUAL-composite case (addendum §2.2).
            %   SemanticRanker = TfidfVectorizer (itself non-symbolic — see
            %   TfidfVectorizer.getMetadata(), tokenize+count+reweight, no
            %   single end-to-end formula) composed with CosineSimilarity
            %   (all-vs-all scoring, HAS a closed form). Because the
            %   vectorization half has no end-to-end formula, the composite
            %   as a whole has no single valid end-to-end phi either — that
            %   formula is NOT fabricated or copied up here. The real
            %   closed-form cosine similarity formula IS inspectable, one
            %   level down, via obj.Similarity.getMetadata().SymbolicLatex
            %   (obj.Similarity holds the wrapped CosineSimilarity
            %   instance). obj.Vectorizer (the wrapped TfidfVectorizer) is
            %   NOT cited here as a source of a formula — it is itself
            %   IsSymbolic=false, so it has none to point to.
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {'this class is a VIRTUAL composition of TfidfVectorizer + CosineSimilarity; no vectorization or similarity math is implemented here (see class THEORY note)', ...
                                 'the vectorization half (TfidfVectorizer) is itself IsSymbolic=false (its own tokenization half has no closed form), so no single end-to-end formula is claimed at this level', ...
                                 'the real closed-form cosine similarity formula is available one level down via obj.Similarity.getMetadata().SymbolicLatex'};
            meta.IsSymbolic = false;
            meta.Reference = "VIRTUAL composite (addendum §2.2 honesty rule): inspect obj.Similarity.getMetadata() for the real closed-form cosine similarity formula";
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata. VIRTUAL: pure grouping of two already-registered
            % operators (TfidfVectorizer, CosineSimilarity); ComposesFrom
            % names both.
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "corpus", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x N (fit)"), ...
                struct('Name', "query", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x 1 (rank)")];
            spec.Outputs = [ ...
                struct('Name', "rankedIdx", 'Domain', "ranked-result", 'Dtype', "double", 'Shape', "1 x N"), ...
                struct('Name', "scores", 'Domain', "ranked-result", 'Dtype', "double", 'Shape', "1 x N")];
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "semantic rank");
            spec.ComposesFrom = {'OceanNLP.FeatureExtraction.TfidfVectorizer', 'OceanStats.Similarity.CosineSimilarity'};
        end
    end
end
