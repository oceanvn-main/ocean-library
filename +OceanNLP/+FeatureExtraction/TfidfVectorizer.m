classdef TfidfVectorizer < OceanNLP.FeatureExtraction.IVectorizer
    % TFIDFVECTORIZER  Raw document corpus -> sparse (N x V) TF-IDF matrix.
    %
    % THEORY:
    %   Pure composition, ZERO duplicated math (Master Plan §2/§4, Agent D
    %   acceptance criteria): internally wraps a
    %   OceanNLP.FeatureExtraction.CountVectorizer (raw text -> sparse
    %   counts) and an OceanStats.Data.Weighting.TfidfWeighting (counts ->
    %   TF-IDF weights). fit()/forward() here are thin delegating wrappers;
    %   the smoothed-IDF formula itself lives ONLY in TfidfWeighting.
    %
    % POLICY:
    %   - Constructor accepts 'Tokenizer' (forwarded to the internal
    %     CountVectorizer) and 'SublinearTf'/'Norm' (forwarded to the
    %     internal TfidfWeighting) — see those classes' help for exact
    %     semantics and defaults.
    %   - fit(corpus) := CountVectorizer.fit(corpus); then
    %     TfidfWeighting.fit(counts) on the resulting count matrix.
    %   - forward(corpus) := CountVectorizer.forward(corpus) piped straight
    %     into TfidfWeighting.forward(counts). Calling this class's
    %     forward() on a corpus is therefore numerically IDENTICAL to
    %     calling CountVectorizer.forward() and TfidfWeighting.forward()
    %     manually in sequence — verified empirically, see class tests.
    %   - Out-of-vocabulary handling, sparse-triplet safety, and OOV column
    %     semantics are entirely inherited from CountVectorizer — nothing
    %     is re-implemented here.
    %
    % USAGE:
    %   tfidf = OceanNLP.FeatureExtraction.TfidfVectorizer();
    %   tfidf.fit(corpus_train);
    %   Xtr = tfidf.forward(corpus_train);
    %   Xte = tfidf.forward(corpus_test);   % reuses fitted IDF — no leakage
    %
    %   tfidf2 = OceanNLP.FeatureExtraction.TfidfVectorizer( ...
    %       'SublinearTf', true, 'Norm', "l1");
    %
    % See also: OceanNLP.FeatureExtraction.IVectorizer,
    %           OceanNLP.FeatureExtraction.CountVectorizer,
    %           OceanStats.Data.Weighting.TfidfWeighting

    properties (SetAccess = private)
        Vectorizer (1,1) OceanNLP.FeatureExtraction.CountVectorizer = OceanNLP.FeatureExtraction.CountVectorizer()
        Weighting  (1,1) OceanStats.Data.Weighting.TfidfWeighting   = OceanStats.Data.Weighting.TfidfWeighting()
    end

    properties (Dependent, SetAccess = private)
        Vocabulary   % passthrough to Vectorizer.Vocabulary, for convenience/introspection
    end

    methods
        function obj = TfidfVectorizer(options)
            % TFIDFVECTORIZER  Construct, forwarding options to the wrapped
            % CountVectorizer (Tokenizer) and TfidfWeighting (SublinearTf, Norm).
            arguments
                options.Tokenizer   (1,1) OceanNLP.Tokenize.ITokenizer = OceanNLP.Tokenize.WhitespaceTokenizer()
                options.SublinearTf (1,1) logical = false
                options.Norm        (1,1) string  = "l2"
            end
            obj.Vectorizer = OceanNLP.FeatureExtraction.CountVectorizer('Tokenizer', options.Tokenizer);
            % TfidfWeighting's row-normalization is now a Strategy
            % (OceanStats.Data.Norm.INormScheme, mirroring IdfScheme) rather
            % than a plain string — convert this class's own 'Norm' string
            % option through the factory before forwarding it, so
            % TfidfVectorizer's public API is unchanged while staying wired
            % to TfidfWeighting's current constructor.
            normScheme = OceanStats.Data.Norm.NormSchemeFactory.create(options.Norm);
            obj.Weighting  = OceanStats.Data.Weighting.TfidfWeighting( ...
                'SublinearTf', options.SublinearTf, 'NormScheme', normScheme);
        end

        function fit(obj, corpus)
            % FIT  Delegates: CountVectorizer.fit -> counts -> TfidfWeighting.fit.
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            obj.Vectorizer.fit(corpus);
            counts = obj.Vectorizer.forward(corpus);
            obj.Weighting.fit(counts);
        end

        function X = forward(obj, corpus)
            % FORWARD  Delegates: CountVectorizer.forward -> TfidfWeighting.forward.
            %   No TF-IDF math is implemented in this method or class.
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            if ~obj.isFitted()
                error('OceanNLP:FeatureExtraction:TfidfVectorizer:NotFitted', ...
                    'Call fit(corpus) before forward().');
            end
            counts = obj.Vectorizer.forward(corpus);
            X = obj.Weighting.forward(counts);
        end

        function tf = isFitted(obj)
            tf = obj.Vectorizer.isFitted() && obj.Weighting.isFitted();
        end

        function v = get.Vocabulary(obj)
            v = obj.Vectorizer.Vocabulary;
        end

        function meta = getMetadata(~)
            % GETMETADATA  Pillar 1 (symbolic) — honest-opacity stub (R7),
            % VIRTUAL-composite case (addendum §2.2).
            %   TfidfVectorizer = CountVectorizer (tokenize+count, no closed
            %   form) composed with TfidfWeighting (reweight, HAS a closed
            %   form). Because the tokenization half has no formula, the
            %   composite as a whole has no single valid end-to-end phi
            %   either — that formula is NOT fabricated or copied up here.
            %   The real closed-form weighting step IS inspectable, one
            %   level down, via obj.Weighting.getMetadata().SymbolicLatex
            %   (obj.Weighting holds the wrapped TfidfWeighting instance).
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = {'this class is a VIRTUAL composition of CountVectorizer + TfidfWeighting; no TF-IDF math is implemented here (see class THEORY note)', ...
                                 'the tokenization half (CountVectorizer) has no closed form, so no single end-to-end formula is claimed at this level', ...
                                 'the real closed-form weighting formula is available one level down via obj.Weighting.getMetadata().SymbolicLatex'};
            meta.IsSymbolic = false;
            meta.Reference = "VIRTUAL composite (addendum §2.2 honesty rule): inspect obj.Weighting.getMetadata() for the real closed-form TF-IDF weighting formula";
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata. VIRTUAL: pure grouping of two already-registered
            % operators (CountVectorizer, TfidfWeighting); ComposesFrom names
            % both.
            spec = struct();
            spec.Inputs = struct('Name', "corpus", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x N");
            spec.Outputs = struct('Name', "X", 'Domain', "weighted-matrix", 'Dtype', "double", 'Shape', "N x V");
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "tf-idf vectorize");
            spec.ComposesFrom = {'OceanNLP.FeatureExtraction.CountVectorizer', 'OceanStats.Data.Weighting.TfidfWeighting'};
        end
    end
end
