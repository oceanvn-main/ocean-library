classdef BM25Weighting < OceanStats.Data.Weighting.IWeighting
    % BM25WEIGHTING  Okapi BM25 term weighting (Robertson & Sparck Jones, 1994).
    %
    % THEORY:
    %   Defaults (K1=1.2, B=0.75) match Elasticsearch/Lucene's production
    %   defaults for the BM25 similarity module.
    %
    %   idf_t   = log( (N - df_t + 0.5) / (df_t + 0.5) + 1 )     (non-negative
    %                                                              variant, Lucene/ES —
    %                                                              this is the DEFAULT
    %                                                              IdfScheme; see below)
    %   w(d,t)  = idf_t * tf(d,t)*(K1+1) / (tf(d,t) + K1*(1-B+B*|d|/avgdl))
    %
    %   IDF SCHEME (Strategy Pattern, +Idf/IIdfScheme.m): the idf_t formula
    %   above is decoupled into a pluggable OceanStats.Data.Weighting.Idf.
    %   IIdfScheme, defaulting to BM25Idf (shown above). Pass a different
    %   scheme via the constructor to use a different idf formula without
    %   touching this class's source — e.g.:
    %     bm25 = OceanStats.Data.Weighting.BM25Weighting( ...
    %         'IdfScheme', OceanStats.Data.Weighting.Idf.IdfSchemeFactory.create('classic'));
    %
    %   forward(counts) returns the full (N x V) per-document-per-term BM25
    %   weight matrix. Ranking a query is then a single sparse dot product:
    %     scores = W(:, queryTermIds) * queryTermCounts';
    %   See OceanNLP.Retrieval.BM25Ranker for the end-to-end query workflow.
    %
    % USAGE:
    %   bm25 = OceanStats.Data.Weighting.BM25Weighting();
    %   bm25.fit(counts_train);
    %   Xtr = bm25.forward(counts_train);
    %
    % SPARSE-SAFETY: computed via the [i,j,v] = find(counts) triplet pattern
    %   (master plan §3.2 item 2) — the saturating tf transform touches only
    %   stored nonzeros; `counts + denom` (which would force a dense (N x V)
    %   matrix) is never formed. Verified empirically with issparse() — see
    %   this class's test file / Agent B verification log.
    %
    % See also: OceanStats.Data.Weighting.IWeighting
    %           OceanStats.Data.Weighting.TfidfWeighting
    %
    % Implements the 3-pillar contract (.agents/pipeline-architecture-spec.md
    % §1.1) via getMetadata() and getPipelineSpec(), in addition to fit/forward.

    properties
        % TermSaturation (textbook k1): how quickly repeated occurrences of a
        %   term stop adding score. Small -> tf saturates fast (binary-like);
        %   large -> score keeps growing almost linearly with tf.
        TermSaturation (1,1) double {mustBePositive} = 1.2
        % LengthNormalization (textbook b): how strongly a document's length
        %   discounts its tf. 0 -> length ignored; 1 -> full |d|/avgdl scaling.
        %   Long documents have more opportunity to repeat a term by chance.
        LengthNormalization (1,1) double {mustBeInRange(LengthNormalization,0,1)} = 0.75
        IdfScheme (1,1) OceanStats.Data.Weighting.Idf.IIdfScheme = OceanStats.Data.Weighting.Idf.BM25Idf()   % Strategy — overridable via constructor
    end

    properties (Dependent)
        K1   % alias of TermSaturation (textbook symbol, kept for backward compatibility)
        B    % alias of LengthNormalization (textbook symbol, kept for backward compatibility)
    end

    properties (SetAccess = private)
        IDF          (1,:) double = []
        AvgDocLength (1,1) double = 0
        VocabSize    (1,1) double = 0
    end

    methods
        function obj = BM25Weighting(options)
            arguments
                options.TermSaturation (1,1) double {mustBePositive} = 1.2
                options.LengthNormalization (1,1) double {mustBeInRange(options.LengthNormalization,0,1)} = 0.75
                % Legacy textbook-symbol names; if given they override the above.
                options.K1 double {mustBePositive} = []
                options.B  double {mustBeInRange(options.B,0,1)} = []
                % Strategy Pattern (see +Idf/IIdfScheme.m): the idf formula
                % itself is decoupled from this class. Default preserves the
                % exact previously-validated Elasticsearch/Lucene behavior;
                % pass a different OceanStats.Data.Weighting.Idf.IIdfScheme
                % to swap formulas WITHOUT modifying this class's source.
                options.IdfScheme (1,1) OceanStats.Data.Weighting.Idf.IIdfScheme = OceanStats.Data.Weighting.Idf.BM25Idf()
            end
            obj.TermSaturation      = options.TermSaturation;
            obj.LengthNormalization = options.LengthNormalization;
            if ~isempty(options.K1), obj.TermSaturation      = options.K1; end
            if ~isempty(options.B),  obj.LengthNormalization = options.B;  end
            obj.IdfScheme = options.IdfScheme;
        end

        function v = get.K1(obj),  v = obj.TermSaturation;      end
        function set.K1(obj, v),   obj.TermSaturation = v;      end
        function v = get.B(obj),   v = obj.LengthNormalization; end
        function set.B(obj, v),    obj.LengthNormalization = v; end

        function fit(obj, counts)
            % FIT  Learn per-term IDF and corpus average document length.
            %   WARNING: this writes directly onto obj's own buffers (IDF,
            %   AvgDocLength, VocabSize) — it does not return a copy. Since
            %   obj is a handle, every reference to it (e.g. a BM25Ranker that
            %   holds this object in its own Weighting property) sees the
            %   new values immediately. Calling fit() a second time overwrites
            %   whatever was previously learned; if that happens on anything
            %   other than the original training corpus, it silently defeats
            %   the anti-leakage guarantee ("fit on train only, forward() on
            %   train and test") this class's design otherwise gives you.
            arguments
                obj
                counts (:,:) double
            end
            if obj.isFitted()
                warning('OceanStats:Data:Weighting:BM25Weighting:RefitOverwrite', ...
                    ['fit() called on an already-fitted BM25Weighting instance. ' ...
                     'This overwrites the previously learned IDF/AvgDocLength in ' ...
                     'place — any other reference to this same object (e.g. a ' ...
                     'BM25Ranker holding it) will immediately see the new values. ' ...
                     'If the original fit was on training data, re-fitting on ' ...
                     'anything else (e.g. test data) introduces data leakage. ' ...
                     'Construct a new %s instance instead if you need independent ' ...
                     'fitted state.'], class(obj));
            end
            N  = size(counts, 1);
            df = sum(counts > 0, 1);
            obj.IDF          = obj.IdfScheme.compute(df, N);
            obj.AvgDocLength = mean(sum(counts, 2));
            obj.VocabSize    = size(counts, 2); %Dimensions of columns = number of unique terms
        end

        function W = forward(obj, counts)
            % FORWARD  Apply fitted BM25 weighting to a (possibly new) count matrix.
            arguments
                obj
                counts (:,:) double
            end
            if ~obj.isFitted()
                error('OceanStats:Data:Weighting:BM25Weighting:NotFitted', ...
                    'Call fit(counts) before forward().');
            end
            if size(counts, 2) ~= obj.VocabSize
                error('OceanStats:Data:Weighting:BM25Weighting:VocabMismatch', ...
                    'forward() got %d columns, fit() saw %d.', size(counts,2), obj.VocabSize);
            end

            % Sparse-safe: transform only stored nonzeros via the triplet pattern
            % (§3.2 item 2) — do NOT compute `counts + denom` directly, it forces
            % a dense (N x V) matrix.

            [i, j, v] = find(counts); %v = term frequency of the word in the document. tf_{t,d}
            % find() on a (1 x V) row-vector matrix is documented to return i/j/v
            % as ROW vectors (special case; every other shape returns columns).
            % Force column orientation unconditionally so the arithmetic below is
            % shape-consistent regardless of counts's dimensions — otherwise
            % obj.IDF(j)' becomes a column while satTf_v stays a row and the
            % elementwise .* implicitly broadcasts to a dense (nnz x nnz) matrix.
            i = i(:); j = j(:); v = v(:); %Ensure col vec
            dl        = sum(counts, 2);                                   % (N x 1), dense, cheap. Look at documnet 1, adds up all of its word counts, put that total in dl(1)..
            denom_i   = obj.TermSaturation * (1 - obj.LengthNormalization + obj.LengthNormalization .* dl(i) / obj.AvgDocLength);
            % satTf_v   = v  ./ (v + denom_i); % Textbook version
            satTf_v   = (v .* (obj.TermSaturation + 1)) ./ (v + denom_i);
            w_v       = satTf_v .* obj.IDF(j)';
            W = sparse(i, j, w_v, size(counts,1), size(counts,2));
        end

        function w = computePostingWeights(obj, tf, df, docLengths, N, avgDocLength)
            % COMPUTEPOSTINGWEIGHTS  Compute BM25 term weights for a vector of postings.
            arguments
                obj
                tf double
                df (1,1) double
                docLengths double
                N (1,1) double
                avgDocLength (1,1) double = obj.AvgDocLength
            end
            if avgDocLength == 0 && obj.AvgDocLength > 0
                avgDocLength = obj.AvgDocLength;
            end
            if avgDocLength == 0
                avgDocLength = 1;
            end

            tf = tf(:); docLengths = docLengths(:);   % avoid implicit-expansion to a matrix
            idf = obj.IdfScheme.compute(df, N);
            denom = obj.TermSaturation * (1 - obj.LengthNormalization + obj.LengthNormalization * (docLengths / avgDocLength));
            satTf = (tf * (obj.TermSaturation + 1)) ./ (tf + denom);
            w = idf .* satTf;
        end

        function tf = isFitted(obj)
            tf = ~isempty(obj.IDF);
        end

        function s = stateDict(obj)
            % STATEDICT  Snapshot fitted buffers + config for save/reload.
            %   IdfScheme is stored by NAME (via IdfSchemeFactory), not as a
            %   serialized object — loadStateDict() reconstructs it through
            %   the factory.
            s = struct('IDF', obj.IDF, 'AvgDocLength', obj.AvgDocLength, ...
                        'VocabSize', obj.VocabSize, 'K1', obj.K1, 'B', obj.B, ...
                        'IdfSchemeName', obj.IdfScheme.name());
        end

        function loadStateDict(obj, s)
            % LOADSTATEDICT  Restore buffers + config from stateDict() output.
            arguments
                obj
                s (1,1) struct
            end
            obj.IDF          = s.IDF;
            obj.AvgDocLength = s.AvgDocLength;
            obj.VocabSize    = s.VocabSize;
            obj.K1           = s.K1;
            obj.B            = s.B;
            if isfield(s, 'IdfSchemeName')
                obj.IdfScheme = OceanStats.Data.Weighting.Idf.IdfSchemeFactory.create(s.IdfSchemeName);
            end
        end

        function meta = getMetadata(obj)
            % GETMETADATA  Pillar 1 (symbolic) — closed-form LaTeX, symbol
            % dictionary, assumptions, and reference backing fit/forward.
            % idf line comes from the plugged-in IdfScheme, not a hardcoded
            % formula — keeps this honest (R7) no matter which scheme is
            % active. The saturating-tf weighting line below is BM25-specific
            % and always applies, regardless of idf scheme.
            % NOTE (future extension point, out of scope for this pass):
            % K1/B are natural ParamSpec candidates (spec §12a) — tunable,
            % LaTeX-symbol-bearing, sensible default+range — do not implement now.
            meta = struct();
            meta.SymbolicLatex = [obj.IdfScheme.symbolicLatex(), ...
                                   "w_{d,t} = \mathrm{idf}_t \cdot \dfrac{\mathrm{tf}_{d,t}\,(k_1+1)}{\mathrm{tf}_{d,t} + k_1\left(1-b+b\,\frac{|d|}{\mathrm{avgdl}}\right)}"];
            meta.Symbols = containers.Map();
            meta.Symbols('N') = "total number of documents (fit-time)";
            meta.Symbols('df_t') = "number of documents containing term t";
            meta.Symbols('tf_{d,t}') = "raw count of term t in document d";
            meta.Symbols('idf_t') = "inverse document frequency of term t (formula set by IdfScheme)";
            meta.Symbols('w_{d,t}') = "BM25 weight";
            meta.Symbols('k_1') = "term-frequency saturation (tunable, default 1.2)";
            meta.Symbols('b') = "document-length normalization strength (tunable, default 0.75)";
            meta.Symbols('|d|') = "length of document d";
            meta.Symbols('avgdl') = "corpus average document length";
            meta.Assumptions = {'counts nonnegative', 'K1 > 0', '0 <= B <= 1', ...
                                 sprintf('idf formula is the "%s" IdfScheme strategy — swap it via the constructor, not by editing this class', obj.IdfScheme.name())};
            meta.IsSymbolic = true;
            meta.Reference = sprintf("idf: %s scheme (%s); weighting: Robertson & Sparck Jones (1994) Okapi BM25 saturating-tf formula, Elasticsearch/Lucene K1/B defaults", ...
                obj.IdfScheme.name(), class(obj.IdfScheme));
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata for this NONVIRTUAL leaf operator.
            spec = struct();
            spec.Inputs = struct('Name', "counts", 'Domain', "count-matrix", 'Dtype', "double", 'Shape', "N x V");
            spec.Outputs = struct('Name', "W", 'Domain', "weighted-matrix", 'Dtype', "double", 'Shape', "N x V");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "BM25");
            spec.ComposesFrom = {};
        end
    end
end
