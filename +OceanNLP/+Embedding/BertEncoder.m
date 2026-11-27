classdef BertEncoder < OceanNLP.Embedding.IDenseEncoder
    % BERTENCODER  Hand-rolled BERT-family sentence encoder: raw text corpus
    % -> dense (N x D) mean-pooled sentence embeddings.
    %
    % THEORY / ASSEMBLY (master plan §3.3): wires together, in order, per
    % document:
    %   1. OceanNLP.Embedding.Tokenize.WordPieceTokenizer  (raw text -> subword tokens)
    %   2. OceanNLP.Embedding.Modules.TokenEmbedding        (token/pos/seg ids -> (d x T))
    %   3. an embedding-level OceanDL.Norm.LayerNormLayer
    %      (Epsilon=1e-12), applied to TokEmbed's output BEFORE the first
    %      transformer block — matches real BERT's `embeddings.LayerNorm`.
    %      Found missing via the master plan §5 external validation against
    %      real prajjwal1/bert-tiny weights (no synthetic-weight unit test
    %      has ground truth to catch an absent normalization step against;
    %      see §7 Execution Log for the diagnostic trail) — not present in
    %      the original design, added as a confirmed correctness fix.
    %   4. a stack of N OceanNLP.Embedding.Modules.TransformerEncoderBlock,
    %      run in sequence (output of block i feeds block i+1)             (d x T) -> (d x T)
    %   5. MEAN POOLING over all T token positions of the FINAL block's
    %      output -> (d x 1) sentence embedding.
    %   No new attention/embedding/normalization math is implemented here —
    %   this class is PURE ASSEMBLY of already-verified parts, wired
    %   together and independently re-verified for the wiring itself (see
    %   class tests: composing already-verified pieces does not guarantee
    %   the composition is correct).
    %
    % N (number of transformer layers) IS READ FROM THE SUPPLIED WEIGHTS,
    % NEVER HARDCODED: N = numel(Weights.Layers). A 2-layer bert-tiny
    % checkpoint and a 12-layer bert-base checkpoint both work through this
    % exact same class with no code change — only the Weights struct differs.
    %
    % POOLING STRATEGY — MEAN POOLING, NOT [CLS]-ONLY (documented choice,
    % load-bearing for later validation, do not silently change):
    %   docEmbedding = mean(X_final, 2)   over ALL T positions (including
    %   [CLS]/[SEP]), NOT X_final(:,1) ([CLS] alone). This is
    %   `sentence-transformers`' actual default pooling strategy for turning
    %   token-level BERT output into a single sentence vector — raw
    %   [CLS]-only pooling from an encoder with no fine-tuned pooling head is
    %   a well-known POOR unsupervised sentence embedding (the [CLS] vector
    %   is only meaningful once something has been trained to read it, e.g.
    %   next-sentence-prediction or a supervised head; this encoder does
    %   neither). The master plan's §5 external validation step compares
    %   this class's output against Python `transformers` output pooled THE
    %   SAME WAY (mean pooling) — a [CLS]-only MATLAB embedding compared
    %   against a mean-pooled Python embedding would silently invalidate that
    %   whole comparison, so this choice is called out explicitly rather than
    %   left implicit.
    %
    % WEIGHTS SCHEMA (the contract a future `export_bert_weights.py`, per
    % master plan §3.3/§5, MUST satisfy — read this section before writing
    % that script). `Weights` is a MATLAB struct with exactly these fields:
    %
    %   Weights.TokEmbedding.WordEmbeddings      (VocabSize x d) double
    %       <- HF state_dict "bert.embeddings.word_embeddings.weight"
    %          (VocabSize x d) AS-IS, no transpose: HF's embedding table is
    %          already row-indexed-by-id, exactly TokenEmbedding's W_tok layout.
    %   Weights.TokEmbedding.PositionEmbeddings  (MaxPositions x d) double
    %       <- HF "bert.embeddings.position_embeddings.weight", AS-IS.
    %   Weights.TokEmbedding.TokenTypeEmbeddings (NumSegmentTypes x d) double
    %       <- HF "bert.embeddings.token_type_embeddings.weight", AS-IS
    %          (BERT: NumSegmentTypes = 2).
    %   Weights.TokEmbedding.LayerNormGamma (d x 1), LayerNormBeta (d x 1)
    %       <- HF "bert.embeddings.LayerNorm.weight"/".bias". REAL BERT
    %          applies a LayerNorm to (word+position+segment embeddings)
    %          BEFORE the first transformer block — confirmed against a real
    %          trained checkpoint (master plan §7 Execution Log: found via
    %          the §5 external validation comparison, absent until then
    %          because no synthetic-weight unit test has ground truth to
    %          catch a missing normalization step against). Applied via
    %          obj.EmbeddingLayerNorm, constructed with Epsilon=1e-12
    %          (BERT's actual LayerNorm eps — NOT LayerNormLayer's own
    %          default of 1e-5; every LayerNormLayer this class constructs,
    %          including each block's LN1/LN2, uses 1e-12 for the same
    %          fidelity reason).
    %
    %   Weights.Layers — (1 x N) struct array, ONE entry per transformer
    %   layer, N = however many layers the checkpoint has (read from
    %   numel(Weights.Layers), never hardcoded). Weights.Layers(i)
    %   corresponds to HF's "bert.encoder.layer.{i-1}" (MATLAB 1-indexed
    %   struct-array position i <-> HF's 0-indexed layer index i-1). Each
    %   entry has:
    %     .WQ (d x d), .bQ (d x 1)  <- attention.self.query.weight/.bias
    %     .WK (d x d), .bK (d x 1)  <- attention.self.key.weight/.bias
    %     .WV (d x d), .bV (d x 1)  <- attention.self.value.weight/.bias
    %     .WO (d x d), .bO (d x 1)  <- attention.output.dense.weight/.bias
    %         ALL FOUR weight matrices are used AS-IS, NOT TRANSPOSED.
    %         PyTorch nn.Linear stores weight as (out_features, in_features)
    %         and computes y_row = x_row @ W^T + b (row-vector convention).
    %         MultiHeadAttention here computes the algebraically IDENTICAL
    %         quantity in column-vector form, Q = WQ*X + bQ, i.e.
    %         y_col = W @ x_col + b — the SAME (out,in) matrix W multiplies
    %         from the left instead of the (transposed) right. No transpose
    %         crosses the MATLAB/PyTorch boundary for any of WQ/WK/WV/WO/
    %         W1/W2 below; a naive export that DOES transpose them would be
    %         a real, silent bug (attention would still run, just wrong).
    %     .NumHeads (1,1) double — NOT a state_dict tensor; comes from the
    %         checkpoint's config.json "num_attention_heads" (constant
    %         across layers for a standard BERT checkpoint, but stored
    %         per-layer here rather than once globally, for schema
    %         generality/robustness against a hypothetical mixed-head-count
    %         export).
    %     .W1 (d_ff x d), .b1 (d_ff x 1) <- intermediate.dense.weight/.bias
    %         (same AS-IS, no-transpose rule as above)
    %     .W2 (d x d_ff), .b2 (d x 1)    <- output.dense.weight/.bias
    %         (same AS-IS, no-transpose rule as above)
    %     .LN1Gamma (d x 1), .LN1Beta (d x 1)
    %         <- attention.output.LayerNorm.weight/.bias (the LayerNorm
    %            applied AFTER the attention residual — TransformerEncoderBlock's
    %            LayerNorm1)
    %     .LN2Gamma (d x 1), .LN2Beta (d x 1)
    %         <- output.LayerNorm.weight/.bias (the LayerNorm applied AFTER
    %            the feed-forward residual — TransformerEncoderBlock's
    %            LayerNorm2)
    %
    %   d (hidden dim) and d_ff (feed-forward width) are inferred from the
    %   weight shapes, never separately specified — TokEmbed.HiddenDim,
    %   size(Layers(i).W1,1) respectively. embeddingDim() returns d.
    %
    % MATLAB CONSTRAINT WORKED AROUND (carried forward from J2, see master
    % plan §7 Execution Log — do not rediscover this): a property typed to a
    % class with no zero-argument constructor (WordPieceTokenizer,
    % TokenEmbedding, TransformerEncoderBlock all require constructor args)
    % fails at CLASS-LOAD time, because MATLAB tries to default-construct it
    % to validate the property declaration itself. Tokenizer/TokEmbed/Blocks
    % below are therefore declared WITHOUT a class-name type restriction;
    % the constructor's own local variables and the wrapped classes'
    % `arguments` blocks (already enforced when those objects are built)
    % provide the actual type guarantee.
    %
    % fit()/isFitted() POLICY (master plan §3.3, IDenseEncoder contract):
    % pretrained weights are loaded ONCE at construction, never learned from
    % a corpus. fit(corpus) is a documented NO-OP beyond flipping the
    % internal Fitted_ flag — it exists ONLY so obj.fit(x); obj.forward(y)
    % works identically for every IVectorizer/IDenseEncoder/IRanker in this
    % library. forward() still REQUIRES fit() to have been called first
    % (errors otherwise) for that same interface-consistency reason, even
    % though nothing corpus-specific actually happens inside fit().
    %
    % SCOPING (master plan §3.3): ONE document at a time, no batching, no
    % padding, so no attention mask is used (MultiHeadAttention itself now
    % supports an optional key mask; BertEncoder never needs it). forward(corpus) loops over
    % the (1 x N) documents internally and stacks per-document (1 x d) rows
    % into (N x d).
    %
    % USAGE:
    %   enc = OceanNLP.Embedding.BertEncoder('Weights', W, 'VocabFile', "vocab.txt", ...
    %                                          'ModelName', "bert-tiny");
    %   enc.fit(corpus);              % required no-op, see POLICY note
    %   E = enc.forward(corpus);      % (N x d)
    %
    % See also: OceanNLP.Embedding.IDenseEncoder, OceanNLP.Embedding.DenseEncoderFactory,
    %           OceanNLP.Embedding.Modules.TokenEmbedding,
    %           OceanNLP.Embedding.Modules.TransformerEncoderBlock,
    %           OceanNLP.Embedding.Tokenize.WordPieceTokenizer
    %
    % Implements the 3-pillar contract (.agents/pipeline-architecture-spec.md
    % §1.1) via getMetadata() and getPipelineSpec(), in addition to forward().

    properties (SetAccess = private)
        % The shared stack (tokenizer, embedding, LayerNorm, blocks) lives in
        % BertBackbone (neural-retrieval addendum 5.1). BertEncoder only adds
        % [CLS]/[SEP] mean pooling and the optional output normalization.
        Backbone     % OceanNLP.Embedding.BertBackbone
        NormalizeOutput (1,1) logical = false   % L2-normalize each pooled row (SBERT Normalize module)
        MaxLength (1,1) double = 512            % token cap incl. [CLS]/[SEP] (all-MiniLM-L6-v2 uses 256)
    end

    properties (Dependent, SetAccess = private)
        % Read-through views onto Backbone, kept so the pre-refactor public
        % surface (enc.Tokenizer, enc.Blocks, ...) is unchanged.
        Tokenizer
        TokEmbed
        EmbeddingLayerNorm
        Blocks
        ModelName
        ClsId
        SepId
        HiddenDim   % d, read from the loaded TokenEmbedding
        NumLayers   % N, numel(Blocks) - read from the supplied Weights, never hardcoded
    end

    properties (Access = private)
        Fitted_ (1,1) logical = false
    end

    methods
        function obj = BertEncoder(options)
            % BERTENCODER  Assemble the shared BertBackbone from a Weights
            % struct (see class WEIGHTS SCHEMA note) and a vocab.txt file.
            %   enc = BertEncoder('Weights', W, 'VocabFile', path)
            %   enc = BertEncoder('Weights', W, 'VocabFile', path, 'ModelName', "bert-tiny", ...
            %                     'NormalizeOutput', true)
            arguments
                options.Weights (1,1) struct
                options.VocabFile (1,1) string
                options.ModelName (1,1) string = "bert (unspecified checkpoint)"
                options.NormalizeOutput (1,1) logical = false
                options.MaxLength (1,1) double {mustBeInteger, mustBeGreaterThan(options.MaxLength, 2)} = 512
            end
            obj.Backbone = OceanNLP.Embedding.BertBackbone('Weights', options.Weights, ...
                'VocabFile', options.VocabFile, 'ModelName', options.ModelName);
            obj.NormalizeOutput = options.NormalizeOutput;
            obj.MaxLength = options.MaxLength;
        end

        function fit(obj, corpus)
            % FIT  Documented NO-OP beyond flipping isFitted() (see class
            % fit()/isFitted() POLICY note) — pretrained weights are loaded
            % at construction, not learned from any corpus. Still validates
            % `corpus` and is still REQUIRED before forward(), for calling-
            % pattern uniformity with every other class in this library.
            arguments
                obj
                corpus (1,:) {mustBeText} %#ok<INUSA>
            end
            obj.Fitted_ = true;
        end

        function tf = isFitted(obj)
            tf = obj.Fitted_;
        end

        function E = forward(obj, corpus)
            % FORWARD  Per-document: Backbone.encodeText -> mean-pool over
            % all T positions (-> optional L2 normalize). See class
            % ASSEMBLY / POOLING STRATEGY notes.
            %   corpus - (1 x N) raw text (single-document-at-a-time).
            %   E      - (N x d) double, row i = document i's sentence embedding.
            arguments
                obj
                corpus (1,:) {mustBeText}
            end
            if ~obj.isFitted()
                error('OceanNLP:Embedding:BertEncoder:NotFitted', ...
                    'Call fit(corpus) before forward().');
            end
            corpus = string(corpus);
            Nd = numel(corpus);
            E  = zeros(Nd, obj.HiddenDim);
            for i = 1:Nd
                X = obj.Backbone.encodeText(corpus(i), 'MaxLength', obj.MaxLength);                 % (d x T)
                row = mean(X, 2)';                                      % (1 x d), MEAN pooling over all T positions
                if obj.NormalizeOutput
                    row = row / max(norm(row), eps);
                end
                E(i, :) = row;
            end
        end

        function d = embeddingDim(obj)
            d = obj.HiddenDim;
        end

        function s = name(obj)
            s = "bert:" + obj.ModelName;
        end

        function d = get.HiddenDim(obj)
            d = obj.TokEmbed.HiddenDim;
        end

        function n = get.NumLayers(obj)
            n = numel(obj.Blocks);
        end

        function meta = getMetadata(obj)
            % GETMETADATA  Pillar 1 (symbolic) — honest-opacity stub (R7),
            % VIRTUAL-composite case (addendum §2.2), same pattern as
            % TfidfVectorizer/TransformerEncoderBlock.
            %
            %   BertEncoder = WordPieceTokenizer (IsSymbolic=false) +
            %   TokenEmbedding (IsSymbolic=true) + N x TransformerEncoderBlock
            %   (IsSymbolic=false, VIRTUAL) + mean pooling (a reduction, not
            %   claimed as a "formula" in its own right). IsSymbolic=false
            %   at THIS level is FORCED (not a judgment call, unlike
            %   TransformerEncoderBlock's own IsSymbolic choice): tokenization
            %   has no closed form, so no single valid end-to-end phi exists
            %   for the composite as a whole (same R7 reasoning as
            %   TfidfVectorizer/CountVectorizer, every composite that touches
            %   tokenization in this codebase).
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = containers.Map();
            meta.Assumptions = { ...
                'this class is a VIRTUAL composition of WordPieceTokenizer + TokenEmbedding + N x TransformerEncoderBlock + mean pooling; no attention/embedding/norm math is implemented here (see class ASSEMBLY note)', ...
                'IsSymbolic=false is FORCED, not a judgment call: the tokenization half has no closed form (R7), so no single end-to-end formula is claimed at this level', ...
                'pooling is MEAN over all T final-layer token positions, NOT [CLS]-only (see class POOLING STRATEGY note) — load-bearing for the §5 external validation comparison, which must pool the Python reference the same way', ...
                sprintf('N=%d transformer layers, hidden dim d=%d, read from the supplied Weights struct, never hardcoded', obj.NumLayers, obj.HiddenDim), ...
                'single document at a time, no batching, no padding, so no attention mask is used (master plan §3.3 scoping)'};
            meta.IsSymbolic = false;
            meta.Reference = sprintf( ...
                "Hand-rolled BERT-family encoder (master plan §3.3); loaded model = ""%s"" (%d layers, d=%d)", ...
                obj.ModelName, obj.NumLayers, obj.HiddenDim);

            % PROOF the pointers to each composed part's own getMetadata()
            % actually resolve (not just claimed in a comment) — call them
            % right now, unlike TransformerEncoderBlock's situation where
            % LayerNormLayer/GELUActivation have NO getMetadata() at all
            % (+OceanDL primitives). Here, every direct sub-part DOES
            % implement getMetadata(): WordPieceTokenizer, TokenEmbedding,
            % and TransformerEncoderBlock all do.
            tokenizerMeta = obj.Tokenizer.getMetadata();
            assert(isfield(tokenizerMeta, 'Reference') && strlength(tokenizerMeta.Reference) > 0, ...
                'OceanNLP:Embedding:BertEncoder:TokenizerMetadataUnresolved', ...
                'obj.Tokenizer.getMetadata() did not resolve to a non-empty Reference.');
            tokEmbedMeta = obj.TokEmbed.getMetadata();
            assert(isfield(tokEmbedMeta, 'SymbolicLatex') && strlength(tokEmbedMeta.SymbolicLatex) > 0, ...
                'OceanNLP:Embedding:BertEncoder:TokEmbedMetadataUnresolved', ...
                'obj.TokEmbed.getMetadata() did not resolve to a non-empty SymbolicLatex formula.');
            composedRef = struct( ...
                'TokenizerReference', tokenizerMeta.Reference, ...
                'TokEmbeddingSymbolicLatex', tokEmbedMeta.SymbolicLatex);
            if obj.NumLayers > 0
                blockMeta = obj.Blocks{1}.getMetadata();
                assert(isfield(blockMeta, 'Reference') && strlength(blockMeta.Reference) > 0, ...
                    'OceanNLP:Embedding:BertEncoder:BlockMetadataUnresolved', ...
                    'obj.Blocks{1}.getMetadata() did not resolve to a non-empty Reference.');
                composedRef.EncoderBlockReference = blockMeta.Reference;
            end
            meta.ComposedPartReference = composedRef;
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata. VIRTUAL: pure grouping of already-registered operator
            % kinds (WordPieceTokenizer, TokenEmbedding, MultiHeadAttention /
            % TransformerEncoderBlock); ComposesFrom names all of them per
            % master plan §3.3.
            spec = struct();
            spec.Inputs = struct('Name', "corpus", 'Domain', "raw-text", 'Dtype', "string", 'Shape', "1 x N");
            spec.Outputs = struct('Name', "E", 'Domain', "dense-vector-batch", 'Dtype', "double", 'Shape', "N x D");
            spec.Kind = "VIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "bert-encode");
            spec.ComposesFrom = {'OceanNLP.Embedding.Tokenize.WordPieceTokenizer', ...
                                  'OceanNLP.Embedding.Modules.TokenEmbedding', ...
                                  'OceanNLP.Embedding.Modules.MultiHeadAttention', ...
                                  'OceanNLP.Embedding.Modules.TransformerEncoderBlock'};
        end

        function v = get.Tokenizer(obj),          v = obj.Backbone.Tokenizer;          end
        function v = get.TokEmbed(obj),           v = obj.Backbone.TokEmbed;           end
        function v = get.EmbeddingLayerNorm(obj), v = obj.Backbone.EmbeddingLayerNorm; end
        function v = get.Blocks(obj),             v = obj.Backbone.Blocks;             end
        function v = get.ModelName(obj),          v = obj.Backbone.ModelName;          end
        function v = get.ClsId(obj),              v = obj.Backbone.ClsId;              end
        function v = get.SepId(obj),              v = obj.Backbone.SepId;              end

    end
end
