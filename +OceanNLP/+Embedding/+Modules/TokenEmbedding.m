classdef TokenEmbedding < handle
    % TOKENEMBEDDING  BERT-style input embedding: token + positional + segment lookup-and-sum.
    %
    % THEORY:
    %   FORMULA (per sequence position t, t = 1..T):
    %     E(:,t) = W_tok[id_t, :]' + W_pos[t, :]' + W_seg[seg_t, :]'
    %   i.e. the embedding at position t is the elementwise SUM of three
    %   independent lookups: which token it is (W_tok), where it is in the
    %   sequence (W_pos), and which segment/sentence it belongs to (W_seg).
    %   This is exactly BERT's input embedding (Devlin et al., 2018, §3.1 /
    %   Figure 2) — no scaling, no further transform, just three table
    %   lookups summed elementwise. (The subsequent LayerNorm that BERT
    %   applies to this sum is NOT part of this class — it belongs to
    %   TransformerEncoderBlock, a later, separate piece; this class's
    %   contract stops at the raw summed embedding.)
    %
    % WEIGHTS ARE PRETRAINED, NOT LEARNED HERE:
    %   W_tok/W_pos/W_seg come from a pretrained checkpoint (e.g. bert-tiny)
    %   loaded elsewhere and passed directly to the constructor. This class
    %   has no fit() — there is nothing corpus-specific to learn; it is a
    %   pure lookup-and-sum operator over fixed tables (matches the master
    %   plan §3.3 "weights loaded at construction, not learned by fit()"
    %   policy for the encoder stack as a whole).
    %
    % INDEXING CONVENTION — READ BEFORE CALLING (0-indexed BERT vs 1-indexed MATLAB):
    %   `ids` and `segmentIds` are expected in BERT/HuggingFace's NATIVE
    %   0-INDEXED convention:
    %     - ids(t)        in [0, VocabSize-1]        (vocab.txt line number, 0-based)
    %     - segmentIds(t) in [0, NumSegmentTypes-1]   (BERT: 0 = sentence A, 1 = sentence B)
    %   Sequence POSITION is likewise 0-indexed internally by this class
    %   (position of the first token is 0, matching BERT's own position-id
    %   convention — NOT the token's 1-based location in a MATLAB array):
    %     pos_t = t - 1   for t = 1..T   (t is the 1-based MATLAB loop/column index)
    %   MATLAB matrices/arrays are always 1-indexed, so every 0-indexed BERT
    %   value above is converted to a MATLAB row index by adding exactly 1
    %   immediately before indexing into W_tok/W_pos/W_seg, and nowhere else:
    %     W_tok row used for token t  = ids(t) + 1
    %     W_pos row used for token t  = pos_t  + 1   =  t          (i.e. row t itself)
    %     W_seg row used for token t  = segmentIds(t) + 1
    %   Concretely: BERT id 0 (commonly "[PAD]" in bert-tiny's vocab.txt,
    %   line 1) reads MATLAB row 1 of W_tok; BERT position 0 (the first
    %   token of the sequence) reads MATLAB row 1 of W_pos; BERT segment 0
    %   reads MATLAB row 1 of W_seg. Passing already-1-indexed/MATLAB-style
    %   ids here would silently look up the WRONG row (off by one) — this is
    %   exactly the class of bug this convention note exists to prevent.
    %
    % CONVENTION (feature-major output, matches +OceanDL):
    %   forward() returns E as (d x T) — column t is the embedding of
    %   sequence position t. This matches GELUActivation.forward(Z) and
    %   LayerNormLayer.normalize(Z_in, mode)'s (d x N) convention (column =
    %   one sample/position), NOT the (N x D) sample-major convention used
    %   by the rest of +OceanNLP (CountVectorizer, Vocabulary, ...) — this
    %   is the documented shape-convention boundary from the master plan
    %   §3.3: everything inside +OceanNLP/+Embedding/+Modules is feature-
    %   major because it feeds directly into the reused +OceanDL layers.
    %
    % SCOPING (matches master plan §3.3): ONE document at a time, no
    %   batching, no padding. ids/segmentIds are (1 x T) for a single
    %   document; multi-document batching is an explicit non-goal here.
    %
    % USAGE:
    %   emb = OceanNLP.Embedding.Modules.TokenEmbedding(W_tok, W_pos, W_seg);
    %   E   = emb.forward(ids);                 % segmentIds defaults to all-zeros (1 x T)
    %   E   = emb.forward(ids, segmentIds);      % explicit two-segment input
    %
    % See also: OceanDL.Activation.GELUActivation,
    %           OceanDL.Norm.LayerNormLayer
    %
    % Implements the 3-pillar contract (.agents/pipeline-architecture-spec.md
    % §1.1) via getMetadata() and getPipelineSpec(), in addition to forward().

    properties (SetAccess = private)
        WTok (:,:) double   % (VocabSize x d) token-id embedding table
        WPos (:,:) double   % (MaxPositions x d) positional embedding table
        WSeg (:,:) double   % (NumSegmentTypes x d) segment/token-type embedding table
    end

    properties (Dependent, SetAccess = private)
        HiddenDim        % d, shared column count across all three tables
        VocabSize         % size(WTok, 1)
        MaxPositions       % size(WPos, 1)
        NumSegmentTypes    % size(WSeg, 1)
    end

    methods
        function obj = TokenEmbedding(W_tok, W_pos, W_seg)
            % TOKENEMBEDDING  Construct directly from pretrained weight tables.
            %   emb = TokenEmbedding(W_tok, W_pos, W_seg)
            %   W_tok — (VocabSize x d) double
            %   W_pos — (MaxPositions x d) double
            %   W_seg — (NumSegmentTypes x d) double, BERT uses NumSegmentTypes = 2
            arguments
                W_tok (:,:) double
                W_pos (:,:) double
                W_seg (:,:) double
            end
            d = size(W_tok, 2);
            if size(W_pos, 2) ~= d || size(W_seg, 2) ~= d
                error('OceanNLP:Embedding:Modules:TokenEmbedding:HiddenDimMismatch', ...
                    ['W_tok, W_pos, and W_seg must all share the same hidden ' ...
                     'dimension d (number of columns). Got d = %d (W_tok), ' ...
                     '%d (W_pos), %d (W_seg).'], d, size(W_pos, 2), size(W_seg, 2));
            end
            obj.WTok = W_tok;
            obj.WPos = W_pos;
            obj.WSeg = W_seg;
        end

        function E = forward(obj, ids, segmentIds)
            % FORWARD  Lookup-and-sum embedding for ONE document.
            %   E = obj.forward(ids)               % segmentIds defaults to zeros(1, T)
            %   E = obj.forward(ids, segmentIds)
            %   ids/segmentIds — (1 x T), BERT-native 0-indexed (see class
            %                    CONVENTION note above). segmentIds entries
            %                    must be in [0, NumSegmentTypes-1].
            %   E — (d x T) double, feature-major (see class CONVENTION note).
            arguments
                obj
                ids (1,:) double {mustBeInteger, mustBeNonnegative}
                segmentIds (1,:) double {mustBeInteger, mustBeNonnegative} = zeros(1, numel(ids))
            end
            T = numel(ids);
            if numel(segmentIds) ~= T
                error('OceanNLP:Embedding:Modules:TokenEmbedding:LengthMismatch', ...
                    'ids has %d elements but segmentIds has %d; they must match.', ...
                    T, numel(segmentIds));
            end
            if any(ids > obj.VocabSize - 1)
                error('OceanNLP:Embedding:Modules:TokenEmbedding:IdOutOfRange', ...
                    'ids must be in [0, %d] (0-indexed, VocabSize=%d); got max id %d.', ...
                    obj.VocabSize - 1, obj.VocabSize, max(ids));
            end
            if T > obj.MaxPositions
                error('OceanNLP:Embedding:Modules:TokenEmbedding:SequenceTooLong', ...
                    'Sequence length T=%d exceeds MaxPositions=%d.', T, obj.MaxPositions);
            end
            if any(segmentIds > obj.NumSegmentTypes - 1)
                error('OceanNLP:Embedding:Modules:TokenEmbedding:SegmentIdOutOfRange', ...
                    'segmentIds must be in [0, %d] (0-indexed, NumSegmentTypes=%d); got max %d.', ...
                    obj.NumSegmentTypes - 1, obj.NumSegmentTypes, max(segmentIds));
            end

            % 0-indexed BERT convention -> 1-indexed MATLAB row, exactly at
            % the point of use (see class CONVENTION note).
            tokRows = ids + 1;                 % (1 x T)
            posRows = (0:(T - 1)) + 1;          % (1 x T), == 1:T
            segRows = segmentIds + 1;          % (1 x T)

            E = obj.WTok(tokRows, :)' + obj.WPos(posRows, :)' + obj.WSeg(segRows, :)';  % (d x T)
        end

        function d = get.HiddenDim(obj)
            d = size(obj.WTok, 2);
        end

        function v = get.VocabSize(obj)
            v = size(obj.WTok, 1);
        end

        function m = get.MaxPositions(obj)
            m = size(obj.WPos, 1);
        end

        function n = get.NumSegmentTypes(obj)
            n = size(obj.WSeg, 1);
        end

        function meta = getMetadata(~)
            % GETMETADATA  Pillar 1 (symbolic) — closed-form LaTeX, symbol
            % dictionary, assumptions, and reference for the lookup-and-sum.
            meta = struct();
            meta.SymbolicLatex = "E_t = W_{\mathrm{tok}}[\mathrm{id}_t] + W_{\mathrm{pos}}[t] + W_{\mathrm{seg}}[\mathrm{seg}_t]";
            meta.Symbols = containers.Map();
            meta.Symbols('E_t') = "embedding vector at sequence position t (column t of the output, d x 1)";
            meta.Symbols('W_tok') = "token embedding table (VocabSize x d), row indexed by token id";
            meta.Symbols('W_pos') = "positional embedding table (MaxPositions x d), row indexed by position";
            meta.Symbols('W_seg') = "segment/token-type embedding table (NumSegmentTypes x d), row indexed by segment id";
            meta.Symbols('id_t') = "0-indexed BERT vocabulary id of the token at position t";
            meta.Symbols('t') = "0-indexed BERT sequence position (first token = 0)";
            meta.Symbols('seg_t') = "0-indexed BERT segment id of the token at position t (0 or 1)";
            meta.Assumptions = {'single document, no batching, no padding (master plan §3.3 scoping)', ...
                                 'ids/segmentIds/position are BERT-native 0-indexed; converted to 1-indexed MATLAB row access internally (see class CONVENTION note)', ...
                                 'ids in [0, VocabSize-1]; segmentIds in [0, NumSegmentTypes-1]; T <= MaxPositions'};
            meta.IsSymbolic = true;
            meta.Reference = "BERT input embedding (Devlin et al., 2018)";
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata for this NONVIRTUAL leaf operator.
            spec = struct();
            spec.Inputs = [ ...
                struct('Name', "ids", 'Domain', "term-id-sequence", 'Dtype', "double", 'Shape', "1 x T"), ...
                struct('Name', "segmentIds", 'Domain', "term-id-sequence", 'Dtype', "double", 'Shape', "1 x T")];
            spec.Outputs = struct('Name', "E", 'Domain', "dense-vector-batch", 'Dtype', "double", 'Shape', "d x T");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "tok-embed");
            spec.ComposesFrom = {};
        end
    end
end
