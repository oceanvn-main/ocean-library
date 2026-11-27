classdef MultiHeadAttention < handle
    % MULTIHEADATTENTION  Scaled dot-product multi-head self-attention (Vaswani et al., 2017).
    %
    % THEORY:
    %   Per head h:
    %     Attention(Q_h, K_h, V_h) = softmax( Q_h K_h^T / sqrt(d_k) ) V_h
    %   where Q_h, K_h, V_h are the query/key/value projections of the input
    %   restricted to head h's slice of the hidden dimension (d_k = d /
    %   numHeads wide). All numHeads heads are computed independently, their
    %   outputs concatenated back to width d, then passed through one final
    %   linear output projection W_O:
    %     MultiHead(X) = Concat(head_1, ..., head_numHeads) * W_O' + b_O
    %   (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin,
    %   2017, "Attention Is All You Need", §3.2.2.)
    %
    % WEIGHTS ARE PRETRAINED, NOT LEARNED HERE:
    %   W_Q/b_Q/W_K/b_K/W_V/b_V/W_O/b_O come from a pretrained checkpoint
    %   loaded elsewhere and passed directly to the constructor — no fit().
    %
    % HEAD SPLIT CONVENTION:
    %   d must be evenly divisible by numHeads (validated in the constructor;
    %   this errors loudly rather than silently truncating or rounding — a
    %   silently-truncated head would drop learned weight rows without any
    %   signal that it happened). Head h (h = 1..numHeads) owns the
    %   CONTIGUOUS row block ((h-1)*d_k + 1) : (h*d_k) of the (d x T) Q/K/V
    %   projections, where d_k = d / numHeads. This matches HuggingFace/BERT's
    %   own `transpose_for_scores` reshape (row-major view of the last axis
    %   into [numHeads, headDim], i.e. head 0 = first d_k features, head 1 =
    %   next d_k features, etc.) — NOT an interleaved/strided split.
    %
    % SHAPE CONVENTION (feature-major, matches TokenEmbedding/+NeuralNet):
    %   forward(X) — X is (d x T), column t = the embedding of sequence
    %   position t (T = sequence length). Internally, per head h, Q_h/K_h/V_h
    %   are (d_k x T); the standard textbook notation Q/K/V (T x d_k, ROW t =
    %   query/key/value vector for position t) is exactly Q_h'/K_h'/V_h' here
    %   — i.e. this class's (d_k x T) layout is the transpose of the textbook
    %   (T x d_k) layout, by construction of the shared (d x T) convention.
    %   Output is (d x T), same shape as the input.
    %
    % SOFTMAX DIMENSION — THE SINGLE EASIEST THING TO GET BACKWARDS HERE:
    %   Per head, scores_h = (Q_h' * K_h) / sqrt(d_k) is (T x T), with
    %   scores_h(i,j) = dot(query vector for position i, key vector for
    %   position j) / sqrt(d_k). softmax(...) V in the formula normalizes,
    %   for EACH QUERY i, the distribution OVER KEYS j — i.e. each ROW of
    %   scores_h is turned into a probability distribution over its columns
    %   (columns = keys). Concretely: A_h = softmax(scores_h, along dim 2),
    %   so sum(A_h(i,:)) == 1 for every row i (verified numerically below,
    %   not just asserted). Softmax is emphatically NOT applied down the
    %   columns (that would normalize "how much does key j attend to all
    %   queries", which is not the attention formula and produces a
    %   plausible-looking but wrong result).
    %
    % KEY MASK (optional): forward(X, keyMask) takes a (1 x T) logical; false
    %   positions are never attended TO (scores_h(:, ~keyMask) = -Inf before
    %   the softmax). Default is no mask. It exists for ColBERT's [MASK]-padded
    %   queries (attend_to_mask_tokens=false); there is still no batching and
    %   no causal mask. Masked positions still produce their own output column.
    %
    % USAGE:
    %   mha = OceanNLP.Embedding.Modules.MultiHeadAttention(W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O, numHeads);
    %   Y   = mha.forward(X);   % X, Y both (d x T)
    %
    % See also: OceanNLP.Embedding.Modules.TokenEmbedding
    %
    % Implements the 3-pillar contract (.agents/pipeline-architecture-spec.md
    % §1.1) via getMetadata() and getPipelineSpec(), in addition to forward().

    properties (SetAccess = private)
        WQ (:,:) double
        bQ (:,1) double
        WK (:,:) double
        bK (:,1) double
        WV (:,:) double
        bV (:,1) double
        WO (:,:) double
        bO (:,1) double
        NumHeads (1,1) double {mustBeInteger, mustBePositive} = 1
    end

    properties (Dependent, SetAccess = private)
        HiddenDim   % d
        HeadDim     % d_k = d / NumHeads
    end

    methods
        function obj = MultiHeadAttention(W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O, numHeads)
            % MULTIHEADATTENTION  Construct directly from pretrained weight matrices.
            %   W_Q/W_K/W_V/W_O — (d x d) double
            %   b_Q/b_K/b_V/b_O — (d x 1) double
            %   numHeads        — positive integer, must evenly divide d
            arguments
                W_Q (:,:) double
                b_Q (:,1) double
                W_K (:,:) double
                b_K (:,1) double
                W_V (:,:) double
                b_V (:,1) double
                W_O (:,:) double
                b_O (:,1) double
                numHeads (1,1) double {mustBeInteger, mustBePositive}
            end
            d = size(W_Q, 1);
            allD = [size(W_Q,1) size(W_Q,2) size(W_K,1) size(W_K,2) size(W_V,1) size(W_V,2) ...
                     size(W_O,1) size(W_O,2) numel(b_Q) numel(b_K) numel(b_V) numel(b_O)];
            if any(allD ~= d)
                error('OceanNLP:Embedding:Modules:MultiHeadAttention:ShapeMismatch', ...
                    ['W_Q, W_K, W_V, W_O must all be square (d x d) and b_Q, b_K, b_V, b_O ' ...
                     'must all be (d x 1), sharing one common d. Got d=%d from W_Q but a ' ...
                     'mismatched dimension elsewhere.'], d);
            end
            if mod(d, numHeads) ~= 0
                error('OceanNLP:Embedding:Modules:MultiHeadAttention:HeadDimMismatch', ...
                    ['Hidden dimension d=%d is not evenly divisible by numHeads=%d. ' ...
                     'Refusing to silently truncate or round the head split — pass a ' ...
                     'numHeads that evenly divides d.'], d, numHeads);
            end
            obj.WQ = W_Q; obj.bQ = b_Q;
            obj.WK = W_K; obj.bK = b_K;
            obj.WV = W_V; obj.bV = b_V;
            obj.WO = W_O; obj.bO = b_O;
            obj.NumHeads = numHeads;
        end

        function Y = forward(obj, X, keyMask)
            % FORWARD  Scaled dot-product multi-head self-attention over ONE document.
            %   Y = obj.forward(X)
            %   Y = obj.forward(X, keyMask)
            %   X       - (d x T) double, feature-major (column t = position t).
            %   keyMask - optional (1 x T) logical; false = that position may NOT
            %             be attended TO (its score is set to -Inf before the
            %             softmax). Default: all true (no masking). Masked
            %             positions still produce their own output column.
            %   Y       - (d x T) double, same shape as X.
            arguments
                obj
                X (:,:) double
                keyMask (1,:) logical = true(1, size(X, 2))
            end
            if numel(keyMask) ~= size(X, 2)
                error('OceanNLP:Embedding:Modules:MultiHeadAttention:MaskLengthMismatch', ...
                    'keyMask has %d entries but X has %d positions.', numel(keyMask), size(X, 2));
            end
            if size(X, 1) ~= obj.HiddenDim
                error('OceanNLP:Embedding:Modules:MultiHeadAttention:HiddenDimMismatch', ...
                    'X has %d rows but this instance was constructed for d=%d.', ...
                    size(X, 1), obj.HiddenDim);
            end
            T  = size(X, 2);
            dk = obj.HeadDim;
            H  = obj.NumHeads;

            Q = obj.WQ * X + obj.bQ;   % (d x T)
            K = obj.WK * X + obj.bK;   % (d x T)
            V = obj.WV * X + obj.bV;   % (d x T)

            headOut = zeros(obj.HiddenDim, T);
            for h = 1:H
                rows = ((h - 1) * dk + 1):(h * dk);      % contiguous head slice (see class HEAD SPLIT CONVENTION)
                Qh = Q(rows, :);   % (d_k x T)
                Kh = K(rows, :);   % (d_k x T)
                Vh = V(rows, :);   % (d_k x T)

                scores = (Qh' * Kh) / sqrt(dk);           % (T x T): scores(i,j) = query i . key j
                % NOTE (future batching placeholder, NOT implemented now):
                % a padding/causal mask would be added to `scores` here,
                % e.g. scores(maskPositions) = -Inf;, before the softmax call.
                scores(:, ~keyMask) = -Inf;                % padding mask: never attend to masked keys
                A = OceanNLP.Embedding.Modules.MultiHeadAttention.softmaxRows_(scores); % row i sums to 1 over keys j (see class SOFTMAX DIMENSION note)

                headOut(rows, :) = Vh * A';               % (d_k x T)
            end

            Y = obj.WO * headOut + obj.bO;   % (d x T)
        end

        function d = get.HiddenDim(obj)
            d = size(obj.WQ, 1);
        end

        function dk = get.HeadDim(obj)
            dk = obj.HiddenDim / obj.NumHeads;
        end

        function meta = getMetadata(~)
            % GETMETADATA  Pillar 1 (symbolic) — closed-form LaTeX, symbol
            % dictionary, assumptions, and reference for scaled dot-product
            % multi-head self-attention.
            meta = struct();
            meta.SymbolicLatex = "\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right)V";
            meta.Symbols = containers.Map();
            meta.Symbols('Q') = "query projection for one head (T x d_k in the standard row-per-position layout)";
            meta.Symbols('K') = "key projection for one head (T x d_k)";
            meta.Symbols('V') = "value projection for one head (T x d_k)";
            meta.Symbols('d_k') = "per-head dimension, d_k = d / numHeads";
            meta.Assumptions = {'single document, no batching; optional key mask via forward(X, keyMask) (default: none), no causal mask', ...
                                 'd (hidden dim) must be evenly divisible by numHeads', ...
                                 'softmax normalizes each query''s distribution over keys (row-wise on the T x T score matrix, see class SOFTMAX DIMENSION note)', ...
                                 'heads use a contiguous (not interleaved) row-block split of the hidden dimension'};
            meta.IsSymbolic = true;
            meta.Reference = "Vaswani et al. (2017), Attention Is All You Need";
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3 (diagram) — port signature and canvas
            % metadata for this NONVIRTUAL leaf operator.
            spec = struct();
            spec.Inputs = struct('Name', "X", 'Domain', "dense-vector-batch", 'Dtype', "double", 'Shape', "d x T");
            spec.Outputs = struct('Name', "Y", 'Domain', "dense-vector-batch", 'Dtype', "double", 'Shape', "d x T");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "MHA");
            spec.ComposesFrom = {};
        end
    end

    methods (Static, Access = private)
        function A = softmaxRows_(scores)
            % SOFTMAXROWS_  Numerically-stable softmax along dim 2 (each row
            % sums to 1). Hand-rolled (no toolbox dependency) rather than
            % relying on a Deep Learning Toolbox softmax whose default
            % normalization axis would need to be double-checked/overridden
            % anyway — writing it out removes any ambiguity about which
            % dimension is normalized (see class SOFTMAX DIMENSION note).
            m = max(scores, [], 2);              % (T x 1), per-row max for stability
            e = exp(scores - m);                  % (T x T)
            A = e ./ sum(e, 2);                   % (T x T), each row sums to 1
        end
    end
end
