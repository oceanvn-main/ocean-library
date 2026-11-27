classdef MultiHeadAttention < OceanDL.nn.Module
    % MULTIHEADATTENTION  Scaled dot-product multi-head attention (Vaswani et
    %   al. 2017, Sec. 3.2), trainable, batched, domain-agnostic: tokens may be
    %   words, image patches, time windows...
    %
    %   Y = mha.forward(X)                              self-attention
    %   Y = mha.forward(X, 'KeyValue', Z)               cross-attention (queries X, keys/values Z)
    %   [Y, A] = mha.forward(X, 'KeyMask', m, 'AttnMask', M)
    %     X  (D x N x Tq)  "CBT" tokens;  Z (D x N x Tk)
    %     m  (N x Tk) logical, true = key may be attended (padding mask)
    %     M  (Tq x Tk) logical (true = allowed, e.g. causal / window) or additive numeric
    %     Y  (D x N x Tq);  A (Tq x Tk x h x N) attention weights (rows sum to 1)
    %
    % COMPUTATION, per head j (d_k = D / h features each, CONTIGUOUS rows
    %   (j-1) d_k + 1 : j d_k -- the same split as PyTorch and the legacy
    %   OceanNLP BERT attention, so their weights load unchanged):
    %     Q_j = Wq_j X,  K_j = Wk_j Z,  V_j = Wv_j Z                (d_k x T)
    %     S_j = Q_j' K_j / sqrt(d_k)  (+ masks: -Inf where not allowed)   (Tq x Tk)
    %     A_j = softmax(S_j) over keys  -- row q = how query q distributes attention
    %     O_j = V_j A_j'                                              (d_k x Tq)
    %     Y   = Wo [O_1; ...; O_h] + bo
    %   All (head, sample) pairs are PAGES of one pagemtimes call
    %   (BatchMatMul), never a loop. The 1/sqrt(d_k) keeps score variance ~1
    %   so the softmax does not saturate as d_k grows.
    %
    % MODERN OPTIONS (off by default; used by DiT / MMDiT / FLUX-style blocks):
    %   'QKNorm' "rms" | "layer": normalise each head's queries and keys over
    %     d_k before the dot product (Dehghani et al. 2023) -- bounds the
    %     attention logits, stabilising large / low-precision training.
    %   forward(..., 'Rope', r): rotary position embedding applied to the
    %     per-head queries and keys, r = OceanDL.Transformer.RoPE.axial2d(...)
    %     (Su et al. 2021; 2-D axial variant) -- relative positions enter the
    %     attention scores directly, no additive embedding needed.
    %
    % REUSABLE CORE (MMDiT joint attention): splitHeads -> headAttention ->
    %   mergeHeads operate on already-projected tensors.

    properties (SetAccess = protected)
        Query
        Key
        Value
        Output
        QNorm = []          % [] or RMSNorm / LayerNorm over d_k
        KNorm = []
        NumHeads (1,1) double
        EmbedDim (1,1) double
        HeadDim (1,1) double
        AttnDropout (1,1) double = 0
    end

    properties
        StoreAttention (1,1) logical = false
        LastAttention = []            % raw (Tq x Tk x h x N) weights when StoreAttention
    end

    methods
        function obj = MultiHeadAttention(embedDim, numHeads, opts)
            arguments
                embedDim (1,1) double {mustBePositive, mustBeInteger}
                numHeads (1,1) double {mustBePositive, mustBeInteger}
                opts.AttnDropout (1,1) double = 0
                opts.Bias (1,1) logical = true
                opts.QKNorm (1,1) string {mustBeMember(opts.QKNorm, ["none", "rms", "layer"])} = "none"
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            if mod(embedDim, numHeads)
                error('OceanDL:Transformer:MultiHeadAttention:HeadSplit', ...
                    'EmbedDim %d is not divisible by NumHeads %d.', embedDim, numHeads);
            end
            seed = @(k) OceanDL.Transformer.MultiHeadAttention.offset(opts.Seed, k);
            lin = @(k) OceanDL.nn.Linear(embedDim, embedDim, 'Bias', opts.Bias, 'Seed', seed(k), 'DataType', opts.DataType);
            obj.Query = lin(1); obj.Key = lin(2); obj.Value = lin(3); obj.Output = lin(4);
            obj.NumHeads = numHeads;
            obj.EmbedDim = embedDim;
            obj.HeadDim = embedDim / numHeads;
            obj.AttnDropout = opts.AttnDropout;
            dk = embedDim / numHeads;
            switch opts.QKNorm
                case "rms"
                    obj.QNorm = OceanDL.nn.RMSNorm(dk, 'DataType', opts.DataType);
                    obj.KNorm = OceanDL.nn.RMSNorm(dk, 'DataType', opts.DataType);
                case "layer"
                    obj.QNorm = OceanDL.nn.LayerNorm(dk, 'DataType', opts.DataType);
                    obj.KNorm = OceanDL.nn.LayerNorm(dk, 'DataType', opts.DataType);
            end
        end

        function [Y, A] = forward(obj, X, opts)
            arguments
                obj
                X
                opts.KeyValue = []
                opts.KeyMask = []
                opts.AttnMask = []
                opts.Rope = []                % struct from OceanDL.Transformer.RoPE (self-attention)
            end
            Z = opts.KeyValue;
            if isempty(Z), Z = X; end
            [D, N, Tq] = size(X, 1:3);
            Tk = size(Z, 3);
            if D ~= obj.EmbedDim
                error('OceanDL:Transformer:MultiHeadAttention:EmbedDim', ...
                    'Expected %d features on axis 1 ("CBT"), got %d.', obj.EmbedDim, D);
            end
            h = obj.NumHeads;
            [Q, K] = obj.normalizeQK(OceanDL.Transformer.MultiHeadAttention.splitHeads(obj.Query.forward(X), h), OceanDL.Transformer.MultiHeadAttention.splitHeads(obj.Key.forward(Z), h));
            if ~isempty(opts.Rope)
                Q = OceanDL.Transformer.RoPE.apply(Q, opts.Rope);
                K = OceanDL.Transformer.RoPE.apply(K, opts.Rope);
            end
            V = OceanDL.Transformer.MultiHeadAttention.splitHeads(obj.Value.forward(Z), h);
            [O, A] = OceanDL.Transformer.MultiHeadAttention.headAttention(Q, K, V, 'KeyMask', opts.KeyMask, 'AttnMask', opts.AttnMask, ...
                'Dropout', obj.AttnDropout, 'Training', obj.Training);
            if obj.StoreAttention
                obj.LastAttention = OceanDL.Autograd.Tensor.raw(A);
            end
            Y = obj.Output.forward(OceanDL.Transformer.MultiHeadAttention.mergeHeads(O));
        end

        function [Q, K] = normalizeQK(obj, Q, K)
            % Per-head QK-norm over d_k (axis 1 of dk x T x h x N); identity when off.
            if isempty(obj.QNorm), return; end
            Q = obj.QNorm.forward(Q);
            K = obj.KNorm.forward(K);
        end
    end

    methods (Static)
        function T = splitHeads(X, h)
            % (D x N x T) -> (dk x T x h x N), heads = contiguous row blocks.
            [D, N, T_] = size(X, 1:3);
            T = permute(reshape(X, D / h, h, N, T_), [1 4 2 3]);
        end

        function X = mergeHeads(O)
            % (dk x T x h x N) -> (D x N x T)
            [dk, T_, h, N] = size(O, 1:4);
            X = reshape(permute(O, [1 3 4 2]), dk * h, N, T_);
        end

        function [O, A] = headAttention(Q, K, V, opts)
            % Scaled dot-product attention on split heads: Q (dk x Tq x h x N),
            % K, V (dk x Tk x h x N) -> O (dk x Tq x h x N), A (Tq x Tk x h x N).
            arguments
                Q
                K
                V
                opts.KeyMask = []
                opts.AttnMask = []
                opts.Dropout (1,1) double = 0
                opts.Training (1,1) logical = false
            end
            dk = size(Q, 1); Tk = size(K, 2); N = size(Q, 4);
            S = pagemtimes(Q, 'transpose', K, 'none') * (1 / sqrt(dk));          % Tq x Tk x h x N
            like = OceanDL.Autograd.Tensor.raw(S);
            if ~isempty(opts.KeyMask)
                allowed = reshape(logical(opts.KeyMask).', 1, Tk, 1, N);
                S = S + OceanDL.Transformer.MultiHeadAttention.additive(allowed, like);
            end
            if ~isempty(opts.AttnMask)
                M = opts.AttnMask;
                if islogical(M)
                    M = OceanDL.Transformer.MultiHeadAttention.additive(M, like);
                end
                S = S + M;
            end
            % explicit F.softmax: with raw (non-Tensor) inputs a bare softmax(S, 2) would
            % resolve to the Deep Learning Toolbox function, which ignores DIM
            A = OceanDL.Autograd.F.softmax(S, 2);                                % over keys
            Ad = OceanDL.Autograd.F.dropout(A, opts.Dropout, opts.Training);
            O = pagemtimes(V, 'none', Ad, 'transpose');                          % dk x Tq x h x N
        end

        function M = additive(allowed, like)
            % logical "may attend" -> additive mask: 0 where allowed, -Inf elsewhere.
            M = zeros(size(allowed), 'like', like);
            M(~allowed) = -Inf;
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
