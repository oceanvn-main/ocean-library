classdef DecoderBlock < OceanDL.nn.Module
    % DECODERBLOCK  Transformer decoder layer (Vaswani et al. 2017, Sec. 3.1), pre-LN,
    %   general purpose: causal masked self-attention over the target prefix,
    %   cross-attention to an encoder memory, position-wise FFN.
    %
    %   blk = OceanDL.Transformer.DecoderBlock(dim, numHeads, ffHidden, 'Activation', "relu", ...
    %           'Dropout', 0, 'AttnDropout', 0, 'Epsilon', 1e-5, 'Seed', [], 'DataType', "double")
    %   y = blk.forward(x, memory, 'MemoryKeyMask', mm, 'TargetKeyMask', tm, 'Causal', true)
    %     x       (D x N x U)  target tokens ("CBT")
    %     memory  (D x N x T)  encoder output
    %     mm      (N x T) logical, true = encoder frame may be attended (padding mask)
    %     tm      (N x U) logical, true = target position may be attended ([] = all)
    %   blk.StoreAttention = true  ->  blk.LastCrossAttention / LastSelfAttention hold the
    %     raw (U x T x h x N) / (U x U x h x N) weights of the last forward call.
    %
    %   COMPUTATION (Pre-LN, Xiong et al. 2020):
    %     x1 = x  + Drop(SelfAttn(LN1(x);  causal mask C, target key mask))
    %     x2 = x1 + Drop(CrossAttn(LN2(x1), keys/values = memory; memory key mask))
    %     y  = x2 + Drop(FFN(LN3(x2))),   FFN(z) = W2 Drop(act(W1 z + b1)) + b2
    %   Causal mask C(u, v) = [v <= u]: position u never sees v > u, so output u is a
    %   function of x_1..x_u (and the memory) only -- teacher forcing with a shifted
    %   target trains every position in one pass, and step-by-step decoding of a prefix
    %   reproduces the teacher-forced outputs exactly.
    %   Semantics match torch.nn.TransformerDecoderLayer(norm_first=True, batch_first=False)
    %   with tgt_mask = causal, memory_key_padding_mask = ~mm (PyTorch: True = ignore);
    %   verified against it with copied weights (OceanTest fixture).
    %   A memory row with no valid frame (length-0 source) attends to all its frames
    %   instead of producing NaN (documented guard; such rows carry no information).
    %
    %   INCREMENTAL DECODING (key/value cache, inference only; Vaswani et al. 2017 decode
    %   one position at a time -- the cache is the standard implementation of that):
    %     mkv = blk.memoryKeyValues(memory)        struct K, V (dk x T x h x N): the cross-
    %           attention keys/values Wk m, Wv m, computed ONCE per utterance batch
    %     kv  = OceanDL.Transformer.DecoderBlock.emptyCache(dk, h, B)   K, V (dk x 0 x h x B)
    %     [y, kv] = blk.forwardStep(x, kv, mkv, 'MemoryKeyMask', mm)
    %       x (D x B x 1) = the NEW position u only; kv holds keys/values of LN1(x_1..x_{u-1})
    %       (per head), the new k_u, v_u are appended; the query of position u attends to
    %       keys 1..u, which is exactly row u of the causal mask, so y equals column u of
    %       forward(x_1..x_u, memory) (tested to ~1e-12). mkv / mm rows must already be
    %       selected for the B decoder rows (e.g. mkv.K(:, :, :, src)).
    %     Cost per step O(u) instead of re-running the O(u^2) prefix. The uncached
    %     forward remains the reference (training, teacher forcing, attention export).

    properties (SetAccess = protected)
        SelfAttention
        CrossAttention
        MLP
        Norm1
        Norm2
        Norm3
        Dim (1,1) double
        NumHeads (1,1) double
        DropoutP (1,1) double
    end

    properties
        StoreAttention (1,1) logical = false
        LastCrossAttention = []
        LastSelfAttention = []
    end

    methods
        function obj = DecoderBlock(dim, numHeads, ffHidden, opts)
            arguments
                dim (1,1) double {mustBePositive, mustBeInteger}
                numHeads (1,1) double {mustBePositive, mustBeInteger}
                ffHidden (1,1) double {mustBePositive, mustBeInteger} = 4 * dim
                opts.Activation (1,1) string {mustBeMember(opts.Activation, ["relu", "gelu"])} = "relu"
                opts.Dropout (1,1) double {mustBeGreaterThanOrEqual(opts.Dropout, 0), mustBeLessThan(opts.Dropout, 1)} = 0
                opts.AttnDropout (1,1) double {mustBeGreaterThanOrEqual(opts.AttnDropout, 0), mustBeLessThan(opts.AttnDropout, 1)} = 0
                opts.Epsilon (1,1) double {mustBePositive} = 1e-5
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            s = @(k) OceanDL.Transformer.DecoderBlock.offset(opts.Seed, k);
            dt = opts.DataType;
            obj.SelfAttention = OceanDL.Transformer.MultiHeadAttention(dim, numHeads, 'AttnDropout', opts.AttnDropout, ...
                'Seed', s(0), 'DataType', dt);
            obj.CrossAttention = OceanDL.Transformer.MultiHeadAttention(dim, numHeads, 'AttnDropout', opts.AttnDropout, ...
                'Seed', s(10), 'DataType', dt);
            obj.MLP = OceanDL.Transformer.FeedForward(dim, ffHidden, 'Activation', opts.Activation, ...
                'Dropout', opts.Dropout, 'Seed', s(20), 'DataType', dt);
            obj.Norm1 = OceanDL.nn.LayerNorm(dim, 'Epsilon', opts.Epsilon, 'DataType', dt);
            obj.Norm2 = OceanDL.nn.LayerNorm(dim, 'Epsilon', opts.Epsilon, 'DataType', dt);
            obj.Norm3 = OceanDL.nn.LayerNorm(dim, 'Epsilon', opts.Epsilon, 'DataType', dt);
            obj.Dim = dim;
            obj.NumHeads = numHeads;
            obj.DropoutP = opts.Dropout;
        end

        function y = forward(obj, x, memory, opts)
            arguments
                obj
                x
                memory
                opts.MemoryKeyMask = []
                opts.TargetKeyMask = []
                opts.Causal (1,1) logical = true
            end
            F = OceanDL.Autograd.F;
            x = OceanDL.Autograd.Tensor.wrap(x);
            [D, N, U] = size(x, 1:3);
            [Dm, Nm, T] = size(memory, 1:3);
            if D ~= obj.Dim || Dm ~= obj.Dim
                error('OceanDL:Transformer:DecoderBlock:Dim', ...
                    'Targets and memory need %d features on axis 1 (got %d and %d).', obj.Dim, D, Dm);
            end
            if Nm ~= N
                error('OceanDL:Transformer:DecoderBlock:Batch', 'Targets have N = %d, memory has N = %d.', N, Nm);
            end
            mm = OceanDL.Transformer.DecoderBlock.checkMask(opts.MemoryKeyMask, N, T, 'MemoryKeyMask');
            tm = OceanDL.Transformer.DecoderBlock.checkMask(opts.TargetKeyMask, N, U, 'TargetKeyMask');
            causal = [];
            if opts.Causal
                causal = tril(true(U, U));
            end
            obj.SelfAttention.StoreAttention = obj.StoreAttention;
            obj.CrossAttention.StoreAttention = obj.StoreAttention;
            drop = @(z) F.dropout(z, obj.DropoutP, obj.Training);
            x = x + drop(obj.SelfAttention.forward(obj.Norm1.forward(x), 'KeyMask', tm, 'AttnMask', causal));
            x = x + drop(obj.CrossAttention.forward(obj.Norm2.forward(x), 'KeyValue', memory, 'KeyMask', mm));
            y = x + drop(obj.MLP.forward(obj.Norm3.forward(x)));
            if obj.StoreAttention
                obj.LastSelfAttention = obj.SelfAttention.LastAttention;
                obj.LastCrossAttention = obj.CrossAttention.LastAttention;
            end
        end
    end

    methods
        function mkv = memoryKeyValues(obj, memory)
            % MEMORYKEYVALUES  Cross-attention keys / values of the memory, split by head.
            M = OceanDL.Autograd.Tensor.raw(memory);
            if size(M, 1) ~= obj.Dim
                error('OceanDL:Transformer:DecoderBlock:Dim', 'Memory needs %d features on axis 1 (got %d).', obj.Dim, size(M, 1));
            end
            ca = obj.CrossAttention;
            h = obj.NumHeads;
            mkv = struct('K', raw(OceanDL.Transformer.MultiHeadAttention.splitHeads(ca.Key.forward(M), h)), ...
                'V', raw(OceanDL.Transformer.MultiHeadAttention.splitHeads(ca.Value.forward(M), h)));
            if ~isempty(ca.KNorm)
                mkv.K = raw(ca.KNorm.forward(mkv.K));
            end
        end

        function [y, kv] = forwardStep(obj, x, kv, mkv, opts)
            % FORWARDSTEP  One new target position with cached self-attention keys/values.
            %   x (D x B x 1), kv.K/kv.V (dk x u-1 x h x B), mkv.K/mkv.V (dk x T x h x B),
            %   MemoryKeyMask (B x T) -> y (D x B x 1), kv with position u appended.
            arguments
                obj
                x
                kv (1,1) struct
                mkv (1,1) struct
                opts.MemoryKeyMask = []
            end
            if obj.Training
                error('OceanDL:Transformer:DecoderBlock:StepTraining', ...
                    'forwardStep is an inference path (call eval()); train with forward.');
            end
            x = raw(x);
            [D, B] = size(x, 1:2);
            if D ~= obj.Dim || size(x, 3) ~= 1
                error('OceanDL:Transformer:DecoderBlock:StepSize', 'forwardStep takes x of size %d x B x 1.', obj.Dim);
            end
            T = size(mkv.K, 2);
            if size(kv.K, 4) ~= B || size(mkv.K, 4) ~= B
                error('OceanDL:Transformer:DecoderBlock:Batch', 'Cache rows (%d, %d) do not match B = %d.', size(kv.K, 4), size(mkv.K, 4), B);
            end
            mm = OceanDL.Transformer.DecoderBlock.checkMask(opts.MemoryKeyMask, B, T, 'MemoryKeyMask');
            split = @(z) OceanDL.Transformer.MultiHeadAttention.splitHeads(z, obj.NumHeads);
            % self-attention over the cached prefix plus the new position
            sa = obj.SelfAttention;
            z = raw(obj.Norm1.forward(x));
            [q, k] = sa.normalizeQK(split(raw(sa.Query.forward(z))), split(raw(sa.Key.forward(z))));
            kv.K = cat(2, kv.K, raw(k));
            kv.V = cat(2, kv.V, split(raw(sa.Value.forward(z))));
            o = OceanDL.Transformer.MultiHeadAttention.headAttention(tensor(q), kv.K, kv.V);
            x = x + raw(sa.Output.forward(OceanDL.Transformer.MultiHeadAttention.mergeHeads(o)));
            % cross-attention to the precomputed memory keys / values
            ca = obj.CrossAttention;
            q = split(raw(ca.Query.forward(raw(obj.Norm2.forward(x)))));
            if ~isempty(ca.QNorm), q = raw(ca.QNorm.forward(q)); end
            o = OceanDL.Transformer.MultiHeadAttention.headAttention(tensor(q), mkv.K, mkv.V, 'KeyMask', mm);
            x = x + raw(ca.Output.forward(OceanDL.Transformer.MultiHeadAttention.mergeHeads(o)));
            y = x + raw(obj.MLP.forward(raw(obj.Norm3.forward(x))));
        end
    end

    methods (Static)
        function kv = emptyCache(headDim, numHeads, B, like)
            % EMPTYCACHE  Self-attention cache for B rows with no position yet.
            if nargin < 4, like = 0; end
            kv = struct('K', zeros(headDim, 0, numHeads, B, 'like', like), 'V', zeros(headDim, 0, numHeads, B, 'like', like));
        end
    end

    methods (Static, Access = private)
        function m = checkMask(m, N, T, name)
            if isempty(m)
                m = [];
                return;
            end
            m = logical(m);
            if ~isequal(size(m), [N T])
                error('OceanDL:Transformer:DecoderBlock:Mask', '%s must be N x T = %d x %d.', name, N, T);
            end
            m(~any(m, 2), :) = true;                 % no valid key at all: finite instead of NaN
        end

        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end

function r = raw(t)
r = OceanDL.Autograd.Tensor.raw(t);
end

function t = tensor(x)
% headAttention needs a Tensor operand: on a plain double, softmax(S, 2) would resolve to
% the legacy Deep Learning Toolbox transfer function (column-wise, ignores the dim).
t = OceanDL.Autograd.Tensor.wrap(raw(x));
end
