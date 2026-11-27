classdef ConformerBlock < OceanDL.nn.Module
    % CONFORMERBLOCK  Convolution-augmented Transformer block (Gulati et al. 2020,
    %   "Conformer", Sec. 2 / Fig. 1) on "CBT" sequences (D x N x T), general purpose.
    %
    %   blk = OceanDL.Transformer.ConformerBlock(dim, numHeads, 'FFMult', 4, 'ConvKernel', 15, ...
    %           'Dropout', 0.1, 'AttnDropout', 0, 'PosEnc', "rope", 'Seed', [], 'DataType', "double")
    %   y = blk.forward(x, 'KeyMask', m)      x (D x N x T), m (N x T) logical, true = valid frame
    %
    %   COMPUTATION (macaron structure, every module pre-normed and residual):
    %     x1 = x  + 1/2 FFN1(x)                    FFN(z) = Drop(W2 Drop(Swish(W1 LN(z) + b1)) + b2)
    %     x2 = x1 + Drop(MHSA(LN(x1); key mask))   scaled dot-product self-attention, h heads
    %     x3 = x2 + Conv(x2)
    %     x4 = x3 + 1/2 FFN2(x3)
    %     y  = LN(x4)
    %   CONVOLUTION MODULE (on z = x2):
    %     a = Pw1(LN(z))  (D -> 2D, kernel 1);   g = a_{1:D} .* sigmoid(a_{D+1:2D})       (GLU)
    %     g = M .* g      (padded frames zeroed BEFORE the temporal convolution)
    %     c = DwConv(g)   (depthwise: one length-k filter per channel, padding (k-1)/2, stride 1)
    %     Conv(z) = Drop(Pw2(Swish(LN_c(c))))     (D -> D, kernel 1)
    %   Swish(u) = u .* sigmoid(u) (= SiLU).
    %
    %   DEVIATIONS FROM THE PAPER (documented choices):
    %   * BatchNorm after the depthwise convolution is replaced by LayerNorm over the
    %     channels of each frame (LN_c above). Reason: BatchNorm statistics mix every
    %     utterance AND every padded frame of the mini-batch, so a frame's output would
    %     depend on what else is in the batch (breaking padded == per-utterance), and it
    %     behaves differently in train and eval mode with the small CPU batches used here.
    %     LayerNorm is per frame, identical in train/eval and padding-invariant
    %     (also offered by ESPnet / WeNet as the conv-module norm option).
    %   * Positions: the paper uses Transformer-XL relative positional self-attention.
    %     Here PosEnc "rope" (default) applies rotary position embeddings (Su et al. 2021,
    %     OceanDL.Transformer.RoPE, "half" pairing) to the per-head queries and keys, so
    %     the scores depend only on the relative offset t1 - t2 -- the same relative
    %     property, without the extra content/position bias parameters, and positions
    %     restart at 0 for every utterance (right padding keeps them exact).
    %     PosEnc "none": no position inside the block (e.g. when the caller adds an
    %     absolute sinusoidal encoding to the input; the depthwise conv also carries
    %     relative order).
    %
    %   PADDING: keys at masked frames get -Inf scores (never attended), and the GLU
    %   output is zeroed on masked frames before the depthwise convolution, so a valid
    %   frame's output equals the output of its utterance run alone (all other
    %   operations are per frame). Masked frames of y are zeroed. A row with no valid
    %   frame at all (length-0 utterance) attends to all its (zero) frames instead of
    %   producing NaN; its outputs are masked to zero anyway.

    properties (SetAccess = protected)
        FF1In           % Linear D -> FFMult*D  (first half-step FFN)
        FF1Out          % Linear FFMult*D -> D
        FF2In
        FF2Out
        NormFF1
        NormFF2
        NormAttn
        Attention
        NormConv
        Pointwise1
        Depthwise
        ConvNorm
        Pointwise2
        NormOut
        Dim (1,1) double
        NumHeads (1,1) double
        ConvKernel (1,1) double
        PosEnc (1,1) string
        DropoutP (1,1) double
    end

    methods
        function obj = ConformerBlock(dim, numHeads, opts)
            arguments
                dim (1,1) double {mustBePositive, mustBeInteger}
                numHeads (1,1) double {mustBePositive, mustBeInteger}
                opts.FFMult (1,1) double {mustBePositive} = 4
                opts.ConvKernel (1,1) double {mustBePositive, mustBeInteger} = 15
                opts.Dropout (1,1) double {mustBeGreaterThanOrEqual(opts.Dropout, 0), mustBeLessThan(opts.Dropout, 1)} = 0
                opts.AttnDropout (1,1) double {mustBeGreaterThanOrEqual(opts.AttnDropout, 0), mustBeLessThan(opts.AttnDropout, 1)} = 0
                opts.PosEnc (1,1) string {mustBeMember(opts.PosEnc, ["rope", "none"])} = "rope"
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            if mod(opts.ConvKernel, 2) == 0
                error('OceanDL:Transformer:ConformerBlock:EvenKernel', ...
                    'ConvKernel must be odd so "same" padding (k-1)/2 keeps the length (got %d).', opts.ConvKernel);
            end
            if opts.PosEnc == "rope" && mod(dim / numHeads, 2)
                error('OceanDL:Transformer:ConformerBlock:RopeDim', ...
                    'RoPE needs an even head dimension (Dim/NumHeads = %g).', dim / numHeads);
            end
            s = @(k) OceanDL.Transformer.ConformerBlock.offset(opts.Seed, k);
            hidden = round(opts.FFMult * dim);
            dt = opts.DataType;
            ln = @() OceanDL.nn.LayerNorm(dim, 'DataType', dt);
            obj.FF1In = OceanDL.nn.Linear(dim, hidden, 'Seed', s(1), 'DataType', dt);
            obj.FF1Out = OceanDL.nn.Linear(hidden, dim, 'Seed', s(2), 'DataType', dt);
            obj.FF2In = OceanDL.nn.Linear(dim, hidden, 'Seed', s(11), 'DataType', dt);
            obj.FF2Out = OceanDL.nn.Linear(hidden, dim, 'Seed', s(12), 'DataType', dt);
            obj.NormFF1 = ln(); obj.NormFF2 = ln(); obj.NormAttn = ln(); obj.NormConv = ln();
            obj.ConvNorm = ln(); obj.NormOut = ln();
            obj.Attention = OceanDL.Transformer.MultiHeadAttention(dim, numHeads, 'AttnDropout', opts.AttnDropout, ...
                'Seed', s(20), 'DataType', dt);
            obj.Pointwise1 = OceanDL.nn.Conv1d(dim, 2 * dim, 1, 'Seed', s(30), 'DataType', dt);
            obj.Depthwise = OceanDL.nn.Conv1d(dim, dim, opts.ConvKernel, 'Padding', (opts.ConvKernel - 1) / 2, ...
                'Groups', dim, 'Seed', s(31), 'DataType', dt);
            obj.Pointwise2 = OceanDL.nn.Conv1d(dim, dim, 1, 'Seed', s(32), 'DataType', dt);
            obj.Dim = dim;
            obj.NumHeads = numHeads;
            obj.ConvKernel = opts.ConvKernel;
            obj.PosEnc = opts.PosEnc;
            obj.DropoutP = opts.Dropout;
        end

        function y = forward(obj, x, opts)
            arguments
                obj
                x
                opts.KeyMask = []
            end
            F = OceanDL.Autograd.F;
            x = OceanDL.Autograd.Tensor.wrap(x);
            [D, N, T] = size(x, 1:3);
            if D ~= obj.Dim
                error('OceanDL:Transformer:ConformerBlock:Dim', 'Expected %d features on axis 1 ("CBT"), got %d.', obj.Dim, D);
            end
            M = opts.KeyMask;
            if isempty(M)
                M = true(N, T);
            end
            M = logical(M);
            if ~isequal(size(M), [N T])
                error('OceanDL:Transformer:ConformerBlock:KeyMask', 'KeyMask must be N x T = %d x %d.', N, T);
            end
            attnMask = M;
            attnMask(~any(M, 2), :) = true;                  % empty rows: finite (masked) output instead of NaN
            drop = @(z) F.dropout(z, obj.DropoutP, obj.Training);
            % 1/2 FFN
            x = x + 0.5 * drop(obj.FF1Out.forward(drop(F.silu(obj.FF1In.forward(obj.NormFF1.forward(x))))));
            % MHSA
            args = {'KeyMask', attnMask};
            if obj.PosEnc == "rope"
                args = [args, {'Rope', OceanDL.Transformer.RoPE.sequence(T, obj.Dim / obj.NumHeads)}];
            end
            x = x + drop(obj.Attention.forward(obj.NormAttn.forward(x), args{:}));
            % convolution module
            a = obj.Pointwise1.forward(obj.NormConv.forward(x));                % 2D x N x T
            g = reshape(a(1:D, :), D, N, T) .* sigmoid(reshape(a(D+1:2*D, :), D, N, T));
            g = OceanDL.Layout.SequenceMask.apply(g, M);
            c = obj.Depthwise.forward(g);
            x = x + drop(obj.Pointwise2.forward(F.silu(obj.ConvNorm.forward(c))));
            % 1/2 FFN
            x = x + 0.5 * drop(obj.FF2Out.forward(drop(F.silu(obj.FF2In.forward(obj.NormFF2.forward(x))))));
            y = OceanDL.Layout.SequenceMask.apply(obj.NormOut.forward(x), M);
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
