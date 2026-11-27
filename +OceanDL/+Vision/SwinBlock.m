classdef SwinBlock < OceanDL.nn.Module
    % SWINBLOCK  (Shifted-)window multi-head self-attention block (Liu et al.
    %   2021, Eq. 3), Pre-LN, on feature maps X4 = (D x N x Hp x Wp):
    %
    %     z = X4 + DropPath( unshift( reverse( W-MSA( partition( shift( LN1(X4) ))) + B )))
    %     y = z  + DropPath( MLP( LN2(z) ))
    %
    %   W-MSA: ordinary multi-head attention restricted to each M x M window.
    %   Shift = 0 in even blocks, M/2 in odd blocks (SW-MSA); shifted blocks add
    %   the region mask (WindowOps.shiftMask) so wrapped-around patches never mix.
    %   B: learned RELATIVE position bias -- one value per head for each of the
    %   (2M-1)^2 possible offsets between two tokens of a window, added to the
    %   attention scores (Swin uses no absolute position embedding).
    %   If the grid is not larger than the window, the window shrinks to the
    %   grid and no shift is applied (the Swin rule).

    properties (SetAccess = protected)
        Norm1
        Attention
        RelBiasTable        % Parameter ((2M-1)^2 x Heads)
        Norm2
        MLP
        WindowSize (1,1) double
        Shift (1,1) double
        DropPath (1,1) double
    end

    methods
        function obj = SwinBlock(dim, numHeads, windowSize, shift, opts)
            arguments
                dim (1,1) double {mustBePositive, mustBeInteger}
                numHeads (1,1) double {mustBePositive, mustBeInteger}
                windowSize (1,1) double {mustBePositive, mustBeInteger}
                shift (1,1) double {mustBeNonnegative, mustBeInteger} = 0
                opts.MLPRatio (1,1) double = 4
                opts.DropPath (1,1) double = 0
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            s = @(k) OceanDL.Vision.SwinBlock.offset(opts.Seed, k);
            obj.Norm1 = OceanDL.nn.LayerNorm(dim, 'DataType', opts.DataType);
            obj.Attention = OceanDL.Transformer.MultiHeadAttention(dim, numHeads, 'Seed', s(1), 'DataType', opts.DataType);
            obj.Norm2 = OceanDL.nn.LayerNorm(dim, 'DataType', opts.DataType);
            obj.MLP = OceanDL.Transformer.FeedForward(dim, round(opts.MLPRatio * dim), 'Seed', s(10), 'DataType', opts.DataType);
            if isempty(opts.Seed), z = randn((2*windowSize - 1)^2, numHeads);
            else, z = randn(RandStream('mt19937ar', 'Seed', s(20)), (2*windowSize - 1)^2, numHeads); end
            obj.RelBiasTable = OceanDL.nn.Parameter(cast(0.02 * z, opts.DataType));
            obj.WindowSize = windowSize;
            obj.Shift = shift;
            obj.DropPath = opts.DropPath;
        end

        function X4 = forward(obj, X4)
            [~, N, Hp, Wp] = size(X4, 1:4);
            M = obj.WindowSize; s = obj.Shift;
            if min(Hp, Wp) <= M                       % Swin rule: window no larger than the grid, no shift
                M = min(Hp, Wp); s = 0;
            end
            W = OceanDL.Vision.WindowOps;
            h = W.shift(obj.Norm1.forward(X4), s, -1);
            windows = W.partition(h, M);
            bias = obj.relativeBias(M);                                       % M^2 x M^2 x heads (Tensor)
            if s > 0
                shiftMask = OceanDL.Transformer.MultiHeadAttention.additive(W.shiftMask(Hp, Wp, M, s, N), ...
                    OceanDL.Autograd.Tensor.raw(windows));
                bias = bias + shiftMask;                                      % -> M^2 x M^2 x heads x (N nW)
            end
            a = obj.Attention.forward(windows, 'AttnMask', bias);
            a = W.shift(W.reverse(a, M, N, Hp, Wp), s, +1);
            F = OceanDL.Autograd.F;
            X4 = X4 + F.dropPath(a, obj.DropPath, obj.Training);
            X4 = X4 + F.dropPath(obj.MLP.forward(obj.Norm2.forward(X4)), obj.DropPath, obj.Training);
        end

        function B = relativeBias(obj, M)
            if M ~= obj.WindowSize
                % Shrunk window (grid smaller than WindowSize): recentre the table lookup.
                idxFull = OceanDL.Vision.WindowOps.relativeIndex(obj.WindowSize);
                sub = OceanDL.Vision.WindowOps.relativeIndex(M);
                [dr, dc] = ind2sub([2*M - 1, 2*M - 1], sub);
                idx = (dr + obj.WindowSize - M) + (2*obj.WindowSize - 1) * (dc + obj.WindowSize - M - 1);
                assert(all(ismember(idx(:), idxFull(:))));
            else
                idx = OceanDL.Vision.WindowOps.relativeIndex(M);
            end
            H = size(obj.RelBiasTable, 2);
            B = reshape(obj.RelBiasTable(idx(:), :), M*M, M*M, H);            % one bias per offset and head
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
