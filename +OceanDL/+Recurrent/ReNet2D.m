classdef ReNet2D < OceanDL.nn.Module
    % RENET2D  2-D recurrent layer over images (ReNet: Visin et al. 2015),
    %   the image extension of bidirectional RNNs in Goodfellow Sec. 10.3
    %   (four directions: up, down, left, right).
    %
    %   1. PATCHES: split the H x W image into non-overlapping p x p patches;
    %      each patch (p*p*C values) is one input vector -> grid Hp x Wp.
    %   2. VERTICAL SWEEP: every column of the patch grid is a sequence of
    %      length Hp; a bidirectional RNN (down + up) reads all Wp*N columns
    %      in parallel (they are just more batch entries).
    %      -> each position now summarises its whole COLUMN.
    %   3. HORIZONTAL SWEEP: every row of that feature map is a sequence of
    %      length Wp read by a second bidirectional RNN (right + left).
    %      -> each position now summarises the WHOLE IMAGE (column context
    %         carried along the row), a receptive field a small CNN only
    %         reaches after many layers.
    %   Output: canonical feature map (Hp x Wp x 2*HiddenSize2 x N).
    %
    %   Every step is permute/reshape of the canonical tensors into
    %   "CBT" sequences and back -- no special kernels, full autograd.

    properties (SetAccess = protected)
        PatchSize (1,1) double
        Vertical          % Bidirectional over patch-grid columns
        Horizontal        % Bidirectional over patch-grid rows
    end

    methods
        function obj = ReNet2D(inChannels, patchSize, hidden1, hidden2, opts)
            arguments
                inChannels (1,1) double {mustBePositive, mustBeInteger}
                patchSize (1,1) double {mustBePositive, mustBeInteger}
                hidden1 (1,1) double {mustBePositive, mustBeInteger}
                hidden2 (1,1) double {mustBePositive, mustBeInteger}
                opts.CellType (1,1) string {mustBeMember(opts.CellType, ["gru", "lstm", "elman"])} = "gru"
                opts.Seed = []
            end
            obj.PatchSize = patchSize;
            D1 = patchSize^2 * inChannels;
            obj.Vertical = OceanDL.Recurrent.Bidirectional( ...
                OceanDL.Recurrent.ReNet2D.cellFactory(opts.CellType, D1, hidden1, opts.Seed));
            obj.Horizontal = OceanDL.Recurrent.Bidirectional( ...
                OceanDL.Recurrent.ReNet2D.cellFactory(opts.CellType, 2*hidden1, hidden2, opts.Seed));
        end

        function Y = forward(obj, X)
            [H, W, C, N] = size(X, 1:4);
            p = obj.PatchSize;
            if mod(H, p) || mod(W, p)
                error('OceanDL:Recurrent:ReNet2D:PatchSize', ...
                    'Image %dx%d is not divisible into %dx%d patches.', H, W, p, p);
            end
            Hp = H / p; Wp = W / p;

            % 1. Patches: (p,Hp,p,Wp,C,N) -> (p,p,C,Hp,Wp,N) -> F x Hp x Wp x N
            P = reshape(X, p, Hp, p, Wp, C, N);
            P = permute(P, [1 3 5 2 4 6]);
            F = p * p * C;
            P = reshape(P, F, Hp, Wp, N);

            % 2. Vertical sweep: sequences run along Hp; batch = (Wp, N)
            S = reshape(permute(P, [1 3 4 2]), F, Wp*N, Hp);
            V = obj.Vertical.forward(S);                         % (2H1) x (Wp*N) x Hp
            D1 = size(V, 1);
            V = permute(reshape(V, D1, Wp, N, Hp), [1 4 2 3]);   % 2H1 x Hp x Wp x N

            % 3. Horizontal sweep: sequences run along Wp; batch = (Hp, N)
            S = reshape(permute(V, [1 2 4 3]), D1, Hp*N, Wp);
            Z = obj.Horizontal.forward(S);                       % (2H2) x (Hp*N) x Wp
            D2 = size(Z, 1);
            Y = permute(reshape(Z, D2, Hp, N, Wp), [2 4 1 3]);   % Hp x Wp x 2H2 x N  ("SSCB")
        end
    end

    methods (Static, Access = private)
        function f = cellFactory(type, inSize, hidden, seed)
            switch type
                case "gru",   f = @() OceanDL.Recurrent.GRUCell(inSize, hidden, 'Seed', seed);
                case "lstm",  f = @() OceanDL.Recurrent.LSTMCell(inSize, hidden, 'Seed', seed);
                case "elman", f = @() OceanDL.Recurrent.ElmanCell(inSize, hidden, 'Seed', seed);
            end
        end
    end
end
