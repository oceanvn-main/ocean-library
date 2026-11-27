classdef Swin < OceanDL.nn.Module
    % SWIN  Swin Transformer classifier (Liu et al. 2021; Szeliski 2nd ed. Ch. 5).
    %
    %   image -> PatchEmbedding (p x p) -> feature map (D x N x Hp x Wp)
    %     -> stage 1: Depths(1) SwinBlocks (window M, shift 0, M/2, 0, ...)
    %     -> PatchMerging -> stage 2 (2D wide, half resolution) -> ...
    %     -> LayerNorm -> global average over the grid -> Linear logits
    %
    %   Attention is local (windows) so cost grows LINEARLY with image area,
    %   versus quadratically for ViT's global attention; shifted windows pass
    %   information across window borders, and the stage pyramid grows the
    %   receptive field like a CNN. No absolute positions (relative bias only),
    %   so any image size divisible by PatchSize * WindowSize * 2^(stages-1) works.

    properties (SetAccess = protected)
        Patch
        Stages cell = {}
        Merges cell = {}
        FinalNorm
        Head
    end

    methods
        function obj = Swin(opts)
            arguments
                opts.Channels (1,1) double = 3
                opts.PatchSize (1,1) double = 4
                opts.Dim (1,1) double = 48
                opts.Depths (1,:) double = [2 2]
                opts.Heads (1,:) double = [2 4]
                opts.WindowSize (1,1) double = 4
                opts.MLPRatio (1,1) double = 4
                opts.NumClasses (1,1) double = 10
                opts.DropPath (1,1) double = 0
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            seed = @(k) OceanDL.Vision.Swin.offset(opts.Seed, k);
            obj.Patch = OceanDL.Vision.PatchEmbedding(opts.Channels, opts.Dim, opts.PatchSize, 'Seed', seed(1), 'DataType', opts.DataType);
            rates = linspace(0, opts.DropPath, sum(opts.Depths));
            D = opts.Dim; k = 0;
            for st = 1:numel(opts.Depths)
                blocks = cell(1, opts.Depths(st));
                for b = 1:opts.Depths(st)
                    k = k + 1;
                    shift = mod(b - 1, 2) * floor(opts.WindowSize / 2);
                    blocks{b} = OceanDL.Vision.SwinBlock(D, opts.Heads(st), opts.WindowSize, shift, ...
                        'MLPRatio', opts.MLPRatio, 'DropPath', rates(k), 'Seed', seed(100 * k), 'DataType', opts.DataType);
                end
                obj.Stages{st} = OceanDL.nn.Sequential(blocks{:});
                if st < numel(opts.Depths)
                    obj.Merges{st} = OceanDL.Vision.PatchMerging(D, 'Seed', seed(50 + st), 'DataType', opts.DataType);
                    D = 2 * D;
                end
            end
            obj.FinalNorm = OceanDL.nn.LayerNorm(D, 'DataType', opts.DataType);
            obj.Head = OceanDL.nn.Linear(D, opts.NumClasses, 'Seed', seed(2), 'DataType', opts.DataType);
        end

        function logits = forward(obj, X)
            [T, grid] = obj.Patch.forward(X);                      % D x N x (Hp Wp), row index fastest
            [D, N] = size(T, 1:2);
            X4 = reshape(T, D, N, grid(1), grid(2));
            for st = 1:numel(obj.Stages)
                X4 = obj.Stages{st}.forward(X4);
                if st <= numel(obj.Merges)
                    X4 = obj.Merges{st}.forward(X4);
                end
            end
            X4 = obj.FinalNorm.forward(X4);
            [D, N] = size(X4, 1:2);
            logits = obj.Head.forward(reshape(mean(mean(X4, 3), 4), D, N));
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
