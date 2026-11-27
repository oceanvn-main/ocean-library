classdef PatchEmbedding < OceanDL.nn.Module
    % PATCHEMBEDDING  Image -> tokens (Dosovitskiy et al. 2021, Eq. 1).
    %   Cut the H x W image into non-overlapping p x p patches, flatten each
    %   patch (p*p*C values) and project it linearly to D features:
    %     token_t = E vec(patch_t) + b,   E: (D x p^2 C)
    %   "Flatten + linear" of non-overlapping patches is EXACTLY a convolution
    %   with kernel p and stride p, which is how it is computed here
    %   (nn.Conv2d; verified equal to the explicit patchify + matrix product).
    %
    %   X (H x W x C x N, "SSCB") -> tokens (D x N x T, "CBT"), T = (H/p)(W/p).
    %   TOKEN ORDER: column-major over the patch grid, t = r + Hp*(c-1)
    %   (row index fastest) -- MATLAB's natural order, the order assumed by
    %   PositionalEncoding.sincos2d and by attention-map visualisation.

    properties (SetAccess = protected)
        Proj                  % nn.Conv2d, kernel = stride = PatchSize
        PatchSize (1,2) double
        Dim (1,1) double
    end

    methods
        function obj = PatchEmbedding(inChannels, dim, patchSize, opts)
            arguments
                inChannels (1,1) double {mustBePositive, mustBeInteger}
                dim (1,1) double {mustBePositive, mustBeInteger}
                patchSize (1,:) double {mustBePositive, mustBeInteger}
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            p = patchSize .* [1 1];
            obj.PatchSize = p;
            obj.Dim = dim;
            obj.Proj = OceanDL.nn.Conv2d(inChannels, dim, p, 'Stride', p, 'Seed', opts.Seed, 'DataType', opts.DataType);
        end

        function [tokens, grid] = forward(obj, X)
            [H, W] = size(X, 1:2);
            if any(mod([H W], obj.PatchSize))
                error('OceanDL:Vision:PatchEmbedding:Size', 'Image %dx%d is not divisible into %dx%d patches.', ...
                    H, W, obj.PatchSize(1), obj.PatchSize(2));
            end
            Fm = obj.Proj.forward(X);                             % Hp x Wp x D x N
            [Hp, Wp, D, N] = size(Fm, 1:4);
            tokens = reshape(permute(Fm, [3 4 1 2]), D, N, Hp * Wp);   % D x N x (Hp Wp), r fastest
            grid = [Hp Wp];
        end
    end
end
