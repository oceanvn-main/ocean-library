classdef Upsample < OceanDL.Autograd.Function
    % UPSAMPLE  Nearest-neighbour upsampling of the two spatial dims of
    %   X (H x W x C x N) by integer factors [sh sw]:
    %     Y(i, j, :, :) = X(ceil(i/sh), ceil(j/sw), :, :)     (PyTorch mode="nearest")
    %
    %   VJP: every input pixel was copied to an sh x sw block, so its gradient
    %   is the SUM of g over that block:
    %     dL/dX(r, c) = sum_{a<sh, b<sw} g(sh(r-1)+a, sw(c-1)+b)
    %
    %   Nearest + conv is the upsampling used by modern conv decoders (Stable
    %   Diffusion's AutoencoderKL): no checkerboard artefacts, unlike a
    %   stride-2 transposed convolution with a kernel not divisible by 2.

    properties (SetAccess = private)
        Scale (1,2) double = [2 2]
    end

    methods
        function obj = Upsample(scale)
            arguments
                scale (1,:) double {mustBePositive, mustBeInteger} = 2
            end
            obj.Scale = scale .* [1 1];
        end

        function Y = forward(obj, X)
            Y = repelem(X, obj.Scale(1), obj.Scale(2), 1, 1);
        end

        function grads = backward(obj, g)
            [Ho, Wo, C, N] = size(g, 1:4);
            s = obj.Scale;
            g = reshape(g, s(1), Ho / s(1), s(2), Wo / s(2), C, N);
            grads = {reshape(sum(sum(g, 1), 3), Ho / s(1), Wo / s(2), C, N)};
        end
    end
end
