classdef GroupNorm < OceanDL.Autograd.Function
    % GROUPNORM  Normalisation part of group normalisation (Wu & He 2018) on
    %   X (H x W x C x N): the C channels are split into G contiguous groups
    %   and each (group, sample) is standardised over its H*W*C/G values:
    %     mu = mean(x),  var = mean((x - mu).^2)   (biased, as PyTorch)
    %     Y  = (x - mu) / sqrt(var + eps)           (eps INSIDE the sqrt)
    %   The per-channel affine (gamma, beta) is applied outside (F.groupNorm).
    %
    %   VJP (same form as layer norm, over each group's m values):
    %     dL/dx = ( g - mean(g) - Y .* mean(g .* Y) ) / sqrt(var + eps)
    %
    %   Unlike batch norm it does not depend on the batch (identical in
    %   training and evaluation, fine at batch size 1); G = 1 is layer norm
    %   over (H, W, C), G = C is instance norm.

    properties (SetAccess = private)
        Groups (1,1) double
        Epsilon (1,1) double
    end

    methods
        function obj = GroupNorm(groups, epsilon)
            arguments
                groups (1,1) double {mustBePositive, mustBeInteger}
                epsilon (1,1) double {mustBePositive} = 1e-5
            end
            obj.Groups = groups;
            obj.Epsilon = epsilon;
        end

        function Y = forward(obj, X)
            sz = size(X, 1:4);
            G = obj.Groups;
            if mod(sz(3), G) ~= 0
                error('OceanDL:Autograd:GroupNorm:Groups', '%d channels are not divisible into %d groups.', sz(3), G);
            end
            Z = reshape(X, sz(1) * sz(2) * sz(3) / G, G, sz(4));    % contiguous channels share a column
            mu = mean(Z, 1);
            Zc = Z - mu;
            sigma = sqrt(mean(Zc .^ 2, 1) + obj.Epsilon);
            Yz = Zc ./ sigma;
            obj.Saved.Y = Yz; obj.Saved.Sigma = sigma; obj.Saved.Size = sz;
            Y = reshape(Yz, sz);
        end

        function grads = backward(obj, g)
            Yz = obj.Saved.Y;
            gz = reshape(g, size(Yz));
            dz = (gz - mean(gz, 1) - Yz .* mean(gz .* Yz, 1)) ./ obj.Saved.Sigma;
            grads = {reshape(dz, obj.Saved.Size)};
        end
    end
end
