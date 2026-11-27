classdef PatchMerging < OceanDL.nn.Module
    % PATCHMERGING  Swin's down-sampling between stages (Liu et al. 2021, Sec. 3.1):
    %   concatenate the features of each 2 x 2 group of neighbouring patches
    %   (4D features), LayerNorm, then a linear map 4D -> 2D.
    %     (D x N x Hp x Wp) -> (2D x N x Hp/2 x Wp/2)
    %   Halves the resolution and doubles the width -- the CNN-like pyramid that
    %   lets Swin feed detection / segmentation heads (Szeliski Ch. 6).

    properties (SetAccess = protected)
        Norm
        Reduction
    end

    methods
        function obj = PatchMerging(dim, opts)
            arguments
                dim (1,1) double {mustBePositive, mustBeInteger}
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            obj.Norm = OceanDL.nn.LayerNorm(4 * dim, 'DataType', opts.DataType);
            obj.Reduction = OceanDL.nn.Linear(4 * dim, 2 * dim, 'Bias', false, 'Seed', opts.Seed, 'DataType', opts.DataType);
        end

        function Y = forward(obj, X4)
            [~, ~, Hp, Wp] = size(X4, 1:4);
            if mod(Hp, 2) || mod(Wp, 2)
                error('OceanDL:Vision:PatchMerging:Odd', 'Patch grid %dx%d must be even to merge 2x2 groups.', Hp, Wp);
            end
            x0 = X4(:, :, 1:2:end, 1:2:end);  x1 = X4(:, :, 2:2:end, 1:2:end);
            x2 = X4(:, :, 1:2:end, 2:2:end);  x3 = X4(:, :, 2:2:end, 2:2:end);
            Y = obj.Reduction.forward(obj.Norm.forward(cat(1, x0, x1, x2, x3)));
        end
    end
end
