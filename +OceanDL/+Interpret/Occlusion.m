classdef Occlusion
    % OCCLUSION  Cover part of the input with a square and record how much a
    %   response drops (Szeliski 2nd ed. Sec. 5.4 (pp. 309-311, Figs. 5.47-5.50), Fig. 5.48; Zeiler & Fergus 2014).
    %   All occluded copies go through the model in one batch.
    %
    %   M = Occlusion.map(model, x, target, layer, 'Size', 4, 'Stride', 1, 'Fill', 0)
    %       layer: the traced layer the target reads (decides how much of the model runs)
    %       x (H x W x C) one image; target(acts) -> (1 x N) response per image
    %       M (H x W): average drop  target(x) - target(x occluded)  over all
    %       squares covering each pixel (positive = the pixel region supports the response)
    %   f = Occlusion.unitTarget(layer, channel, row, col)   one unit ([] row/col = channel mean)
    %   'Fill': value of the occluder (Zeiler & Fergus use grey; for ink-on-black
    %   digits 0 = "erase", the dataset mean = "grey").

    methods (Static)
        function [M, base] = map(model, x, target, layer, opts)
            arguments
                model
                x
                target function_handle
                layer
                opts.Size (1,1) double = 4
                opts.Stride (1,1) double = 1
                opts.Fill (1,1) double = 0
            end
            [H, W, C] = size(x, 1:3);
            ys = 1:opts.Stride:H - opts.Size + 1; xs = 1:opts.Stride:W - opts.Size + 1;
            n = numel(ys) * numel(xs);
            X = repmat(reshape(x, H, W, C, 1), 1, 1, 1, n + 1);
            k = 1;
            for i = ys
                for j = xs
                    k = k + 1;
                    X(i:i + opts.Size - 1, j:j + opts.Size - 1, :, k) = opts.Fill;
                end
            end
            acts = OceanDL.Autograd.GradMode.noGrad(@() model.trace(X, 'Parts', OceanDL.Interpret.Probe.partsFor(layer)));
            r = double(target(acts));
            base = r(1);
            drop = base - r(2:end);
            M = zeros(H, W); cnt = zeros(H, W); k = 0;
            for i = ys
                for j = xs
                    k = k + 1;
                    M(i:i + opts.Size - 1, j:j + opts.Size - 1) = M(i:i + opts.Size - 1, j:j + opts.Size - 1) + drop(k);
                    cnt(i:i + opts.Size - 1, j:j + opts.Size - 1) = cnt(i:i + opts.Size - 1, j:j + opts.Size - 1) + 1;
                end
            end
            M = M ./ max(cnt, 1);
        end

        function f = unitTarget(layer, channel, row, col)
            layer = char(layer);
            f = @(acts) OceanDL.Interpret.Occlusion.pickUnit(OceanDL.Autograd.Tensor.raw(acts(layer)), channel, row, col);
        end

        function v = pickUnit(a, channel, row, col)
            if isempty(row)
                v = reshape(mean(a(:, :, channel, :), [1 2]), 1, []);
            else
                v = reshape(a(row, col, channel, :), 1, []);
            end
        end
    end
end
