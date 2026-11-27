classdef ReceptiveField
    % RECEPTIVEFIELD  Which input pixels a unit of a layer looks at, to crop the
    %   "top activating patches" of Szeliski Fig. 5.47 (higher layers -> larger patches).
    %
    %   EFFECTIVE receptive field (Luo et al. 2016 -- not in the texts): with
    %   GroupNorm every unit depends, weakly, on EVERY pixel, so the theoretical
    %   field is the whole image. We therefore measure where the gradient of the
    %   centre unit is concentrated: |d a / d x| averaged over a few images,
    %   thresholded at 'Threshold' x its maximum; the bounding box is the field.
    %
    %   rf = ReceptiveField.effective(model, X, layer)  struct Size [h w] (pixels),
    %        Stride (input pixels per layer cell), LayerSize [h w]
    %   P  = ReceptiveField.crop(x, rf, row, col)       the input patch under unit (row, col)
    %                                                    (zero-padded at the borders)

    methods (Static)
        function rf = effective(model, X, layer, opts)
            arguments
                model
                X                                  % (H x W x C x N) images, e.g. dataset samples
                layer
                opts.Images (1,1) double = 8
                opts.Threshold (1,1) double = 0.1
            end
            a = OceanDL.Interpret.Probe.activation(model, X(:, :, :, 1), layer);
            [h, w, ~, ~] = size(a);
            r0 = ceil(h / 2); c0 = ceil(w / 2);
            acc = zeros(size(X, 1), size(X, 2));
            n = min(opts.Images, size(X, 4));
            for k = 1:n                                     % all channels of the centre unit at once
                g = OceanDL.Interpret.Probe.unitGradient(model, X(:, :, :, k), layer, [], r0, c0);
                acc = acc + sum(abs(double(g)), 3);
            end
            if max(acc(:)) == 0                             % the layer does not depend on x (e.g. a learned constant)
                acc(:) = 1;
            end
            mask = acc >= opts.Threshold * max(acc(:));
            rows = find(any(mask, 2)); cols = find(any(mask, 1));
            rf.Size = [numel(rows(1):rows(end)), numel(cols(1):cols(end))];
            rf.Stride = [size(X, 1) / h, size(X, 2) / w];
            rf.LayerSize = [h w];
            rf.Map = acc / max(acc(:));
        end

        function P = crop(x, rf, row, col)
            H = size(x, 1); W = size(x, 2);
            cy = round((row - 0.5) * rf.Stride(1) + 0.5); cx = round((col - 0.5) * rf.Stride(2) + 0.5);
            ry = cy - floor(rf.Size(1) / 2) + (0:rf.Size(1) - 1);
            rx = cx - floor(rf.Size(2) / 2) + (0:rf.Size(2) - 1);
            P = zeros(rf.Size(1), rf.Size(2), size(x, 3), 'like', x);
            iy = ry >= 1 & ry <= H; ix = rx >= 1 & rx <= W;
            P(iy, ix, :) = x(ry(iy), rx(ix), :);
        end
    end
end
