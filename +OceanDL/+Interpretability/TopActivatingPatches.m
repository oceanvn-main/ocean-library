classdef TopActivatingPatches
    % TOPACTIVATINGPATCHES  "What kinds of inputs excite this feature?"
    %   Forward-passes a batch of candidate images, ranks them (and their
    %   spatial location) by a chosen filter channel's activation, and
    %   crops an input-space patch around each top location.
    %
    %   RECEPTIVE-FIELD APPROXIMATION (stated explicitly, not silently
    %   assumed exact -- honest-opacity in spirit): the patch is centered
    %   at input coordinates scaled by input-size/output-size (exact for a
    %   single conv+pool chain, an approximation for deeper stacks since it
    %   ignores padding-induced offset). PatchSize can be overridden
    %   directly when the true receptive field is known.
    %
    %   USAGE:
    %     r = TopActivatingPatches.find(engine, images, 'conv2', 5, 'TopK', 9);
    %     r.Patches{1}   % the single highest-activating crop for channel 5

    methods (Static)
        function result = find(engine, images, layerName, channelIdx, opts)
            arguments
                engine
                images (:,:,:,:) double
                layerName (1,:) char
                channelIdx (1,1) double {mustBePositive, mustBeInteger}
                opts.TopK (1,1) double {mustBePositive, mustBeInteger} = 9
                opts.PatchSize (1,2) double {mustBeNonnegative, mustBeInteger} = [0 0]  % [0 0] = auto
            end
            [H, W, ~, N] = size(images);
            engine.forward(images);
            maps = engine.getActivations(layerName);    % (Ho, Wo, C, N)
            [Ho, Wo, ~, ~] = size(maps);

            if channelIdx > size(maps, 3)
                error('OceanDL:Interpretability:TopActivatingPatches:BadChannel', ...
                    'channelIdx=%d but layer "%s" only has %d channels.', channelIdx, layerName, size(maps, 3));
            end

            strideH = H / Ho; strideW = W / Wo;
            if isequal(opts.PatchSize, [0 0])
                ph = max(1, round(strideH * 3));
                pw = max(1, round(strideW * 3));
            else
                ph = opts.PatchSize(1); pw = opts.PatchSize(2);
            end

            actMap = reshape(maps(:, :, channelIdx, :), Ho, Wo, N);
            k = min(opts.TopK, numel(actMap));
            [vals, linIdx] = maxk(actMap(:), k);
            [oi, oj, on] = ind2sub([Ho, Wo, N], linIdx);

            patches = cell(k, 1);
            centers = zeros(k, 2);
            for t = 1:k
                ci = round((oi(t) - 0.5) * strideH);
                cj = round((oj(t) - 0.5) * strideW);
                r1 = min(H, max(1, ci) + floor(ph/2)); r0 = max(1, r1 - ph + 1); r1 = min(H, r0 + ph - 1);
                c1 = min(W, max(1, cj) + floor(pw/2)); c0 = max(1, c1 - pw + 1); c1 = min(W, c0 + pw - 1);
                centers(t, :) = [ci, cj];
                patches{t} = images(r0:r1, c0:c1, :, on(t));
            end

            result = struct('Values', vals, 'ImageIndex', on, 'OutputRow', oi, 'OutputCol', oj, ...
                'Centers', centers, 'Patches', {patches}, 'PatchSize', [ph pw]);
        end
    end
end
