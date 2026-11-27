classdef JointBilateralFilter < OceanMath.Image.Filters.INeighborhoodFilter
    % JOINTBILATERALFILTER  Joint Bilateral Filter for guided image upsampling.
    %
    %   Adapter of the standard Bilateral Filter where the range weights are
    %   computed from a separate HIGH-RESOLUTION guidance image (e.g., luminance Y),
    %   rather than from the low-quality input signal being filtered.
    %
    %   This decoupling makes it suitable for chrominance upsampling:
    %     - Standard Bilateral uses: w_range = f(|p(x) - p(x_i)|)
    %     - Joint Bilateral uses:    w_range = f(|I(x) - I(x_i)|)
    %   where I is the guidance (sharp luma) and p is the sparse chroma.
    %
    %   Algorithm per pixel x:
    %     q(x) = SUM_i [ w_spatial(x, x_i) * w_range(I(x), I(x_i)) * p(x_i) ]
    %            / SUM_i [ w_spatial(x, x_i) * w_range(I(x), I(x_i)) ]
    %
    %     w_spatial(x, x_i) = exp(-||x - x_i||^2 / (2 * sigma_s^2))
    %     w_range(I(x), I(x_i)) = exp(-(I(x) - I(x_i))^2 / (2 * sigma_r^2))
    %
    %   Two modes of use:
    %     1. Standard apply(p)           — uses p as its own guidance (standard bilateral)
    %     2. applyGuided(p, guidance_I)  — uses high-res guidance for range weights
    %
    %   Usage:
    %     jbf = OceanMath.Image.Filters.JointBilateralFilter( ...
    %               'SpatialSigma', 5, 'RangeSigma', 0.15);
    %     chroma_up = jbf.applyGuided(chroma_nn, luma);
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-19
    %   Ref:    Eisemann & Durand, "Flash Photography Enhancement via Intrinsic Relighting",
    %           SIGGRAPH 2004. Kopf et al., "Joint Bilateral Upsampling", SIGGRAPH 2007.

    properties
        % SpatialSigma: standard deviation for spatial Gaussian weights (pixels)
        SpatialSigma (1,1) double {mustBePositive} = 5.0

        % RangeSigma: standard deviation for range Gaussian weights (intensity units [0,1])
        RangeSigma   (1,1) double {mustBePositive} = 0.15
    end

    methods
        function obj = JointBilateralFilter(opts)
            % JOINTBILATERALFILTER Constructor.
            arguments
                opts.SpatialSigma   (1,1) double = 5.0
                opts.RangeSigma     (1,1) double = 0.15
                opts.Boundary       (1,:) char   = 'replicate'
                opts.ConstantValue  (1,1) double = 0.0
            end
            obj = obj@OceanMath.Image.Filters.INeighborhoodFilter(...
                'Boundary', opts.Boundary, 'ConstantValue', opts.ConstantValue);
            obj.SpatialSigma = opts.SpatialSigma;
            obj.RangeSigma   = opts.RangeSigma;
        end

        function q = applyGuided(obj, p, guidance)
            % APPLYGUIDED  Filter p using a separate guidance image for range weights.
            %
            %   Inputs:
            %     p        — (H x W) signal to filter (e.g., upsampled sparse chroma)
            %     guidance — (H x W) guidance image (e.g., full-resolution luminance Y)
            %
            %   Output:
            %     q        — (H x W) filtered output
            arguments
                obj
                p        (:,:) double
                guidance (:,:) double
            end
            q = obj.runJBF(p, guidance);
        end
    end

    methods (Access = protected)
        function pad_size = getPadSize(obj)
            % Template Method hook: pad window = 3 * sigma_s
            pad = ceil(3 * obj.SpatialSigma);
            pad_size = [pad, pad];
        end

        function I_out = applyFilter(obj, I_padded, I_in)
            % Template Method hook: self-guided bilateral filtering (guidance == input).
            if size(I_in, 3) == 1
                I_out = obj.runJBF(I_in, I_in);
            else
                I_out = zeros(size(I_in));
                for ch = 1:size(I_in, 3)
                    chan = I_in(:,:,ch);
                    I_out(:,:,ch) = obj.runJBF(chan, chan);
                end
            end
        end
    end

    methods (Access = private)
        function q = runJBF(obj, p, guidance)
            % RUNJBF  Core Joint Bilateral Filter computation.
            [H, W] = size(p);
            pad = ceil(3 * obj.SpatialSigma);

            % Pre-compute spatial weight kernel
            [dx, dy] = meshgrid(-pad:pad, -pad:pad);
            w_spatial = exp(-(dx.^2 + dy.^2) / (2 * obj.SpatialSigma^2));

            % Pad both p and guidance
            p_pad = padarray(p,        [pad, pad], 'replicate', 'both');
            g_pad = padarray(guidance, [pad, pad], 'replicate', 'both');

            q = zeros(H, W);
            k_size = 2*pad + 1;

            for r = 1:H
                for c = 1:W
                    % Extract guidance patch
                    g_patch = g_pad(r:r+k_size-1, c:c+k_size-1);

                    % Range weights from guidance (NOT from p)
                    diffs_g = g_patch - guidance(r, c);
                    w_range = exp(-diffs_g.^2 / (2 * obj.RangeSigma^2));

                    % Combined weights
                    w = w_spatial .* w_range;
                    w_sum = sum(w(:));

                    if w_sum < 1e-12
                        q(r, c) = p(r, c);
                    else
                        % Weighted average of input p
                        p_patch = p_pad(r:r+k_size-1, c:c+k_size-1);
                        q(r, c) = sum(w(:) .* p_patch(:)) / w_sum;
                    end
                end
            end
        end
    end
end
