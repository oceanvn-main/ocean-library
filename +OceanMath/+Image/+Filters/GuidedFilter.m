classdef GuidedFilter < OceanMath.Image.Filters.INeighborhoodFilter
    % GUIDEDFILTER  Guided Image Filter (He et al. 2013).
    %
    %   The Guided Image Filter uses a guidance image I to compute a local linear
    %   model between the guidance and the filter input p. It is equivalent to
    %   edge-preserving smoothing in the self-guided case (I == p), and to
    %   high-quality upsampling when I is a high-resolution structure image.
    %
    %   Algorithm (He et al. 2013, TPAMI):
    %     In each local r x r window omega_k centred at k:
    %       a_k = (cov(I, p) + eps) / (var(I) + eps)
    %       b_k = mean(p) - a_k * mean(I)
    %     Linear coefficients (a, b) are averaged over all overlapping windows:
    %       a_bar(i) = mean_{k: i in omega_k} a_k
    %       b_bar(i) = mean_{k: i in omega_k} b_k
    %     Output:
    %       q(i) = a_bar(i) * I(i) + b_bar(i)
    %
    %   All window statistics are computed efficiently via integral images (box filters),
    %   giving O(N) complexity regardless of window radius.
    %
    %   Two modes of use:
    %     1. Self-guided smoothing:  q = gf.apply(I)           (guidance == input)
    %     2. Guided upsampling:      q = gf.applyGuided(p, I)  (guidance != input)
    %
    %   Usage:
    %     gf = OceanMath.Image.Filters.GuidedFilter('Radius', 8, 'Epsilon', 0.01);
    %     p_smooth = gf.apply(p_noisy);                % self-guided
    %     chroma_up = gf.applyGuided(chroma_nn, luma); % guided upsampling
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-19
    %   Ref:    K. He, J. Sun, X. Tang, "Guided Image Filtering",
    %           IEEE TPAMI 35(6), 2013. https://doi.org/10.1109/TPAMI.2012.213

    properties
        % Radius: half-width of the local square window (window size = 2*r+1)
        Radius  (1,1) double {mustBePositive, mustBeInteger} = 8

        % Epsilon: regularization preventing division by zero in flat regions
        Epsilon (1,1) double {mustBePositive} = 0.01
    end

    methods
        function obj = GuidedFilter(opts)
            % GUIDEDFILTER Constructor.
            arguments
                opts.Radius      (1,1) double  = 8
                opts.Epsilon     (1,1) double  = 0.01
                opts.Boundary    (1,:) char    = 'replicate'
                opts.ConstantValue (1,1) double = 0.0
            end
            obj = obj@OceanMath.Image.Filters.INeighborhoodFilter(...
                'Boundary', opts.Boundary, 'ConstantValue', opts.ConstantValue);
            obj.Radius  = opts.Radius;
            obj.Epsilon = opts.Epsilon;
        end

        function q = applyGuided(obj, p, guidance)
            % APPLYGUIDED  Filter p using a separate guidance image.
            %
            %   Inputs:
            %     p        — (H x W) signal to filter (e.g., upsampled chroma channel)
            %     guidance — (H x W) guidance image (e.g., full-resolution luminance Y)
            %
            %   Output:
            %     q        — (H x W) filtered output with edges transferred from guidance
            arguments
                obj
                p        (:,:) double
                guidance (:,:) double
            end
            q = obj.runGuidedFilter(guidance, p);
        end
    end

    methods (Access = protected)
        function pad_size = getPadSize(obj)
            % Template Method hook: pad by Radius on each side.
            r = obj.Radius;
            pad_size = [r, r];
        end

        function I_out = applyFilter(obj, ~, I_in)
            % Template Method hook: self-guided filtering (I == p).
            %
            %   Note: INeighborhoodFilter pads the image before calling this.
            %   We ignore the pre-padded I_padded and recompute internally
            %   to use efficient box-filter integrals consistently.
            if size(I_in, 3) == 1
                I_out = obj.runGuidedFilter(I_in, I_in);
            else
                % Multi-channel: filter each channel independently (self-guided)
                I_out = zeros(size(I_in));
                for ch = 1:size(I_in, 3)
                    chan = I_in(:,:,ch);
                    I_out(:,:,ch) = obj.runGuidedFilter(chan, chan);
                end
            end
        end
    end

    methods (Access = private)
        function q = runGuidedFilter(obj, I, p)
            % RUNGUIDEDFILTER  Core guided filter computation (single channel).
            %
            %   I — guidance (H x W) double
            %   p — input signal (H x W) double
            %   q — output (H x W) double
            r = obj.Radius;
            eps = obj.Epsilon;

            [H, W] = size(I);
            N = obj.boxFilter(ones(H, W), r);  % number of pixels per window

            % Local window statistics of guidance I
            mean_I = obj.boxFilter(I, r) ./ N;
            mean_p = obj.boxFilter(p, r) ./ N;

            corr_Ip = obj.boxFilter(I .* p, r) ./ N;
            corr_II = obj.boxFilter(I .* I, r) ./ N;

            var_I  = corr_II  - mean_I .* mean_I;
            cov_Ip = corr_Ip  - mean_I .* mean_p;

            % Compute local linear model coefficients
            a = cov_Ip ./ (var_I + eps);
            b = mean_p - a .* mean_I;

            % Average coefficients over overlapping windows
            a_bar = obj.boxFilter(a, r) ./ N;
            b_bar = obj.boxFilter(b, r) ./ N;

            % Reconstruct output
            q = a_bar .* I + b_bar;
        end

        function out = boxFilter(~, img, r)
            % BOXFILTER  2-D box filter using the cumulative sum (integral image) trick.
            %
            %   For each pixel (i,j) in the original HxW image, computes the sum
            %   of all values in the (2r+1) x (2r+1) window centred on (i,j).
            %   Zero-pads the boundary so every pixel has a full window.
            %   Complexity: O(H*W) regardless of window size r.
            %
            %   Derivation:
            %     After padding: img_pad is (H+2r) x (W+2r).
            %     For pixel (i,j) of original image, the window spans rows
            %     [i, i+2r] and cols [j, j+2r] in img_pad.
            %     Cumsum cs_r(k,*) = sum of img_pad(1:k, *).
            %     Prepend zero row → cs_r(0,*) = 0.
            %     Row window sum = cs_r(i+2r, *) - cs_r(i-1, *) for i=1..H
            %                    = cs_r(2r+2:H+2r+1) - cs_r(1:H)   [1-indexed with prepend]
            %     Apply the same trick column-wise on the row-summed result.

            [H, W] = size(img);

            % Pad by r pixels on each side with zeros
            img_pad = padarray(img, [r, r], 0, 'both');  % (H+2r) x (W+2r)

            % ── Row sliding sum (window height 2r+1) ──────────────────────────
            cs_r = cumsum(img_pad, 1);                       % (H+2r)   x (W+2r)
            cs_r = [zeros(1, size(cs_r, 2)); cs_r];          % (H+2r+1) x (W+2r)
            row_sum = cs_r(2*r+2:end, :) - cs_r(1:H, :);    % H        x (W+2r)

            % ── Column sliding sum (window width 2r+1) ────────────────────────
            cs_c = cumsum(row_sum, 2);                        % H x (W+2r)
            cs_c = [zeros(size(cs_c, 1), 1), cs_c];          % H x (W+2r+1)
            out  = cs_c(:, 2*r+2:end) - cs_c(:, 1:W);        % H x W
        end

    end
end
