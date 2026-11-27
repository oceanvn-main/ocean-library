classdef ImageQualityMetrics
    % IMAGEQUALITYMETRICS  Static PSNR and SSIM image quality metrics.
    %
    %   Provides peak signal-to-noise ratio (PSNR) and structural similarity
    %   index (SSIM) for quantitative comparison of two images.
    %
    %   All methods are static — no instantiation required:
    %     psnr_val = OceanStats.Image.ImageQualityMetrics.psnr(img, ref);
    %     ssim_val = OceanStats.Image.ImageQualityMetrics.ssim(img, ref);
    %     OceanStats.Image.ImageQualityMetrics.printReport(img, ref, 'POCS Output');
    %
    %   PSNR Formula (per channel, then averaged):
    %     MSE  = mean((img - ref).^2)
    %     PSNR = 10 * log10(MAX^2 / MSE)   where MAX=1 for double [0,1] images
    %
    %   SSIM Formula (Wang et al. 2004):
    %     SSIM = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
    %            / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))
    %     Computed locally in 11x11 Gaussian windows, then averaged spatially.
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-19
    %   Ref:    Wang et al., "Image Quality Assessment: From Error Visibility to
    %           Structural Similarity", IEEE Trans. Image Proc. 13(4), 2004.

    methods (Static)
        function val = psnr(img, ref)
            % PSNR  Peak Signal-to-Noise Ratio in dB.
            %
            %   Inputs:
            %     img — estimated image (H x W or H x W x C), double [0,1] or uint8
            %     ref — reference/ground truth image (same size as img)
            %
            %   Output:
            %     val — PSNR in dB (scalar). Returns Inf if images are identical.
            arguments
                img {mustBeNumeric}
                ref {mustBeNumeric}
            end

            [img_d, ref_d, max_val] = OceanStats.Image.ImageQualityMetrics.toDouble(img, ref);

            mse = mean((img_d(:) - ref_d(:)).^2);
            if mse < 1e-20
                val = Inf;
                return;
            end
            val = 10 * log10(max_val^2 / mse);
        end

        function val = ssim(img, ref)
            % SSIM  Structural Similarity Index Measure in [0, 1].
            %
            %   Inputs:
            %     img — estimated image (H x W or H x W x C), double [0,1] or uint8
            %     ref — reference/ground truth image
            %
            %   Output:
            %     val — Mean SSIM over all channels (scalar, 1 = perfect match)
            arguments
                img {mustBeNumeric}
                ref {mustBeNumeric}
            end

            [img_d, ref_d, ~] = OceanStats.Image.ImageQualityMetrics.toDouble(img, ref);

            C = size(img_d, 3);
            ssim_per_channel = zeros(1, C);

            for ch = 1:C
                ssim_per_channel(ch) = OceanStats.Image.ImageQualityMetrics.computeSSIM(...
                    img_d(:,:,ch), ref_d(:,:,ch));
            end

            val = mean(ssim_per_channel);
        end

        function printReport(img, ref, label)
            % PRINTREPORT  Print a formatted quality comparison table row.
            %
            %   Usage:
            %     OceanStats.Image.ImageQualityMetrics.printReport(img_pocs, img_clean, 'POCS');
            arguments
                img    {mustBeNumeric}
                ref    {mustBeNumeric}
                label  (1,:) char = 'Image'
            end

            psnr_v = OceanStats.Image.ImageQualityMetrics.psnr(img, ref);
            ssim_v = OceanStats.Image.ImageQualityMetrics.ssim(img, ref);
            fprintf('  %-28s   PSNR = %7.3f dB   SSIM = %.4f\n', label, psnr_v, ssim_v);
        end

        function printTable(results)
            % PRINTTABLE  Print a full comparison table from a results struct array.
            %
            %   Input:
            %     results — struct array with fields .Label, .PSNR, .SSIM
            %
            %   Usage:
            %     results(1).Label = 'JPEG Compressed'; results(1).PSNR = ...; results(1).SSIM = ...;
            %     OceanStats.Image.ImageQualityMetrics.printTable(results);
            arguments
                results struct
            end

            fprintf('\n%s\n', repmat('─', 1, 58));
            fprintf('  %-28s   %-12s   %-8s\n', 'Method', 'PSNR (dB)', 'SSIM');
            fprintf('%s\n', repmat('─', 1, 58));
            for k = 1:numel(results)
                fprintf('  %-28s   %-12.3f   %.4f\n', ...
                    results(k).Label, results(k).PSNR, results(k).SSIM);
            end
            fprintf('%s\n', repmat('─', 1, 58));
        end
    end

    methods (Static, Access = private)
        function [img_d, ref_d, max_val] = toDouble(img, ref)
            % TODOUBLE  Normalize images to double [0, 1] for metric computation.
            if isa(img, 'uint8')
                img_d   = double(img) / 255.0;
                ref_d   = double(ref) / 255.0;
                max_val = 1.0;
            else
                img_d   = double(img);
                ref_d   = double(ref);
                max_val = 1.0;
            end
        end

        function val = computeSSIM(x, y)
            % COMPUTESSIM  SSIM for a single 2D channel (Wang et al. 2004).
            K1 = 0.01; K2 = 0.03;
            L  = 1.0;          % Dynamic range [0, 1]
            C1 = (K1 * L)^2;
            C2 = (K2 * L)^2;

            % 11x11 Gaussian window (sigma=1.5) — per Wang et al.
            win_size = 11;
            sigma_w  = 1.5;
            [gx, gy] = meshgrid(-(win_size-1)/2 : (win_size-1)/2);
            win = exp(-(gx.^2 + gy.^2) / (2 * sigma_w^2));
            win = win / sum(win(:));

            % Local means
            mu_x = conv2(x, win, 'valid');
            mu_y = conv2(y, win, 'valid');

            mu_x2 = mu_x .* mu_x;
            mu_y2 = mu_y .* mu_y;
            mu_xy = mu_x .* mu_y;

            % Local variances and covariance
            sig_x2 = conv2(x .* x, win, 'valid') - mu_x2;
            sig_y2 = conv2(y .* y, win, 'valid') - mu_y2;
            sig_xy = conv2(x .* y, win, 'valid') - mu_xy;

            % SSIM map
            ssim_map = ((2 * mu_xy + C1) .* (2 * sig_xy + C2)) ./ ...
                       ((mu_x2 + mu_y2 + C1) .* (sig_x2 + sig_y2 + C2));

            val = mean(ssim_map(:));
        end
    end
end
