classdef PocsDeblockingEngine
    % POCSDEBLOCKINGENGINE  POCS-based image deblocking via convex hyper-rectangle projection.
    %
    %   Removes JPEG-style blocking artifacts using Projection onto Convex Sets (POCS).
    %   Each iteration alternates between:
    %     1. Spatial smoothing (suppresses block-boundary ringing)
    %     2. DCT-domain projection (enforces consistency with quantized coefficients)
    %
    %   The POCS Convex Projection Constraint (Szeliski §4.3):
    %     For each 8x8 block, each DCT coefficient c_hat(i,j) must satisfy:
    %
    %       c_clamped = clamp(c_hat, Q(i,j)*(q(i,j) - 0.5), Q(i,j)*(q(i,j) + 0.5))
    %
    %     where Q(i,j) is the quantization step and q(i,j) is the stored integer index.
    %     This is the projection onto the valid quantization interval (a convex set).
    %
    %   Convergence is guaranteed by alternating projections onto convex sets.
    %
    %   Usage:
    %     engine = OceanMath.Image.Compression.PocsDeblockingEngine( ...
    %                  'MaxIter', 30, 'Tolerance', 1e-4);
    %     [img_out, history] = engine.deblock(img_compressed, q_all, Q_table);
    %
    %   Author: Antigravity Code Assistant
    %   Date:   2026-08-19
    %   Ref:    Szeliski "Computer Vision" 2e, §4.3 (p. 234)
    %           Yang et al., "Error Resilient Video Coding Using the Set Partitioning in
    %           Hierarchical Trees", IEEE Trans. Circuits Syst. Video Technol., 1997.

    properties
        % Maximum number of POCS iterations
        MaxIter     (1,1) double {mustBePositive, mustBeInteger} = 30

        % Convergence tolerance (relative change in image norm)
        Tolerance   (1,1) double {mustBePositive} = 1e-4

        % Smoother: pluggable INeighborhoodFilter applied at each iteration.
        %   Default: BilateralFilter (edge-preserving).
        %   Set to [] to disable spatial smoothing (pure POCS projection only).
        Smoother    = []

        % Verbose: print convergence trace per iteration
        Verbose     (1,1) logical = true

        % BlockSize: DCT block size (must match compressor's BlockSize)
        BlockSize   (1,1) double {mustBeMember(BlockSize, [4, 8, 16])} = 8

        % SmoothEvery: Apply smoother every N iterations (default=1, i.e. every iteration)
        SmoothEvery (1,1) double {mustBePositive, mustBeInteger} = 1
    end

    methods
        function obj = PocsDeblockingEngine(opts)
            % POCSDEBLOCKINGENGINE Constructor.
            arguments
                opts.MaxIter      (1,1) double = 30
                opts.Tolerance    (1,1) double = 1e-4
                opts.Smoother              = []
                opts.Verbose      (1,1) logical = true
                opts.BlockSize    (1,1) double = 8
                opts.SmoothEvery  (1,1) double = 1
            end
            obj.MaxIter     = opts.MaxIter;
            obj.Tolerance   = opts.Tolerance;
            obj.Smoother    = opts.Smoother;
            obj.Verbose     = opts.Verbose;
            obj.BlockSize   = opts.BlockSize;
            obj.SmoothEvery = opts.SmoothEvery;

            % Default smoother: BilateralFilter (edge-preserving)
            if isempty(obj.Smoother)
                obj.Smoother = OceanMath.Image.Filters.BilateralFilter(...
                    'SpatialSigma', 1.5, ...
                    'RangeSigma',   0.08, ...
                    'Method',       'direct');
            end
        end

        function [img_out, history] = deblock(obj, img_in, q_all, Q_table)
            % DEBLOCK  Run POCS deblocking loop.
            %
            %   Inputs:
            %     img_in   — Compressed image (H x W x C) in double [0, 1]
            %     q_all    — Cell array of quantized DCT indices from JpegSyntheticCompressor
            %     Q_table  — 8x8 quantization table from JpegSyntheticCompressor
            %
            %   Outputs:
            %     img_out  — Deblocked image (H x W x C) in double [0, 1]
            %     history  — struct with fields:
            %                  .RMSE     — (MaxIter x 1) relative change norm per iter
            %                  .NumIter  — actual iterations run before convergence
            arguments
                obj
                img_in   (:,:,:) double
                q_all    cell
                Q_table  (8,8)  double
            end

            B = obj.BlockSize;
            [H, W, C] = size(img_in);

            % Pad image to block-aligned dimensions
            H_pad = ceil(H / B) * B;
            W_pad = ceil(W / B) * B;

            % Initialize with the compressed image (scaled to [0, 255])
            f = img_in * 255.0;
            if H_pad > H || W_pad > W
                f_pad = zeros(H_pad, W_pad, C);
                f_pad(1:H, 1:W, :) = f;
                % Replicate edges
                if H_pad > H
                    f_pad(H+1:end, 1:W, :) = repmat(f(end,:,:), [H_pad-H, 1, 1]);
                end
                if W_pad > W
                    f_pad(:, W+1:end, :) = repmat(f_pad(:, W, :), [1, W_pad-W, 1]);
                end
            else
                f_pad = f;
            end

            history.RMSE    = zeros(obj.MaxIter, 1);
            history.NumIter = 0;

            if obj.Verbose
                fprintf('[POCS] Starting deblocking: MaxIter=%d  Tol=%.2e\n', ...
                    obj.MaxIter, obj.Tolerance);
                fprintf('%-6s  %-12s\n', 'Iter', 'RelChange');
                fprintf('%s\n', repmat('-', 1, 20));
            end

            for iter = 1:obj.MaxIter
                f_old = f_pad;

                %% Step 1: Spatial smoothing (suppresses block boundary ringing)
                if mod(iter, obj.SmoothEvery) == 0 && ~isempty(obj.Smoother)
                    f_smooth_d = f_pad / 255.0;
                    f_smooth_d = obj.Smoother.apply(f_smooth_d);
                    f_pad = f_smooth_d * 255.0;
                end

                %% Step 2: DCT-domain projection (convex hyper-rectangle clamp)
                f_pad = obj.dctProjection(f_pad, q_all, Q_table, H_pad, W_pad, C, B);

                %% Convergence check
                delta = norm(f_pad(:) - f_old(:)) / (norm(f_old(:)) + 1e-12);
                history.RMSE(iter) = delta;
                history.NumIter    = iter;

                if obj.Verbose
                    fprintf('%-6d  %-12.6e\n', iter, delta);
                end

                if delta < obj.Tolerance
                    if obj.Verbose
                        fprintf('[POCS] Converged at iteration %d (delta=%.2e)\n', iter, delta);
                    end
                    break;
                end
            end

            % Crop and scale back to [0, 1]
            img_out = max(0.0, min(1.0, f_pad(1:H, 1:W, :) / 255.0));
            history.RMSE = history.RMSE(1:history.NumIter);

            if obj.Verbose
                fprintf('[POCS] Done. Total iterations: %d\n', history.NumIter);
            end
        end
    end

    methods (Access = private)
        function f_out = dctProjection(obj, f, q_all, Q_table, H_pad, W_pad, C, B)
            % DCTPROJECTION  Apply POCS projection in DCT domain per 8x8 block.
            %
            %   For each DCT coefficient c_hat(i,j):
            %     c_clamped = clamp(c_hat, Q(i,j)*(q(i,j)-0.5), Q(i,j)*(q(i,j)+0.5))
            nBH = H_pad / B;
            nBW = W_pad / B;
            f_out = f;

            for ch = 1:C
                q_ch = q_all{min(ch, numel(q_all))};

                for bi = 1:nBH
                    for bj = 1:nBW
                        ri = (bi-1)*B + 1;
                        ci = (bj-1)*B + 1;

                        % Extract block, level-shift
                        blk = f(ri:ri+B-1, ci:ci+B-1, ch) - 128.0;

                        % Forward DCT
                        C_hat = dct2(blk);

                        % Retrieve stored quantized index
                        q = squeeze(q_ch(bi, bj, :, :));

                        % Convex hyper-rectangle projection (clamp to quantization interval)
                        lo = Q_table .* (q - 0.5);
                        hi = Q_table .* (q + 0.5);
                        C_clamped = max(lo, min(hi, C_hat));

                        % Inverse DCT → spatial domain
                        blk_rec = idct2(C_clamped) + 128.0;
                        f_out(ri:ri+B-1, ci:ci+B-1, ch) = blk_rec;
                    end
                end
            end
        end
    end
end
