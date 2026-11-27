classdef AdaptiveHistogramEqualization < OceanMath.Image.Pixels.IPixelTransform
    % ADAPTIVEHISTOGRAMEQUALIZATION  Contrast-Limited Adaptive Histogram Equalization (CLAHE).
    %
    %   Partitions the image into grid tiles, computes local contrast-limited 
    %   histograms, and bilinearly interpolates intensity transfer mappings.
    %
    %   Constructor:
    %       AdaptiveHistogramEqualization('NumTiles', [M, N], 'ClipLimit', limit, ...
    %                                     'BilinearAccumulate', tf, 'FilterCDFSpatial', tf, 'FilterSigma', s)
    %
    %   Inputs:
    %       NumTiles           - [1 x 2] grid layout of tiles (default: [8, 8]).
    %       ClipLimit          - Contrast limit fraction in [0.0, 1.0] (default: 0.02).
    %       BilinearAccumulate - Distribute counts to adjacent vertices bilinearly (default: false).
    %       FilterCDFSpatial   - Spatially smooth transfer functions across grid (default: false).
    %       FilterSigma        - Sigma for spatial Gaussian smoothing kernel (default: 1.0).
    %
    %   Author: Ocean Library System
    %   Date: 2026-07-09
    
    properties (Access = private)
        NumTiles           (1,2) double = [8, 8]
        ClipLimit          (1,1) double = 0.02
        BilinearAccumulate (1,1) logical = false
        FilterCDFSpatial   (1,1) logical = false
        FilterSigma        (1,1) double = 1.0
    end
    
    methods
        function obj = AdaptiveHistogramEqualization(options)
            % ADAPTIVEHISTOGRAMEQUALIZATION Construct a CLAHE operator.
            arguments
                options.NumTiles           (1,2) double = [8, 8]
                options.ClipLimit          (1,1) double = 0.02
                options.BilinearAccumulate (1,1) logical = false
                options.FilterCDFSpatial   (1,1) logical = false
                options.FilterSigma        (1,1) double = 1.0
            end
            obj.NumTiles = options.NumTiles;
            obj.ClipLimit = options.ClipLimit;
            obj.BilinearAccumulate = options.BilinearAccumulate;
            obj.FilterCDFSpatial = options.FilterCDFSpatial;
            obj.FilterSigma = options.FilterSigma;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Performs the adaptive contrast enhancement.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            is_uint8 = isa(I_in, 'uint8');
            I_double = double(I_in);
            
            [height, width, num_channels] = size(I_double);
            I_res = zeros(size(I_double));
            
            M = obj.NumTiles(1);
            N = obj.NumTiles(2);
            
            tile_h = floor(height / M);
            tile_w = floor(width / N);
            
            if tile_h < 2 || tile_w < 2
                error('Tile size is too small for the given image size and grid layout.');
            end
            
            tile_pixels = tile_h * tile_w;
            clip_threshold = max(1.0, round(obj.ClipLimit * tile_pixels));
            
            % Compute tile centers
            center_y = (0.5 : M-0.5) * tile_h;
            center_x = (0.5 : N-0.5) * tile_w;
            
            % Precompute meshgrids of pixel coordinates
            [X, Y] = meshgrid(1:width, 1:height);
            
            % Clamp pixel coordinates to the tile centers bounding box. Solve for the boundary where 4-neighbor doesn't exist
            X_clamped = max(center_x(1), min(center_x(end), X));
            Y_clamped = max(center_y(1), min(center_y(end), Y));
            
            % Determine the lower-left enclosing tile indices r, c
            % (r is row index in [1, M-1], c is col index in [1, N-1])
            c = floor((X_clamped - center_x(1)) / tile_w) + 1; %Which one is left most
            r = floor((Y_clamped - center_y(1)) / tile_h) + 1; % Which one is bottom most
            
            c = max(1, min(N - 1, c));
            r = max(1, min(M - 1, r));
            
            % Interpolation fractions
            s = (X_clamped - center_x(c)) / tile_w;
            t = (Y_clamped - center_y(r)) / tile_h;
            
            for chan = 1:num_channels
                channel_data = I_double(:, :, chan);
                
                % Check if range is normalized [0, 1]
                is_normalized = ~is_uint8 && max(channel_data(:)) <= 1.0;
                if is_normalized
                    channel_data = channel_data * 255.0;
                end
                
                % Round and clamp to ensure valid mapping table indices [0, 255]
                channel_rounded = round(channel_data);
                channel_rounded = max(0, min(255, channel_rounded));
                
                % Step 1: Compute local contrast-limited transfer functions for all tiles
                T = zeros(256, M, N);
                
                if obj.BilinearAccumulate
                    % Enclosing tile indices and weights for bilinear accumulation
                    c_acc = floor((X_clamped - center_x(1)) / tile_w) + 1;
                    r_acc = floor((Y_clamped - center_y(1)) / tile_h) + 1;
                    c_acc = max(1, min(N - 1, c_acc));
                    r_acc = max(1, min(M - 1, r_acc));
                    
                    s_acc = max(0, min(1, (X - center_x(c_acc)) / tile_w));
                    t_acc = max(0, min(1, (Y - center_y(r_acc)) / tile_h));
                    
                    w00 = (1.0 - s_acc) .* (1.0 - t_acc);
                    w01 = s_acc .* (1.0 - t_acc);
                    w10 = (1.0 - s_acc) .* t_acc;
                    w11 = s_acc .* t_acc;
                    
                    v_acc = double(channel_rounded) + 1;
                    
                    subs00 = [v_acc(:), r_acc(:),     c_acc(:)];
                    subs01 = [v_acc(:), r_acc(:),     c_acc(:) + 1];
                    subs10 = [v_acc(:), r_acc(:) + 1, c_acc(:)];
                    subs11 = [v_acc(:), r_acc(:) + 1, c_acc(:) + 1];
                    
                    h_tensor = accumarray(subs00, w00(:), [256, M, N]) + ...
                               accumarray(subs01, w01(:), [256, M, N]) + ...
                               accumarray(subs10, w10(:), [256, M, N]) + ...
                               accumarray(subs11, w11(:), [256, M, N]);
                    
                    for r_idx = 1:M
                        for c_idx = 1:N
                            h = h_tensor(:, r_idx, c_idx);
                            excess = sum(max(0, h - clip_threshold));
                            h_clipped = min(h, clip_threshold);
                            h_final = h_clipped + (excess / 256.0);
                            
                            cdf = cumsum(h_final);
                            T(:, r_idx, c_idx) = 255.0 * (cdf / sum(h_final));
                        end
                    end
                else
                    % Standard discrete tile-based calculation
                    for r_idx = 1:M
                        for c_idx = 1:N
                            % Extract tile sub-image
                            r_start = (r_idx - 1) * tile_h + 1;
                            r_end = r_idx * tile_h;
                            c_start = (c_idx - 1) * tile_w + 1;
                            c_end = c_idx * tile_w;
                            
                            tile_sub = channel_rounded(r_start:r_end, c_start:c_end);
                            
                            % Local histogram
                            h = histcounts(tile_sub(:), 0:256)';
                            
                            % Apply contrast limiting (clipping)
                            excess = sum(max(0, h - clip_threshold));
                            h_clipped = min(h, clip_threshold);
                            
                            % Redistribute excess uniformly
                            bin_incr = excess / 256.0;
                            h_final = h_clipped + bin_incr;
                            
                            % Compute CDF and map
                            cdf = cumsum(h_final);
                            T(:, r_idx, c_idx) = 255.0 * (cdf / sum(h_final));
                        end
                    end
                end
                
                % Optional: Apply spatial Gaussian filtering to CDFs across grid nodes
                if obj.FilterCDFSpatial
                    for v_idx = 1:256
                        T(v_idx, :, :) = obj.spatialFilter2D(squeeze(T(v_idx, :, :)));
                    end
                end
                
                % Step 2: Vectorized Bilinear Interpolation of lookup mappings
                % Compute indices for sub2ind mapping
                v = channel_rounded + 1; % Convert 0:255 values to 1:256 indices
                
                % sub2ind values for the 4 neighboring centers
                idx00 = sub2ind([256, M, N], v, r,     c);
                idx01 = sub2ind([256, M, N], v, r,     c + 1);
                idx10 = sub2ind([256, M, N], v, r + 1, c);
                idx11 = sub2ind([256, M, N], v, r + 1, c + 1);
                
                % Extract mapped values from T
                v00 = T(idx00);
                v01 = T(idx01);
                v10 = T(idx10);
                v11 = T(idx11);
                
                % Bilinear interpolation blending (Equation 3.10)
                channel_out = (1.0 - s) .* (1.0 - t) .* v00 + ...
                              s .* (1.0 - t) .* v01 + ...
                              (1.0 - s) .* t .* v10 + ...
                              s .* t .* v11;
                
                if is_normalized
                    channel_out = channel_out / 255.0;
                end
                
                I_res(:, :, chan) = channel_out;
            end
            
            if is_uint8
                I_out = uint8(I_res);
            else
                I_out = I_res;
            end
        end
    end
    
    methods (Access = private)
        function grid_out = spatialFilter2D(obj, grid_in)
            % spatialFilter2D  Applies a self-contained 2D Gaussian filter to grid_in.
            [m, n] = size(grid_in);
            grid_out = grid_in;
            
            % Construct 3x3 Gaussian kernel
            sigma = obj.FilterSigma;
            [X_k, Y_k] = meshgrid(-1:1, -1:1);
            kernel = exp(-(X_k.^2 + Y_k.^2) / (2 * sigma^2));
            kernel = kernel / sum(kernel(:));
            
            % Pad grid by 1 pixel (replicate boundary)
            padded = zeros(m + 2, n + 2);
            padded(2:end-1, 2:end-1) = grid_in;
            padded(1, 2:end-1) = grid_in(1, :);
            padded(end, 2:end-1) = grid_in(end, :);
            padded(:, 1) = padded(:, 2);
            padded(:, end) = padded(:, end-1);
            
            for r_f = 1:m
                for c_f = 1:n
                    patch = padded(r_f:r_f+2, c_f:c_f+2);
                    grid_out(r_f, c_f) = sum(patch(:) .* kernel(:));
                end
            end
        end
    end
end
