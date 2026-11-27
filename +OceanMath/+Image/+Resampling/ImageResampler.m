classdef ImageResampler < handle
    % IMAGERESAMPLER  High-quality separable image resizing and resampling.
    %
    %   Supports upscaling (magnification) and downscaling (minification / decimation)
    %   using various reconstruction filters: Nearest, Bilinear, Bicubic, and Lanczos.
    %
    %   Constructor:
    %       resampler = ImageResampler('TargetSize', [H, W], 'FilterType', filter)
    %
    %   Inputs:
    %       TargetSize - [1 x 2] target dimensions [Height, Width].
    %       FilterType - Interpolation filter ('Nearest', 'Bilinear', 'Bicubic', 'Lanczos') (default: 'Lanczos').
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-13
    
    properties (Access = private)
        TargetSize (1,2) double = [128, 128]
        FilterType (1,1) string = 'Lanczos'
    end
    
    methods
        function obj = ImageResampler(options)
            % IMAGERESAMPLER Construct a resampler operator.
            arguments
                options.TargetSize (1,2) double = [128, 128]
                options.FilterType (1,1) string = 'Lanczos'
            end
            obj.TargetSize = options.TargetSize;
            obj.FilterType = options.FilterType;
            
            valid_filters = ["Nearest", "Bilinear", "Bicubic", "Lanczos", "WindowedSinc"];
            if ~any(valid_filters == obj.FilterType)
                error('FilterType must be one of: Nearest, Bilinear, Bicubic, Lanczos, WindowedSinc.');
            end
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Separably resamples the input image to the TargetSize.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            is_uint8 = isa(I_in, 'uint8');
            I_double = double(I_in);
            
            % Step 1: Resample vertically (resize Height)
            % input: [H_in x W_in x C] -> output: [H_new x W_in x C]
            temp = obj.resample1D(I_double, obj.TargetSize(1));
            
            % Step 2: Swap dimensions to resample horizontally (resize Width)
            % [H_new x W_in x C] -> [W_in x H_new x C]
            temp = permute(temp, [2, 1, 3]);
            
            % Resample along the vertical dimension of transposed matrix
            % [W_in x H_new x C] -> [W_new x H_new x C]
            temp = obj.resample1D(temp, obj.TargetSize(2));
            
            % Step 3: Permute back to standard orientation
            % [W_new x H_new x C] -> [H_new x W_new x C]
            I_res = permute(temp, [2, 1, 3]);
            
            if is_uint8
                I_out = uint8(max(0, min(255, I_res)));
            else
                I_out = I_res;
            end
        end
    end
    
    methods (Access = private)
        function dst = resample1D(obj, src, new_len)
            % Separable 1D Resampler. Stretches kernels during downsampling for anti-aliasing.
            L_in = size(src, 1);
            W = size(src, 2);
            C = size(src, 3);
            dst = zeros(new_len, W, C);
            
            scale = new_len / L_in;
            f = min(1.0, scale); % scale factor (< 1.0 for downsampling / decimation)
            
            switch obj.FilterType
                case 'Nearest'
                    j = (1:new_len)';
                    x_src = (j - 0.5) / scale + 0.5;
                    k = max(1, min(L_in, round(x_src)));
                    dst = src(k, :, :);
                    return;
                case 'Bilinear'
                    R = 1.0;
                    weight_func = @(x) max(0, 1.0 - abs(x));
                case 'Bicubic'
                    R = 2.0;
                    weight_func = @obj.bicubic_kernel;
                case 'Lanczos'
                    R = 3.0;
                    weight_func = @obj.lanczos_kernel;
                case 'WindowedSinc'
                    R = 3.0;
                    weight_func = @obj.windowed_sinc_kernel;
            end
            
            % Calculate window support coordinates
            R_eff = R / f;
            j = (1:new_len)';
            x_src = (j - 0.5) / scale + 0.5;
            
            % Generate neighboring offsets
            offsets = -floor(R_eff) : ceil(R_eff);
            num_neighbors = length(offsets);
            k_base = floor(x_src);
            
            W_mat = zeros(new_len, num_neighbors);
            K_mat = zeros(new_len, num_neighbors);
            
            % Compute weights for the neighborhood
            for idx = 1:num_neighbors
                offset = offsets(idx);
                k = k_base + offset;
                dist = k - x_src;
                % Scale the distance argument to stretch filter (anti-aliasing)
                W_mat(:, idx) = weight_func(dist * f);
                K_mat(:, idx) = max(1, min(L_in, k));
            end
            
            % Normalize weights to prevent brightness shifts
            sum_w = sum(W_mat, 2);
            sum_w(sum_w == 0) = 1e-5;
            W_mat = W_mat ./ sum_w;
            
            % Perform vectorized interpolation gather
            for c = 1:C
                for col = 1:W
                    col_data = src(:, col, c);
                    gather = col_data(K_mat);
                    dst(:, col, c) = sum(gather .* W_mat, 2);
                end
            end
        end
        
        function w = bicubic_kernel(obj, x)
            % Keys' cubic convolution interpolation (a = -0.5)
            ax = abs(x);
            w = zeros(size(x));
            
            idx1 = (ax < 1);
            w(idx1) = 1.5 * ax(idx1).^3 - 2.5 * ax(idx1).^2 + 1;
            
            idx2 = (ax >= 1 & ax < 2);
            w(idx2) = -0.5 * ax(idx2).^3 + 2.5 * ax(idx2).^2 - 4 * ax(idx2) + 2;
        end
        
        function w = lanczos_kernel(obj, x)
            % Lanczos-3 windowed sinc filter
            ax = abs(x);
            w = zeros(size(x));
            idx = (ax < 3);
            w(idx) = obj.sinc(ax(idx)) .* obj.sinc(ax(idx) / 3);
        end
        
        % 3-lobe Hamming Windowed Sinc Kernel
        function w = windowed_sinc_kernel(obj, x)
            ax = abs(x);
            w = zeros(size(x));
            idx = (ax < 3.0);
            % Hamming window function
            window = 0.54 + 0.46 * cos(pi * ax(idx) / 3.0);
            w(idx) = obj.sinc(ax(idx)) .* window;
        end
        
        function s = sinc(obj, x)
            % Normalized sinc function
            s = ones(size(x));
            idx = (x ~= 0);
            s(idx) = sin(pi * x(idx)) ./ (pi * x(idx));
        end
    end
end
