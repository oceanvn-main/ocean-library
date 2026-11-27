classdef NonLocalMeansFilter < OceanMath.Image.Filters.INeighborhoodFilter
    % NONLOCALMEANSFILTER  Applies Non-Local Means (NL-Means) patch-based denoising.
    %
    %   Smooths an image by performing a weighted average of pixels with 
    %   similar neighborhood patches. Implements a fast O(N) vectorized shift
    %   method using box-filter convolutions of squared differences.
    %
    %   Constructor:
    %       NonLocalMeansFilter('SearchRadius', sr, 'PatchRadius', pr, 'FilteringParameter', h)
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-16
    
    properties (Access = public)
        SearchRadius       (1,1) double {mustBeInteger, mustBeNonnegative} = 7
        PatchRadius        (1,1) double {mustBeInteger, mustBeNonnegative} = 2
        FilteringParameter (1,1) double {mustBePositive} = 0.1
    end
    
    methods
        function obj = NonLocalMeansFilter(options)
            % NONLOCALMEANSFILTER Construct a NonLocalMeansFilter.
            arguments
                options.SearchRadius       (1,1) double = 7
                options.PatchRadius        (1,1) double = 2
                options.FilteringParameter (1,1) double = 0.1
                options.Boundary = 'replicate'
                options.ConstantValue = 0.0
            end
            obj = obj@OceanMath.Image.Filters.INeighborhoodFilter(...
                'Boundary', options.Boundary, 'ConstantValue', options.ConstantValue);
            
            obj.SearchRadius = options.SearchRadius;
            obj.PatchRadius = options.PatchRadius;
            obj.FilteringParameter = options.FilteringParameter;
        end
    end
    
    methods (Access = protected)
        function pad_size = getPadSize(obj)
            % GETPADSIZE Subclass hook returning padding size required for shift.
            % We need to pad by SearchRadius + PatchRadius to avoid out-of-bounds indexing.
            pad = obj.SearchRadius + obj.PatchRadius;
            pad_size = [pad, pad];
        end
        
        function I_out = applyFilter(obj, I_padded, I_in)
            % APPLYFILTER Subclass hook performing fast patch-based NL-Means.
            [H, W, C] = size(I_in);
            
            % Normalize color values for filtering parameter consistency
            is_uint8 = isa(I_in, 'uint8');
            if is_uint8
                I_norm = double(I_in) / 255.0;
                I_pad_norm = double(I_padded) / 255.0;
            else
                I_norm = double(I_in);
                I_pad_norm = double(I_padded);
            end
            
            pad = obj.SearchRadius + obj.PatchRadius;
            P = obj.PatchRadius;
            S = obj.SearchRadius;
            h2 = obj.FilteringParameter^2;
            
            % Precompute box filter kernel
            k_size = 2 * P + 1;
            box_kernel = ones(k_size, k_size) / (k_size^2);
            
            Accum = zeros(H, W, C);
            Norm = zeros(H, W);
            
            % Extract reference padded patch zone
            row_start_ref = pad + 1 - P;
            row_end_ref   = pad + H + P;
            col_start_ref = pad + 1 - P;
            col_end_ref   = pad + W + P;
            I_pad_ref = I_pad_norm(row_start_ref:row_end_ref, col_start_ref:col_end_ref, :);
            
            % Loop over search window shifts
            for dy = -S:S
                for dx = -S:S
                    % Extract shifted padded patch zone
                    row_start_sh = pad + 1 + dy - P;
                    row_end_sh   = pad + H + dy + P;
                    col_start_sh = pad + 1 + dx - P;
                    col_end_sh   = pad + W + dx + P;
                    I_pad_shift = I_pad_norm(row_start_sh:row_end_sh, col_start_sh:col_end_sh, :);
                    
                    % Compute squared patch differences
                    diff2_pad = (I_pad_ref - I_pad_shift).^2;
                    
                    % Box filter the squared differences to sum over the patch
                    patch_dist_sq = zeros(H, W);
                    for ch = 1:C
                        patch_dist_sq = patch_dist_sq + conv2(diff2_pad(:, :, ch), box_kernel, 'valid');
                    end
                    
                    % Compute weights
                    w = exp(-patch_dist_sq / h2);
                    
                    % Extract shifted center pixel image
                    I_shift_center = I_pad_norm(pad + 1 + dy : pad + H + dy, pad + 1 + dx : pad + W + dx, :);
                    
                    % Accumulate
                    for ch = 1:C
                        Accum(:, :, ch) = Accum(:, :, ch) + w .* I_shift_center(:, :, ch);
                    end
                    Norm = Norm + w;
                end
            end
            
            % Normalize weighted sum
            Norm(Norm < 1e-12) = 1.0;
            I_out_double = zeros(H, W, C);
            for ch = 1:C
                I_out_double(:, :, ch) = Accum(:, :, ch) ./ Norm;
            end
            
            % Rescale back to uint8 range if needed
            if is_uint8
                I_out = I_out_double * 255.0;
            else
                I_out = I_out_double;
            end
        end
    end
end
