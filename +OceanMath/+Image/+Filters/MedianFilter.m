classdef MedianFilter < OceanMath.Image.Filters.INeighborhoodFilter
    % MEDIANFILTER  Applies non-linear spatial median filtering.
    %
    %   Median filtering is highly effective at removing impulse noise
    %   (salt-and-pepper noise) while preserving sharp image edges.
    %
    %   Constructor:
    %       MedianFilter('Size', [kH, kW], 'Boundary', border)
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-17
    
    properties (Access = private)
        Size (1,2) double = [3, 3]
    end
    
    methods
        function obj = MedianFilter(options)
            % MEDIANFILTER Construct a MedianFilter.
            arguments
                options.Size = [3, 3]
                options.Boundary = 'replicate'
                options.ConstantValue = 0.0
            end
            obj = obj@OceanMath.Image.Filters.INeighborhoodFilter(...
                'Boundary', options.Boundary, 'ConstantValue', options.ConstantValue);
            
            sz = options.Size;
            if isscalar(sz)
                sz = [sz, sz];
            end
            if any(mod(sz, 2) == 0) || any(sz <= 0)
                error('MedianFilter:InvalidSize', 'Size must be odd positive integers.');
            end
            obj.Size = sz;
        end
    end
    
    methods (Access = protected)
        function pad_size = getPadSize(obj)
            % GETPADSIZE Calculates the symmetric padding size.
            pad_size = [floor(obj.Size(1)/2), floor(obj.Size(2)/2)];
        end
        
        function I_out = applyFilter(obj, I_padded, I_in)
            % APPLYFILTER Vectorized median calculation over sliding patches.
            [H, W, C] = size(I_in);
            kH = obj.Size(1);
            kW = obj.Size(2);
            
            pad_rows = floor(kH / 2);
            pad_cols = floor(kW / 2);
            [H_pad, ~, ~] = size(I_padded);
            
            % Generate pixel coordinate grid for the output image
            [R, C_grid] = ndgrid(1:H, 1:W);
            
            % Locate the center coordinates of each neighborhood in the padded image
            R_cen = R + pad_rows;
            C_cen = C_grid + pad_cols;
            
            % Map 2D coordinates to 1D linear indices in a single channel
            cen_indices = R_cen + (C_cen - 1) * H_pad;
            
            % Generate relative offsets for a neighborhood window
            [d_r, d_c] = ndgrid(-pad_rows:pad_rows, -pad_cols:pad_cols);
            offsets = d_r + d_c * H_pad;
            
            % Form the patch indices matrix [kH*kW x H*W]
            patch_indices = cen_indices(:)' + offsets(:);
            
            % Loop through each channel to extract patches and compute median
            I_out = zeros(H, W, C);
            for ch = 1:C
                I_ch_padded = double(I_padded(:, :, ch));
                patches = I_ch_padded(patch_indices);
                
                % Compute the median along the rows (first dimension)
                I_out(:, :, ch) = reshape(median(patches, 1), H, W);
            end
        end
    end
end
