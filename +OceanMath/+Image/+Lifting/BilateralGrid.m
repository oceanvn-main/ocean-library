classdef BilateralGrid < handle
    % BILATERALGRID  Advanced 3D Bilateral Grid computational engine.
    %
    %   Lifts a 2D image with coordinates (x, y) and range (intensity) values
    %   into a 3D grid space: ( [x / s_s], [y / s_s], [I(x,y) / s_r] ). In this
    %   higher-dimensional grid space, the non-linear, space-variant bilateral
    %   filter is computed via a linear, space-invariant 3D Gaussian blur.
    %   Slicing the grid back to 2D using trilinear interpolation yields the
    %   final edge-preserving smoothed result in constant time O(1).
    %
    %   Constructor:
    %       BilateralGrid('SpatialSampling', s_s, 'RangeSampling', s_r)
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-17
    
    properties (Access = public)
        SpatialSampling (1,1) double {mustBePositive} = 4.0
        RangeSampling   (1,1) double {mustBePositive} = 0.1
    end
    
    methods
        function obj = BilateralGrid(options)
            % BILATERALGRID Construct a BilateralGrid engine.
            arguments
                options.SpatialSampling (1,1) double = 4.0
                options.RangeSampling   (1,1) double = 0.1
            end
            obj.SpatialSampling = options.SpatialSampling;
            obj.RangeSampling = options.RangeSampling;
        end
        
        function I_out = filter(obj, I_in, spatial_sigma, range_sigma)
            % FILTER Applies 3D Bilateral Grid filtering to the image.
            %   I_in must be a double array normalized to [0, 1].
            arguments
                obj
                I_in           double
                spatial_sigma  (1,1) double {mustBePositive}
                range_sigma    (1,1) double {mustBePositive}
            end
            
            % Get input image dimensions (H = height, W = width, C = color channels)
            [H, W, C] = size(I_in);
            s_s = obj.SpatialSampling; %s_s smaller than original image in both dimension
            s_r = obj.RangeSampling; %We need s_r because of mismatch between intensity and pixel
            
            % Compute the dimensions of the downsampled 3D grid.
            % Downsampling spatially by s_s and along the range by s_r shrinks the grid footprint
            % from H x W x Bins to (H/s_s) x (W/s_s) x (1/s_r), reducing cells from millions to
            % thousands, which makes the subsequent 3D linear Gaussian convolution O(1) in speed.
            max_range = H - 1;
            step_size = s_s;
            interval_need = ceil(max_range / step_size);

            grid_h = interval_need + 1;

            grid_w = ceil((W - 1) / s_s) + 1;
            grid_z = ceil(1.0 / s_r) + 1;
            
            % Generate coordinate arrays for mapping pixels to the grid
            [R, C_grid] = ndgrid(1:H, 1:W); % R hold rows indices from 1:H, c_grid hold col indices from 1:W
            r_vec = R(:);
            c_vec = C_grid(:);
            
            % Map pixel coordinates to continuous coordinates in grid-space.
            % Mesgrid/interp3 uses (X = Columns, Y = Rows, Z = Range slices) convention.
            Xq = (c_vec - 1) / s_s + 1;
            Yq = (r_vec - 1) / s_s + 1;
            
            % Clamp spatial coordinate matrices to grid boundaries to prevent out-of-bounds NaNs
            % during interpolation (slicing) later.
            Xq_mat = reshape(max(1.0, min(double(grid_w), Xq)), H, W);
            Yq_mat = reshape(max(1.0, min(double(grid_h), Yq)), H, W);
            
            I_out = zeros(H, W, C);
            
            % Process each color channel independently
            for ch = 1:C
                I_ch = I_in(:, :, ch);
                val_vec = I_ch(:);
                
                % Map dynamic intensity values to continuous range coordinates in grid-space:
                % Intensities [0, 1] are mapped to grid coordinate range [1, grid_z].
                Zq = val_vec / s_r + 1; % map from [0..1] to [1 ... intensity / RangeSampling]
                Zq_mat = reshape(max(1.0, min(double(grid_z), Zq)), H, W); % Reshape from 1D to 2D matrix H x W and ensure valid range
                
                % --- STEP 1: SPLATTING (Lifting 2D pixels to 3D Grid) ---
                % Identify the enclosing integer coordinates of the 8 grid cell corners
                x0 = floor(Yq); x1 = x0 + 1;
                y0 = floor(Xq); y1 = y0 + 1;
                z0 = floor(Zq); z1 = z0 + 1;
                
                % Clamp subscripts to prevent index bounds violations on the grid edges
                x0_c = max(1, min(grid_h, x0)); x1_c = max(1, min(grid_h, x1));
                y0_c = max(1, min(grid_w, y0)); y1_c = max(1, min(grid_w, y1));
                z0_c = max(1, min(grid_z, z0)); z1_c = max(1, min(grid_z, z1));
                
                % Calculate trilinear interpolation fractional weights along all three axes
                wx1 = Yq - x0; wx0 = 1 - wx1;
                wy1 = Xq - y0; wy0 = 1 - wy1;
                wz1 = Zq - z0; wz0 = 1 - wz1;
                
                % Product of coordinate weights yields the 8 corners' splatting weights
                w000 = wx0 .* wy0 .* wz0;
                w100 = wx1 .* wy0 .* wz0;
                w010 = wx0 .* wy1 .* wz0;
                w110 = wx1 .* wy1 .* wz0;
                w001 = wx0 .* wy0 .* wz1;
                w101 = wx1 .* wy0 .* wz1;
                w011 = wx0 .* wy1 .* wz1;
                w111 = wx1 .* wy1 .* wz1;
                
                % Form composite subscript matrices (X, Y, Z coordinate indices in 3D grid)
                subs = [ ...
                    x0_c, y0_c, z0_c; ...
                    x1_c, y0_c, z0_c; ...
                    x0_c, y1_c, z0_c; ...
                    x1_c, y1_c, z0_c; ...
                    x0_c, y0_c, z1_c; ...
                    x1_c, y0_c, z1_c; ...
                    x0_c, y1_c, z1_c; ...
                    x1_c, y1_c, z1_c ...
                ];
                
                % Concatenate weight lists and values scaled by their respective weights
                weights = [w000; w100; w010; w110; w001; w101; w011; w111];
                vals = [val_vec; val_vec; val_vec; val_vec; val_vec; val_vec; val_vec; val_vec] .* weights;
                
                % Vectorized splat accumulation: sums intensities and weights into 3D grids
                grid_data = accumarray(subs, vals, [grid_h, grid_w, grid_z]);
                grid_weight = accumarray(subs, weights, [grid_h, grid_w, grid_z]);
                
                % --- STEP 2: BLURRING (Linear Smoothing in 3D Space) ---
                % Scale the spatial and range filter sigmas into grid units
                sigma_s_g = spatial_sigma / s_s;
                sigma_r_g = range_sigma / s_r;
                
                % Smooth both data and weight grids linearly. By treating range as a coordinate,
                % the non-linear range blur is converted into a linear 3D separable Gaussian blur.
                grid_data_blurred = obj.blur3D(grid_data, sigma_s_g, sigma_r_g);
                grid_weight_blurred = obj.blur3D(grid_weight, sigma_s_g, sigma_r_g);
                
                % --- STEP 3: SLICING (Reconstruction back to 2D) ---
                % Query the low-frequency blurred 3D grids at the continuous pixel coordinates,
                % performing trilinear interpolation to reconstruct a full-resolution 2D image.
                I_data_sliced = interp3(grid_data_blurred, Xq_mat, Yq_mat, Zq_mat, 'linear');
                I_weight_sliced = interp3(grid_weight_blurred, Xq_mat, Yq_mat, Zq_mat, 'linear');
                
                % Enforce the sum-of-weights = 1 normalization step of the bilateral formula.
                % Divide the interpolated blurred values by their corresponding weights.
                valid = I_weight_sliced > 1e-5;
                I_out_ch = I_ch;
                I_out_ch(valid) = I_data_sliced(valid) ./ I_weight_sliced(valid);
                I_out(:, :, ch) = I_out_ch;
            end
        end
    end
    
    methods (Access = private)
        function blurred = blur3D(obj, grid, sigma_s_g, sigma_r_g)
            % BLUR3D Performs separable 1D Gaussian blurring along rows, columns, and range.
            blurred = grid;
            
            % Blur along Dimension 1 (Rows / Vertical)
            if sigma_s_g > 0
                ks = obj.makeGaussian1D(sigma_s_g);
                blurred = imfilter(blurred, ks(:), 'replicate');
            end
            % Blur along Dimension 2 (Columns / Horizontal)
            if sigma_s_g > 0
                ks = obj.makeGaussian1D(sigma_s_g);
                blurred = imfilter(blurred, ks(:)', 'replicate');
            end
            % Blur along Dimension 3 (Range / Z-axis)
            if sigma_r_g > 0
                kr = obj.makeGaussian1D(sigma_r_g);
                blurred = imfilter(blurred, reshape(kr, 1, 1, []), 'replicate');
            end
        end
        
        function k = makeGaussian1D(~, sigma)
            % MAKEGAUSSIAN1D Generates a 1D normalized Gaussian kernel.
            limit = ceil(3 * sigma);
            x = -limit:limit;
            k = exp(-x.^2 / (2 * sigma^2));
            k = k / sum(k);
        end
    end
end
