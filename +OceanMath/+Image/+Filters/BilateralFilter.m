classdef BilateralFilter < OceanMath.Image.Filters.INeighborhoodFilter
    % BILATERALFILTER  Applies non-linear, edge-preserving bilateral smoothing.
    %
    %   Smooths flat regions while preserving sharp edges by combining spatial
    %   closeness weights with range (intensity) similarity weights. Supports
    %   both direct spatial-domain search and the fast 3D Bilateral Grid
    %   lifting trick for large images.
    %
    %   Constructor:
    %       BilateralFilter('SpatialSigma', ss, 'RangeSigma', sr, 'Method', method)
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-17
    
    properties (Access = public)
        SpatialSigma    (1,1) double {mustBePositive} = 3.0
        RangeSigma      (1,1) double {mustBePositive} = 0.1
        Method          char {mustBeMember(Method, {'direct', 'grid'})} = 'grid'
        SpatialSampling       = [] % Empty by default (defaults to SpatialSigma)
        RangeSampling         = [] % Empty by default (defaults to RangeSigma)
    end
    
    methods
        function obj = BilateralFilter(options)
            % BILATERALFILTER Construct a BilateralFilter.
            arguments
                options.SpatialSigma    (1,1) double = 3.0
                options.RangeSigma      (1,1) double = 0.1
                options.Method          char = 'grid'
                options.SpatialSampling      = []
                options.RangeSampling        = []
                options.Boundary        = 'replicate'
                options.ConstantValue   = 0.0
            end
            obj = obj@OceanMath.Image.Filters.INeighborhoodFilter(...
                'Boundary', options.Boundary, 'ConstantValue', options.ConstantValue);
            
            obj.SpatialSigma = options.SpatialSigma;
            obj.RangeSigma = options.RangeSigma;
            obj.Method = lower(options.Method);
            obj.SpatialSampling = options.SpatialSampling;
            obj.RangeSampling = options.RangeSampling;
        end
    end
    
    methods (Access = protected)
        function pad_size = getPadSize(obj)
            % GETPADSIZE Determines padding size based on the execution path.
            if strcmp(obj.Method, 'direct')
                % Use 3*sigma window size for direct search coverage
                pad = ceil(3 * obj.SpatialSigma);
                pad_size = [pad, pad];
            else
                % Grid mode handles boundaries internally via coordinate clamping
                pad_size = [0, 0];
            end
        end
        
        function I_out = applyFilter(obj, I_padded, I_in)
            % APPLYFILTER Dispatches to direct or grid-based bilateral filter.
            [H, W, C] = size(I_in);
            
            % Save and normalize datatype ranges
            is_uint8 = isa(I_in, 'uint8');
            
            if strcmp(obj.Method, 'direct')
                % --- DIRECT SPATIAL DOMAIN METHOD ---
                pad = ceil(3 * obj.SpatialSigma);
                k_size = 2 * pad + 1;
                
                % Pre-calculate spatial weight matrix
                [dX, dY] = meshgrid(-pad:pad, -pad:pad);
                w_spatial = exp(-(dX.^2 + dY.^2) / (2 * obj.SpatialSigma^2));
                w_spatial = w_spatial(:);
                
                % Compute patch indexing arrays
                [R, C_grid] = ndgrid(1:H, 1:W);
                R_cen = R + pad;
                C_cen = C_grid + pad;
                [H_pad, ~, ~] = size(I_padded);
                cen_indices = R_cen + (C_cen - 1) * H_pad;
                
                [d_r, d_c] = ndgrid(-pad:pad, -pad:pad);
                offsets = d_r + d_c * H_pad;
                patch_indices = cen_indices(:)' + offsets(:);
                
                I_out = zeros(H, W, C);
                
                % Apply filter channel-by-channel
                for ch = 1:C
                    I_ch_padded = double(I_padded(:, :, ch));
                    I_ch_in = double(I_in(:, :, ch));
                    
                    if is_uint8
                        % Map to [0, 1] range for sigma similarity consistency
                        I_ch_padded = I_ch_padded / 255.0;
                        I_ch_in = I_ch_in / 255.0;
                    end
                    
                    I_out_ch = zeros(H, W);
                    
                    % Process column-by-column to avoid out-of-memory errors on large images
                    for c = 1:W
                        col_cen_indices = cen_indices(:, c);
                        col_patch_indices = col_cen_indices' + offsets(:);
                        
                        col_patches = I_ch_padded(col_patch_indices);
                        col_centers = I_ch_in(:, c)';
                        
                        % Calculate range weights
                        diffs = col_patches - col_centers;
                        w_range = exp(-diffs.^2 / (2 * obj.RangeSigma^2));
                        
                        % Combine weights
                        w_total = w_spatial .* w_range;
                        w_sum = sum(w_total, 1);
                        w_sum(w_sum < 1e-12) = 1.0;
                        
                        % Normalize and compute output row values
                        val_sum = sum(w_total .* col_patches, 1);
                        I_out_ch(:, c) = val_sum ./ w_sum;
                    end
                    
                    if is_uint8
                        I_out(:, :, ch) = I_out_ch * 255.0;
                    else
                        I_out(:, :, ch) = I_out_ch;
                    end
                end
            else
                % --- BILATERAL GRID METHOD ---
                if is_uint8
                    I_norm = double(I_padded) / 255.0;
                else
                    I_norm = double(I_padded);
                end
                
                % Configure sampling rates
                s_s = obj.SpatialSampling;
                if isempty(s_s)
                    s_s = max(1.0, obj.SpatialSigma);
                end
                s_r = obj.RangeSampling;
                if isempty(s_r)
                    s_r = max(0.01, obj.RangeSigma);
                end
                
                % Delegate to the fast BilateralGrid dimensional lifting engine
                grid_engine = OceanMath.Image.Lifting.BilateralGrid(...
                    'SpatialSampling', s_s, 'RangeSampling', s_r);
                
                I_out_norm = grid_engine.filter(I_norm, obj.SpatialSigma, obj.RangeSigma);
                
                if is_uint8
                    I_out = I_out_norm * 255.0;
                else
                    I_out = I_out_norm;
                end
            end
        end
    end
end
