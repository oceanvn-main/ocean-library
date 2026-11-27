classdef BilateralFilter < OceanMath.Image.Pixels.IPixelTransform
    % BILATERALFILTER  Edge-preserving smoothing and noise removal.
    %
    %   Smooths localized noise while preserving sharp boundaries by combining
    %   spatial distance weights and color range similarity weights.
    %
    %   Constructor:
    %       BilateralFilter('WindowSize', [Wy, Wx], 'SigmaSpatial', s_s, 'SigmaRange', s_r, 'BoundaryMode', mode)
    %
    %   Inputs:
    %       WindowSize   - [1 x 2] dimensions of the local window (default: [5, 5]).
    %       SigmaSpatial - Standard deviation of the spatial domain Gaussian (default: 3.0).
    %       SigmaRange   - Standard deviation of the color range similarity Gaussian (default: 0.1).
    %                      Note: Range weights are computed on normalized [0, 1] color space.
    %       BoundaryMode - Padding strategy ('clamp', 'zero', 'wrap', 'mirror') (default: 'clamp').
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-11
    
    properties (Access = private)
        WindowSize   (1,2) double = [5, 5]
        SigmaSpatial (1,1) double = 3.0
        SigmaRange   (1,1) double = 0.1
        BoundaryMode (1,1) string = 'clamp'
    end
    
    methods
        function obj = BilateralFilter(options)
            % BILATERALFILTER Construct a bilateral filter operator.
            arguments
                options.WindowSize   (1,2) double = [5, 5]
                options.SigmaSpatial (1,1) double = 3.0
                options.SigmaRange   (1,1) double = 0.1
                options.BoundaryMode (1,1) string = 'clamp'
            end
            obj.WindowSize = options.WindowSize;
            obj.SigmaSpatial = options.SigmaSpatial;
            obj.SigmaRange = options.SigmaRange;
            obj.BoundaryMode = options.BoundaryMode;
            
            % Ensure parameters are valid
            if any(mod(obj.WindowSize, 2) == 0) || any(obj.WindowSize <= 0)
                error('Window size dimensions must be positive odd integers.');
            end
            if obj.SigmaSpatial <= 0 || obj.SigmaRange <= 0
                error('SigmaSpatial and SigmaRange parameters must be positive.');
            end
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Performs the edge-preserving bilateral filtering.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            is_uint8 = isa(I_in, 'uint8');
            I_double = double(I_in);
            
            % Normalize to [0, 1] internally for robust range weight calculation
            is_normalized = ~is_uint8 && max(I_double(:)) <= 1.0;
            if ~is_normalized
                I_double = I_double / 255.0;
            end
            
            [height, width, num_channels] = size(I_double);
            I_res = zeros(size(I_double));
            
            Wy = obj.WindowSize(1);
            Wx = obj.WindowSize(2);
            py = floor(Wy / 2);
            px = floor(Wx / 2);
            
            % Precompute spatial distance weights
            [X_w, Y_w] = meshgrid(-px:px, -py:py);
            spatial_weights = exp(-(X_w.^2 + Y_w.^2) / (2 * obj.SigmaSpatial^2));
            
            % Pre-pad using BoundaryPadder
            padded_img = OceanMath.Image.Pixels.BoundaryPadder.pad(I_double, [py, px], obj.BoundaryMode);
            
            % Accumulation matrices for weighted average
            numerator = zeros(height, width, num_channels);
            denominator = zeros(height, width);
            
            % Vectorized shift-and-accumulate loop (only loops over window dimensions)
            for dy = -py:py
                for dx = -px:px
                    w_spatial = spatial_weights(dy + py + 1, dx + px + 1);
                    
                    % Extract the shifted image patch
                    I_shift = padded_img(py + 1 + dy : py + height + dy, px + 1 + dx : px + width + dx, :);
                    
                    % Compute range weight based on L2 distance in color space (RGB-safe)
                    diff_sq = sum((I_shift - I_double).^2, 3);
                    w_range = exp(-diff_sq / (2 * obj.SigmaRange^2));
                    
                    % Combined bilateral weight
                    w = w_spatial * w_range;
                    
                    % Accumulate numerator and denominator
                    numerator = numerator + bsxfun(@times, I_shift, w);
                    denominator = denominator + w;
                end
            end
            
            % Normalize values and divide
            denominator(denominator == 0) = 1e-5;
            for c = 1:num_channels
                I_res(:, :, c) = numerator(:, :, c) ./ denominator;
            end
            
            % Denormalize back to original scale if needed
            if ~is_normalized
                I_res = I_res * 255.0;
            end
            
            if is_uint8
                I_out = uint8(max(0, min(255, I_res)));
            else
                I_out = I_res;
            end
        end
    end
end
