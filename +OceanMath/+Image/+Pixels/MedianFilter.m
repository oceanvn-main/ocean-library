classdef MedianFilter < OceanMath.Image.Pixels.IPixelTransform
    % MEDIANFILTER  Non-linear sliding window median filter for noise removal.
    %
    %   Replaces each pixel value with the median of its neighborhood window.
    %   Highly effective at removing impulse (salt-and-pepper) noise while
    %   preserving sharp edges.
    %
    %   Constructor:
    %       MedianFilter('WindowSize', [Wy, Wx], 'BoundaryMode', mode)
    %
    %   Inputs:
    %       WindowSize   - [1 x 2] dimensions of the local window (default: [3, 3]).
    %       BoundaryMode - Padding strategy ('clamp', 'zero', 'wrap', 'mirror') (default: 'clamp').
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-11
    
    properties (Access = private)
        WindowSize   (1,2) double = [3, 3]
        BoundaryMode (1,1) string = 'clamp'
    end
    
    methods
        function obj = MedianFilter(options)
            % MEDIANFILTER Construct a median filter operator.
            arguments
                options.WindowSize   (1,2) double = [3, 3]
                options.BoundaryMode (1,1) string = 'clamp'
            end
            obj.WindowSize = options.WindowSize;
            obj.BoundaryMode = options.BoundaryMode;
            
            % Ensure window dimensions are positive odd integers
            if any(mod(obj.WindowSize, 2) == 0) || any(obj.WindowSize <= 0)
                error('Window size dimensions must be positive odd integers.');
            end
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Applies median filtering to each channel of the input image.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            is_uint8 = isa(I_in, 'uint8');
            I_double = double(I_in);
            
            [height, width, num_channels] = size(I_double);
            I_res = zeros(size(I_double));
            
            Wy = obj.WindowSize(1);
            Wx = obj.WindowSize(2);
            py = floor(Wy / 2);
            px = floor(Wx / 2);
            
            for chan = 1:num_channels
                channel_data = I_double(:, :, chan);
                
                % Pre-pad using BoundaryPadder
                padded_channel = OceanMath.Image.Pixels.BoundaryPadder.pad(channel_data, [py, px], obj.BoundaryMode);
                
                % Vectorized shift-and-stack approach for neighborhood median
                S = zeros(height, width, Wy * Wx);
                count = 0;
                for dy = -py:py
                    for dx = -px:px
                        count = count + 1;
                        S(:, :, count) = padded_channel(py + 1 + dy : py + height + dy, px + 1 + dx : px + width + dx);
                    end
                end
                
                % Compute median across the neighborhood dimension
                I_res(:, :, chan) = median(S, 3);
            end
            
            if is_uint8
                I_out = uint8(I_res);
            else
                I_out = I_res;
            end
        end
    end
end
