classdef ZeroStuffing < OceanMath.Image.Pixels.IPixelTransform
    % ZEROSTUFFING  Injects zeros between pixels to expand/upsample an image.
    %
    %   This operation (zero-stuffing) physically compresses the original
    %   frequency spectrum and creates mirrored high-frequency "ghost" copies
    %   of the data (imaging artifact or synthesis aliasing).
    %   Visually, the result looks like it is covered in a dense, high-frequency
    %   checkerboard grid of black dots.
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-23
    
    properties (Constant)
        Description = "Zero-Stuffing (Expand) upsamples an image by inserting zeros between original pixel samples. In the frequency domain, this physically compresses the spectrum, creating mirrored high-frequency 'ghost' copies of the data. Visually, the spatial image appears covered in a dense, checkerboard-like grid of black dots.";
    end
    
    properties (Access = private)
        Factor (1,1) double = 2
    end
    
    methods
        function obj = ZeroStuffing(options)
            arguments
                options.Factor (1,1) double = 2
            end
            % Set the upsampling/expansion factor (e.g. L=2 means double size)
            obj.Factor = options.Factor;
        end
        
        function I_out = apply(obj, I_in)
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            % Get original image dimensions
            [H, W, C] = size(I_in);
            L = obj.Factor;
            
            % Compute the new upsampled size
            H_out = H * L;
            W_out = W * L;
            
            % Initialize the output image with zeros (black grid)
            % maintaining the exact numeric type (uint8, single, or double)
            if isa(I_in, 'uint8')
                I_out = zeros(H_out, W_out, C, 'uint8');
            elseif isa(I_in, 'single')
                I_out = zeros(H_out, W_out, C, 'single');
            else
                I_out = zeros(H_out, W_out, C);
            end
            
            % Place the original image pixels at L-spaced strides: (1, L+1, 2L+1, ...)
            % The remaining intermediate pixels remain zero, forming the checkerboard grid.
            I_out(1:L:end, 1:L:end, :) = I_in;
        end
    end
end
