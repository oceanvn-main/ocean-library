classdef SpatialFilter < OceanMath.Image.Pixels.IPixelTransform
    % SPATIALFILTER  Applies 2D neighborhood linear filtering (convolution) from scratch.
    %
    %   Applies a spatial convolution/correlation kernel to the image.
    %   It works completely from scratch without using imfilter.
    %
    %   Constructor:
    %       SpatialFilter('Kernel', h, 'Boundary', border)
    %
    %   Inputs:
    %       Kernel   - An IFilterKernel, a numeric matrix, or a string name.
    %                  Default: [3 x 3] average box blur.
    %       Boundary - Boundary padding option (default: 'replicate').
    %
    %   Author: Ocean Library System & Antigravity
    %   Date: 2026-06-19
    
    properties (Access = private)
        Kernel   % Resolved IFilterKernel instance
        Boundary = 'replicate'
    end
    
    methods
        function obj = SpatialFilter(options)
            % SPATIALFILTER Construct a SpatialFilter object.
            arguments
                options.Kernel   = 1/9 * ones(3,3)
                options.Boundary = 'replicate'
            end
            obj.Kernel = OceanMath.Signal.Kernels.KernelFactory.resolve(options.Kernel);
            obj.Boundary = options.Boundary;
            obj.Description = 'Input: Spatial image array (Grayscale/RGB). Output: Spatial image array. Safely applies 2D convolution kernel (e.g. Blur, Edge) without breaking the pipeline.';
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Applies the 2D filter correlation from scratch.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            I_dbl = double(I_in);
            [H, W, C] = size(I_dbl);
            
            % Get concrete 2D kernel matrix
            K = obj.Kernel.getMatrix();
            [Kh, Kw] = size(K);
            
            pad_rows = (Kh - 1) / 2;
            pad_cols = (Kw - 1) / 2;
            
            % Perform boundary padding using our custom BoundaryPadder
            I_pad = OceanMath.Image.Pixels.BoundaryPadder.pad(I_dbl, [pad_rows, pad_cols], obj.Boundary);
            
            % Vectorized 2D correlation summation
            I_out = zeros(H, W, C);
            for r = 1:Kh
                for c = 1:Kw
                    I_out = I_out + K(r, c) * I_pad(r : r + H - 1, c : c + W - 1, :);
                end
            end
            
            % Restore original class type
            if isa(I_in, 'uint8')
                I_out = uint8(I_out);
            elseif isa(I_in, 'single')
                I_out = single(I_out);
            end
        end
    end
end
