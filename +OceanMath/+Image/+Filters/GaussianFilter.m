classdef GaussianFilter < OceanMath.Image.Filters.INeighborhoodFilter
    % GAUSSIANFILTER  Applies linear 2D Gaussian smoothing convolution.
    %
    %   Smooths an image by convolving it with a circular symmetric Gaussian 
    %   kernel. This represents the classic linear scale-space filter.
    %
    %   Constructor:
    %       GaussianFilter('Sigma', sigma)
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-16
    
    properties (Access = public)
        Sigma (1,1) double {mustBePositive} = 1.0
    end
    
    methods
        function obj = GaussianFilter(options)
            % GAUSSIANFILTER Construct a GaussianFilter.
            arguments
                options.Sigma (1,1) double = 1.0
                options.Boundary = 'replicate'
                options.ConstantValue = 0.0
            end
            obj = obj@OceanMath.Image.Filters.INeighborhoodFilter(...
                'Boundary', options.Boundary, 'ConstantValue', options.ConstantValue);
            
            obj.Sigma = options.Sigma;
        end
    end
    
    methods (Access = protected)
        function pad_size = getPadSize(obj)
            % GETPADSIZE Subclass hook to return boundary padding size.
            pad = ceil(3 * obj.Sigma);
            pad_size = [pad, pad];
        end
        
        function I_out = applyFilter(obj, I_padded, I_in)
            % APPLYFILTER Subclass hook performing the 2D Gaussian convolution.
            [H, W, C] = size(I_in);
            
            pad = ceil(3 * obj.Sigma);
            [X, Y] = meshgrid(-pad:pad, -pad:pad);
            
            % Compute 2D Gaussian kernel from scratch
            K = exp(-(X.^2 + Y.^2) / (2 * obj.Sigma^2));
            K = K / sum(K(:));
            
            [Kh, Kw] = size(K);
            
            I_pad_dbl = double(I_padded);
            
            % Apply 2D correlation summation (fully vectorized over pixels)
            I_out = zeros(H, W, C);
            for r = 1:Kh
                for c = 1:Kw
                    I_out = I_out + K(r, c) * I_pad_dbl(r : r + H - 1, c : c + W - 1, :);
                end
            end
        end
    end
end
