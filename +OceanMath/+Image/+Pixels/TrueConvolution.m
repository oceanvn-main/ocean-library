classdef TrueConvolution < OceanMath.Image.Pixels.IPixelTransform
    % TRUECONVOLUTION  Applies 2D neighborhood true convolution from scratch.
    %
    %   The kernel is rotated by 180 degrees (horizontally and vertically)
    %   before being laid on the image patch to perform the sum of products.
    %
    %   Constructor:
    %       TrueConvolution('Kernel', h, 'Boundary', border)
    %
    %   Inputs:
    %       Kernel   - An IFilterKernel, a numeric matrix, or a string name.
    %                  Default: [3 x 3] average box blur.
    %       Boundary - Boundary padding option (default: 'replicate').
    %
    %   Author: Antigravity
    %   Date: 2026-06-19
    
    properties (Access = private)
        Kernel   % Resolved IFilterKernel instance
        Boundary = 'replicate'
    end
    
    methods
        function obj = TrueConvolution(options)
            % TRUECONVOLUTION Construct a TrueConvolution filter.
            arguments
                options.Kernel                = 1/9 * ones(3,3)
                options.Boundary              = "replicate"
            end
            obj.Kernel = OceanMath.Signal.Kernels.KernelFactory.resolve(options.Kernel);
            obj.Boundary = options.Boundary;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Applies the 2D mathematical convolution from scratch.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            I_dbl = double(I_in);
            [H, W, C] = size(I_dbl);
            
            if obj.Kernel.isSeparable()
                % 1. Separable filtering
                [h, v] = obj.Kernel.get1DDecomposition();
                
                % In true convolution, flip the 1D components
                h_conv = fliplr(h);
                v_conv = flipud(v);
                
                % Horizontal Pass:
                Kh = length(h_conv);
                pad_cols = (Kh - 1) / 2;
                I_pad_h = OceanMath.Image.Pixels.BoundaryPadder.pad(I_dbl, [0, pad_cols], obj.Boundary);
                I_h = zeros(H, W, C);
                for m = 1:Kh
                    I_h = I_h + h_conv(m) * I_pad_h(:, m : m + W - 1, :);
                end
                
                % Vertical Pass:
                Kv = length(v_conv);
                pad_rows = (Kv - 1) / 2;
                I_pad_v = OceanMath.Image.Pixels.BoundaryPadder.pad(I_h, [pad_rows, 0], obj.Boundary);
                I_out = zeros(H, W, C);
                for m = 1:Kv
                    I_out = I_out + v_conv(m) * I_pad_v(m : m + H - 1, :, :);
                end
            else
                % 2. Standard 2D convolution (Non-separable)
                K = obj.Kernel.getMatrix();
                
                % In true convolution, rotate kernel by 180 degrees
                K_conv = rot90(K, 2);
                [Kh, Kw] = size(K_conv);
                
                pad_rows = (Kh - 1) / 2;
                pad_cols = (Kw - 1) / 2;
                
                I_pad = OceanMath.Image.Pixels.BoundaryPadder.pad(I_dbl, [pad_rows, pad_cols], obj.Boundary);
                I_out = zeros(H, W, C);
                for r = 1:Kh
                    for c = 1:Kw
                        I_out = I_out + K_conv(r, c) * I_pad(r : r + H - 1, c : c + W - 1, :);
                    end
                end
            end
            
            % Restore class type of the input
            if isa(I_in, 'uint8')
                I_out = uint8(I_out);
            elseif isa(I_in, 'single')
                I_out = single(I_out);
            end
        end
        
        function h = getKernel(obj)
            % GETKERNEL Returns the active filter kernel object.
            h = obj.Kernel;
        end
        
        function b = getBoundary(obj)
            % GETBOUNDARY Returns the active boundary mode.
            b = obj.Boundary;
        end
    end
end
