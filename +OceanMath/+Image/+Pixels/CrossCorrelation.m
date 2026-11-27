classdef CrossCorrelation < OceanMath.Image.Pixels.IPixelTransform
    % CROSSCORRELATION  Applies 2D neighborhood cross-correlation from scratch.
    %
    %   The kernel is laid directly on top of the image patch, and the overlap
    %   is summed using a direct dot product (without 180-degree rotation).
    %
    %   Constructor:
    %       CrossCorrelation('Kernel', h, 'Boundary', border)
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
        function obj = CrossCorrelation(options)
            % CROSSCORRELATION Construct a CrossCorrelation filter.
            arguments
                options.Kernel                = 1/9 * ones(3,3)
                options.Boundary              = "replicate"
            end
            obj.Kernel = OceanMath.Signal.Kernels.KernelFactory.resolve(options.Kernel);
            obj.Boundary = options.Boundary;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Applies the 2D cross-correlation filter from scratch.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            I_dbl = double(I_in);
            [H, W, C] = size(I_dbl);
            
            if obj.Kernel.isSeparable()
                % 1. Separable filtering
                [h, v] = obj.Kernel.get1DDecomposition();
                
                % Horizontal Pass (using direct 1D kernel h):
                Kh = length(h);
                pad_cols = (Kh - 1) / 2;
                I_pad_h = OceanMath.Image.Pixels.BoundaryPadder.pad(I_dbl, [0, pad_cols], obj.Boundary);
                I_h = zeros(H, W, C);
                for m = 1:Kh
                    I_h = I_h + h(m) * I_pad_h(:, m : m + W - 1, :);
                end
                
                % Vertical Pass (using direct 1D kernel v):
                Kv = length(v);
                pad_rows = (Kv - 1) / 2;
                I_pad_v = OceanMath.Image.Pixels.BoundaryPadder.pad(I_h, [pad_rows, 0], obj.Boundary);
                I_out = zeros(H, W, C);
                for m = 1:Kv
                    I_out = I_out + v(m) * I_pad_v(m : m + H - 1, :, :);
                end
            else
                % 2. Standard 2D cross-correlation (Non-separable)
                K = obj.Kernel.getMatrix();
                [Kh, Kw] = size(K);
                
                pad_rows = (Kh - 1) / 2;
                pad_cols = (Kw - 1) / 2;
                
                I_pad = OceanMath.Image.Pixels.BoundaryPadder.pad(I_dbl, [pad_rows, pad_cols], obj.Boundary);
                I_out = zeros(H, W, C);
                for r = 1:Kh
                    for c = 1:Kw
                        I_out = I_out + K(r, c) * I_pad(r : r + H - 1, c : c + W - 1, :);
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
