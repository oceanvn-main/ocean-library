classdef Decimation < OceanMath.Image.Pixels.IPixelTransform
    % DECIMATION  Downsamples an image with optional anti-aliasing low-pass filtering.
    %
    %   Performs decimation by keeping every L-th pixel. Can be run blindly
    %   (causes aliasing/moiré artifacts) or with an anti-aliasing low-pass
    %   filter shield first to prevent frequency fold-over.
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-23
    
    properties (Constant)
        Description = "Decimation (Downsample) reduces image resolution by keeping every L-th pixel. If Anti-Aliasing is enabled, a low-pass filter (Gaussian or Average) is applied first to blur away high frequencies, shielding the image from folding artifacts (aliasing). Without it, high frequencies fold back and scramble the low frequencies.";
    end

    properties (Access = private)
        Factor    (1,1) double = 2
        AntiAlias (1,1) logical = true
        FilterType (1,1) string = "Gaussian"
        Boundary = 'replicate'
    end
    
    methods
        function obj = Decimation(options)
            arguments
                options.Factor    (1,1) double = 2
                options.AntiAlias (1,1) logical = true
                options.FilterType (1,1) string = "Gaussian"
                options.Boundary = 'replicate'
            end
            obj.Factor = options.Factor;
            obj.AntiAlias = options.AntiAlias;
            obj.FilterType = options.FilterType;
            obj.Boundary = options.Boundary;
        end
        
        function I_out = apply(obj, I_in)
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            L = obj.Factor;
            I_dbl = double(I_in);
            
            if obj.AntiAlias
                % Define low-pass kernel based on FilterType
                switch obj.FilterType
                    case "Gaussian"
                        % Use standard binomial 5x5 Gaussian
                        g = [1, 4, 6, 4, 1] / 16;
                        K = g' * g;
                    case "Average"
                        % Average box blur based on Factor size (e.g. L x L)
                        K = ones(L, L) / (L^2);
                    otherwise
                        g = [1, 4, 6, 4, 1] / 16;
                        K = g' * g;
                end
                
                % Filter first (spatial correlation/convolution)
                [H, W, C] = size(I_dbl);
                [Kh, Kw] = size(K);
                
                pad_rows = ceil((Kh - 1) / 2);
                pad_cols = ceil((Kw - 1) / 2);
                
                I_pad = OceanMath.Image.Pixels.BoundaryPadder.pad(I_dbl, [pad_rows, pad_cols], obj.Boundary);
                
                % Center of the kernel
                center_r = floor((Kh + 1) / 2);
                center_c = floor((Kw + 1) / 2);
                
                I_filt = zeros(H, W, C);
                for r = 1:Kh
                    row_start = pad_rows + 1 + (r - center_r);
                    for c = 1:Kw
                        col_start = pad_cols + 1 + (c - center_c);
                        I_filt = I_filt + K(r, c) * I_pad(row_start : row_start + H - 1, col_start : col_start + W - 1, :);
                    end
                end
                
                % Then downsample
                I_out = I_filt(1:L:end, 1:L:end, :);
            else
                % Downsample blindly (no filtering)
                I_out = I_dbl(1:L:end, 1:L:end, :);
            end
            
            % Restore original class type
            if isa(I_in, 'uint8')
                I_out = uint8(round(I_out));
            elseif isa(I_in, 'single')
                I_out = single(I_out);
            end
        end
    end
end
