classdef SynthesisFilter < OceanMath.Image.Pixels.IPixelTransform
    % SYNTHESISFILTER  Applies wavelet/subband synthesis filtering to reconstruct signals.
    %
    %   Performs low-pass or high-pass synthesis filtering on zero-stuffed
    %   images to reconstruct approximation (LL) or detail (LH, HL, HH) components.
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-23
    
    properties (Constant)
        Description = "Synthesis Filter reconstructs details or approximations from zero-stuffed subbands. By convolving the zero-stuffed grid with low-pass (LL) or high-pass (LH, HL, HH) filters (Haar or LeGall 5/3), it smooths out imaging artifacts and separates scale components.";
    end

    properties (Access = private)
        Preset (1,1) string = "Haar"
        Mode   (1,1) string = "Diagonal High-Pass (HH)"
        Boundary = 'replicate'
    end
    
    methods
        function obj = SynthesisFilter(options)
            arguments
                options.Preset (1,1) string = "Haar"
                options.Mode   (1,1) string = "Diagonal High-Pass (HH)"
                options.Boundary = 'replicate'
            end
            obj.Preset = options.Preset;
            obj.Mode = options.Mode;
            obj.Boundary = options.Boundary;
        end
        
        function I_out = apply(obj, I_in)
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            % 1. Get 1D low-pass (g) and high-pass (h) synthesis filters
            switch obj.Preset
                case "Haar"
                    g = [0.5, 0.5] * 2.0; % Scale by 2 to offset upsampling loss
                    h = [-0.5, 0.5] * 2.0;
                case "LeGall 5/3"
                    g = [-0.25, 0.5, 1.5, 0.5, -0.25];
                    h = [-0.5, 1.0, -0.5];
                otherwise % Default to Haar
                    g = [1.0, 1.0];
                    h = [-1.0, 1.0];
            end
            
            % 2. Form 2D kernel based on Mode
            switch obj.Mode
                case "Low-Pass (LL)"
                    K = g' * g;
                case "Horizontal High-Pass (LH)"
                    K = g' * h;
                case "Vertical High-Pass (HL)"
                    K = h' * g;
                case "Diagonal High-Pass (HH)"
                    K = h' * h;
                otherwise
                    K = h' * h;
            end
            
            % 3. Apply 2D spatial filtering (correlation) from scratch
            I_dbl = double(I_in);
            [H, W, C] = size(I_dbl);
            [Kh, Kw] = size(K);
            
            pad_rows = ceil((Kh - 1) / 2);
            pad_cols = ceil((Kw - 1) / 2);
            
            % Perform boundary padding using our custom BoundaryPadder
            I_pad = OceanMath.Image.Pixels.BoundaryPadder.pad(I_dbl, [pad_rows, pad_cols], obj.Boundary);
            
            % Center of the kernel
            center_r = floor((Kh + 1) / 2);
            center_c = floor((Kw + 1) / 2);
            
            % 2D correlation summation with explicit centering offset
            I_out = zeros(H, W, C);
            for r = 1:Kh
                row_start = pad_rows + 1 + (r - center_r);
                for c = 1:Kw
                    col_start = pad_cols + 1 + (c - center_c);
                    I_out = I_out + K(r, c) * I_pad(row_start : row_start + H - 1, col_start : col_start + W - 1, :);
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
