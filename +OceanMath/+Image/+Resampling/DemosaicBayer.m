classdef DemosaicBayer < handle
    % DEMOSAICBAYER  Bayer Color Filter Array (CFA) demosaicing engine.
    %
    %   Performs reconstruction of full color RGB images from 2D Bayer patterns
    %   using:
    %     1. Bilinear Interpolation: independent channel bilinear averaging.
    %     2. Malvar-He-Cutler (MHC): high-quality gradient-corrected linear interpolation
    %        using 5x5 spatial derivatives (Malvar, He, Cutler 2004).
    %
    %   Supports the standard 'rggb' Bayer layout starting with Red at (1,1).
    %
    %   Author: Ocean Library System
    %   Date: 2026-07-01
    
    properties (Access = private)
        Pattern char = 'rggb'
    end
    
    methods
        function obj = DemosaicBayer(pattern)
            % DEMOSAICBAYER Construct a demosaicing engine.
            arguments
                pattern char = 'rggb'
            end
            obj.Pattern = lower(pattern);
            if ~strcmp(obj.Pattern, 'rggb')
                error('Currently only standard ''rggb'' pattern is supported.');
            end
        end
        
        function I_rgb = bilinear(obj, CFA)
            % BILINEAR Performs standard bilinear color channel interpolation.
            arguments
                obj
                CFA (:,:) double
            end
            
            [H, W] = size(CFA);
            
            % Generate coordinates (1-indexed)
            [C, R] = meshgrid(1:W, 1:H);
            
            % RGGB Masks:
            % Red (odd r, odd c)
            maskR = (mod(R, 2) == 1) & (mod(C, 2) == 1);
            % Green1 (odd r, even c) and Green2 (even r, odd c)
            maskG = (mod(R + C, 2) == 1);
            % Blue (even r, even c)
            maskB = (mod(R, 2) == 0) & (mod(C, 2) == 0);
            
            % Separate raw color channels
            CFA_R = CFA .* maskR;
            CFA_G = CFA .* maskG;
            CFA_B = CFA .* maskB;
            
            % Interpolation filters
            H_RB = [0.25, 0.5, 0.25; 0.5, 1.0, 0.5; 0.25, 0.5, 0.25];
            H_G  = [0.0, 0.25, 0.0; 0.25, 1.0, 0.25; 0.0, 0.25, 0.0];
            
            % Apply convolutions
            R_interp = imfilter(CFA_R, H_RB, 'replicate');
            G_interp = imfilter(CFA_G, H_G, 'replicate');
            B_interp = imfilter(CFA_B, H_RB, 'replicate');
            
            I_rgb = cat(3, R_interp, G_interp, B_interp);
        end
        
        function I_rgb = malvarHeCutler(obj, CFA)
            % MALVARHECUTLER Performs Malvar-He-Cutler gradient-corrected demosaicing.
            arguments
                obj
                CFA (:,:) double
            end
            
            [H, W] = size(CFA);
            [C, R] = meshgrid(1:W, 1:H);
            
            % RGGB Masks:
            % Red: odd r, odd c
            maskR = (mod(R, 2) == 1) & (mod(C, 2) == 1);
            % Green in Red rows: odd r, even c
            maskG1 = (mod(R, 2) == 1) & (mod(C, 2) == 0);
            % Green in Blue rows: even r, odd c
            maskG2 = (mod(R, 2) == 0) & (mod(C, 2) == 1);
            % Blue: even r, even c
            maskB = (mod(R, 2) == 0) & (mod(C, 2) == 0);
            
            % Separate raw color channels
            CFA_R = CFA .* maskR;
            CFA_G = CFA .* (maskG1 | maskG2);
            CFA_B = CFA .* maskB;
            
            % 1. GREEN Channel Interpolation:
            % Green at Red or Blue pixels is corrected using the surrounding channel derivatives.
            H_G_at_R = [  0,   0, -1,   0,   0;
                          0,   0,  2,   0,   0;
                         -1,   2,  4,   2,  -1;
                          0,   0,  2,   0,   0;
                          0,   0, -1,   0,   0 ] / 8.0;
            
            G_interp = CFA_G;
            G_filtered = imfilter(CFA, H_G_at_R, 'replicate');
            G_interp(maskR | maskB) = G_filtered(maskR | maskB);
            
            % 2. RED Channel Interpolation:
            % Red at Green pixels on Red rows (horizontal Red neighbors)
            H_R_at_G_horiz = [  0,  0, 0.5,  0,  0;
                                0, -1,   0, -1,  0;
                               -1,  4,   5,  4, -1;
                                0, -1,   0, -1,  0;
                                0,  0, 0.5,  0,  0 ] / 8.0;
            
            % Red at Green pixels on Blue rows (vertical Red neighbors)
            H_R_at_G_vert = H_R_at_G_horiz';
            
            % Red at Blue pixels (diagonal Red neighbors)
            H_R_at_B = [   0,   0, -1.5,   0,   0;
                           0,   2,    0,   2,   0;
                        -1.5,   0,    6,   0, -1.5;
                           0,   2,    0,   2,   0;
                           0,   0, -1.5,   0,   0 ] / 8.0;
            
            R_interp = CFA_R;
            R_filtered_horiz = imfilter(CFA, H_R_at_G_horiz, 'replicate');
            R_filtered_vert  = imfilter(CFA, H_R_at_G_vert, 'replicate');
            R_filtered_diag  = imfilter(CFA, H_R_at_B, 'replicate');
            
            R_interp(maskG1) = R_filtered_horiz(maskG1);
            R_interp(maskG2) = R_filtered_vert(maskG2);
            R_interp(maskB)  = R_filtered_diag(maskB);
            
            % 3. BLUE Channel Interpolation:
            % Blue uses the exact same symmetric kernels.
            % For Blue at Green pixels: on Blue rows (maskG2), Blue has horizontal neighbors.
            % On Red rows (maskG1), Blue has vertical neighbors.
            B_interp = CFA_B;
            B_filtered_horiz = imfilter(CFA, H_R_at_G_horiz, 'replicate');
            B_filtered_vert  = imfilter(CFA, H_R_at_G_vert, 'replicate');
            B_filtered_diag  = imfilter(CFA, H_R_at_B, 'replicate');
            
            B_interp(maskG2) = B_filtered_horiz(maskG2);
            B_interp(maskG1) = B_filtered_vert(maskG1);
            B_interp(maskR)  = B_filtered_diag(maskR);
            
            I_rgb = cat(3, R_interp, G_interp, B_interp);
        end
    end
end
