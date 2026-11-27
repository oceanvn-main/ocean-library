classdef BlueScreenMatting < handle
    % BLUESCREENMATTING  Extracts foreground objects from blue or green screens.
    %
    %   Implements the Smith-Blinn (1996) and Vlahos matting formulation.
    %   Uses a known background image to perform precise difference-based
    %   matting, threshold tuning, and color despilling.
    %
    %   Author: Ocean Library System
    %   Date: 2026-07-01
    
    properties (Access = private)
        Background double % Background image [H x W x 3] in range [0, 1]
        KeyChannel double % 2 for Green, 3 for Blue
    end
    
    methods
        function obj = BlueScreenMatting(backgroundImage)
            % BLUESCREENMATTING Construct a matting engine.
            arguments
                backgroundImage (:,:,:) double
            end
            
            % Normalize to double [0, 1] if input is uint8
            if isa(backgroundImage, 'uint8')
                backgroundImage = double(backgroundImage) / 255.0;
            end
            obj.Background = backgroundImage;
            
            % Automatically determine key channel based on background color
            meanR = mean(backgroundImage(:,:,1), 'all');
            meanG = mean(backgroundImage(:,:,2), 'all');
            meanB = mean(backgroundImage(:,:,3), 'all');
            
            if meanG > meanR && meanG > meanB
                obj.KeyChannel = 2; % Green Screen
                fprintf('[Matting] Detected GREEN background key channel.\n');
            else
                obj.KeyChannel = 3; % Blue Screen
                fprintf('[Matting] Detected BLUE background key channel.\n');
            end
        end
        
        function [alpha, foreground] = pullMatte(obj, compositeImage, varargin)
            % PULLMATTE Extracts the alpha matte and the despilled foreground.
            %
            %   [alpha, foreground] = pullMatte(C, 'AlphaMin', 0.05, 'AlphaMax', 0.95)
            
            arguments
                obj
                compositeImage (:,:,:) double
            end
            arguments (Repeating)
                varargin
            end
            
            % Parse optional name-value arguments manually to keep syntax simple
            alphaMin = 0.0;
            alphaMax = 1.0;
            despillStrength = 1.0;
            mode = 'colorkey';
            
            for i = 1:2:length(varargin)
                name = varargin{i};
                val = varargin{i+1};
                switch lower(name)
                    case 'alphamin'
                        alphaMin = val;
                    case 'alphamax'
                        alphaMax = val;
                    case 'despillstrength'
                        despillStrength = val;
                    case 'mode'
                        mode = lower(val);
                end
            end
            
            % Ensure composite is double [0, 1]
            if isa(compositeImage, 'uint8')
                compositeImage = double(compositeImage) / 255.0;
            end
            
            C = compositeImage;
            B = obj.Background;
            
            [H, W, C_ch] = size(C);
            [Hb, Wb, B_ch] = size(B);
            if H ~= Hb || W ~= Wb
                error('Composite image size must match the background image size.');
            end
            
            % Compute alpha matte
            if strcmp(mode, 'euclidean')
                % 1a. Compute alpha based on Euclidean distance in RGB color space
                % Default distance thresholds for background subtraction if not specified
                if alphaMin == 0.0 && alphaMax == 1.0
                    alphaMin = 0.15;
                    alphaMax = 0.25;
                end
                
                dist = sqrt(sum((C - B).^2, 3));
                alpha = (dist - alphaMin) / (alphaMax - alphaMin);
                alpha = max(0.0, min(1.0, alpha));
            else
                % 1b. Compute alpha based on key channel separation
                if obj.KeyChannel == 2
                    % Green Screen Keying
                    diff_c = C(:,:,2) - max(C(:,:,1), C(:,:,3));
                    diff_b = B(:,:,2) - max(B(:,:,1), B(:,:,3));
                else
                    % Blue Screen Keying
                    diff_c = C(:,:,3) - max(C(:,:,1), C(:,:,2));
                    diff_b = B(:,:,3) - max(B(:,:,1), B(:,:,2));
                end
                
                % Avoid dividing by zero or negative values
                diff_b = max(diff_b, 0.05);
                ratio = diff_c ./ diff_b;
                ratio = max(0.0, min(1.0, ratio));
                alpha = 1.0 - ratio;
                
                if alphaMax > alphaMin
                    alpha = (alpha - alphaMin) / (alphaMax - alphaMin);
                    alpha = max(0.0, min(1.0, alpha));
                end
            end
            
            % 2. Extract foreground via background subtraction
            % C = alpha * F + (1 - alpha) * B => alpha * F = C - (1 - alpha) * B
            foreground = C - (1.0 - alpha) .* B;
            foreground = max(0.0, min(1.0, foreground));
            
            % 3. Apply Despilling on the extracted foreground
            if despillStrength > 0
                R_fg = foreground(:,:,1);
                G_fg = foreground(:,:,2);
                B_fg = foreground(:,:,3);
                
                if obj.KeyChannel == 2
                    % Limit green channel to the maximum of red and blue
                    limit = max(R_fg, B_fg);
                    excess = max(0.0, G_fg - limit);
                    G_fg = G_fg - despillStrength * excess;
                    foreground(:,:,2) = max(0.0, G_fg);
                else
                    % Limit blue channel to the maximum of red and green
                    limit = max(R_fg, G_fg);
                    excess = max(0.0, B_fg - limit);
                    B_fg = B_fg - despillStrength * excess;
                    foreground(:,:,3) = max(0.0, B_fg);
                end
            end
        end
        
        function composite = blendOn(obj, foreground, alpha, newBackground)
            % BLENDON Composites the extracted foreground onto a new background.
            arguments
                obj
                foreground (:,:,:) double
                alpha (:,:) double
                newBackground (:,:,:) double
            end
            
            if isa(newBackground, 'uint8')
                newBackground = double(newBackground) / 255.0;
            end
            
            [H, W, ~] = size(foreground);
            
            % Resize new background to match foreground if sizes differ
            if size(newBackground, 1) ~= H || size(newBackground, 2) ~= W
                newBackground = imresize(newBackground, [H, W]);
            end
            
            % Compositing equation: C_new = alpha * F + (1 - alpha) * B_new
            % Note: foreground is already unspilled/pre-multiplied by alpha!
            % If we did: composite = foreground + (1.0 - alpha) .* newBackground;
            % Wait, let's verify if the foreground returned by pullMatte is pre-multiplied.
            % Yes! foreground = C - (1 - alpha) * B. So it is pre-multiplied: alpha * F.
            composite = foreground + (1.0 - alpha) .* newBackground;
            composite = max(0.0, min(1.0, composite));
        end
    end
end
