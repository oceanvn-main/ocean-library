classdef BackgroundSubtractor < handle
    % BACKGROUNDSUBTRACTOR  Statistical background subtraction model.
    %
    %   Implements difference keying (Toyama et al. 1999) by modeling the
    %   background statistical profile (mean and variance) at each pixel.
    %   Supports foreground segmentation via Mahalanobis/Chi-Square distance
    %   and morphological object tracking.
    %
    %   Author: Ocean Library System
    %   Date: 2026-07-08
    
    properties (SetAccess = private)
        CentralTendency double % Pixel-wise location parameter image [H x W x C]
        Variance double        % Pixel-wise variance image [H x W x C]
    end
    
    methods
        function obj = BackgroundSubtractor(trainFrames, varargin)
            % BACKGROUNDSUBTRACTOR Construct a background subtraction model.
            %
            %   obj = BackgroundSubtractor(trainFrames, 'Method', 'MedianMAD')
            %
            %   Input:
            %     trainFrames: [H x W x C x N] double or uint8 array of N training frames.
            %     varargin: Optional name-value arguments (e.g. 'Method', 'MedianMAD')
            
            arguments
                trainFrames (:,:,:,:) double
            end
            arguments (Repeating)
                varargin
            end
            
            % Normalize to double [0, 1] if input is uint8
            if isa(trainFrames, 'uint8')
                trainFrames = double(trainFrames) / 255.0;
            end
            
            % Parse name-value parameters manually
            method = 'meanstd';
            for i = 1:2:length(varargin)
                name = varargin{i};
                val = varargin{i+1};
                if strcmpi(name, 'method')
                    method = lower(val);
                end
            end
            
            if strcmp(method, 'medianmad')
                % 1. Compute robust temporal median
                obj.CentralTendency = median(trainFrames, 4);
                
                % 2. Compute Median Absolute Deviation (MAD): median(|X - median(X)|)
                diff = abs(trainFrames - obj.CentralTendency);
                madVal = median(diff, 4);
                
                % 3. Compute robust variance estimate: (1.4826 * MAD)^2
                obj.Variance = 2.1981 * (madVal.^2);
            else
                % Default: Compute standard arithmetic mean and variance
                obj.CentralTendency = mean(trainFrames, 4);
                obj.Variance = var(trainFrames, 0, 4);
            end
            
            % Add a regularization factor to prevent division by zero in static pixels.
            % We only enforce a lower bound to prevent division by zero; capping the maximum
            % variance is not recommended as it prevents dynamic background elements (like trees)
            % from being correctly modeled.
            obj.Variance = max(obj.Variance, 1e-4);
        end
        
        function [fgMask, foreground] = apply(obj, frame, threshold)
            % APPLY Segment foreground from a given video frame.
            %
            %   [fgMask, foreground] = apply(obj, frame, threshold)
            %
            %   Inputs:
            %     frame: [H x W x C] double or uint8 video frame.
            %     threshold: Chi-Square distance threshold (default = 11.34 for 99% confidence, 3 DoF)
            %
            %   Outputs:
            %     fgMask: Binary foreground mask [H x W].
            %     foreground: RGBA image [H x W x 4] of the frame with background pixels set to 0.
            
            arguments
                obj
                frame (:,:,:) double
                threshold (1,1) double = 11.34
            end
            
            if isa(frame, 'uint8')
                frame = double(frame) / 255.0;
            end
            
            [H, W, C] = size(frame);
            [Hm, Wm, Cm] = size(obj.CentralTendency);
            if H ~= Hm || W ~= Wm || C ~= Cm
                error('Input frame dimensions must match the trained background model.');
            end
            
            % Compute pixel-wise squared Mahalanobis distance across color channels
            diff = frame - obj.CentralTendency;
            dist = sum((diff.^2) ./ obj.Variance, 3);
            
            % Segment foreground pixels exceeding the threshold
            fgMask = dist > threshold;
            
            % Extract foreground object and set background to zero (RGBA = 0)
            foreground = cat(3, frame, double(fgMask));
            foreground(repmat(~fgMask, [1, 1, 4])) = 0.0;
        end
        
        function [cleanMask, centroids] = postProcess(obj, fgMask, varargin)
            % POSTPROCESS Clean up the mask using morphology and extract object centroids.
            %
            %   [cleanMask, centroids] = postProcess(obj, fgMask, 'MinArea', 100, 'MorphSize', 3)
            
            arguments
                obj
                fgMask (:,:) logical
            end
            arguments (Repeating)
                varargin
            end
            
            % Parse name-value pairs
            minArea = 100;
            morphSize = 3;
            
            for i = 1:2:length(varargin)
                name = varargin{i};
                val = varargin{i+1};
                switch lower(name)
                    case 'minarea'
                        minArea = val;
                    case 'morphsize'
                        morphSize = val;
                end
            end
            
            % Apply morphological opening to remove isolated noise pixels
            se = strel('disk', morphSize);
            cleanMask = imopen(fgMask, se);
            
            % Apply morphological closing to fill small holes inside the object
            cleanMask = imclose(cleanMask, se);
            
            % Find connected components in the binary image
            cc = bwconncomp(cleanMask);
            
            % Extract centroids and area measurements
            stats = regionprops(cc, 'Centroid', 'Area');
            
            % Filter out components smaller than the minimum area threshold
            validIdx = [stats.Area] >= minArea;
            validStats = stats(validIdx);
            
            if isempty(validStats)
                centroids = double.empty(0, 2);
            else
                centroids = cat(1, validStats.Centroid);
            end
        end
        
        function composite = blendOn(obj, foregroundRGBA, newBackground)
            % BLENDON Composites the extracted RGBA foreground over a new background.
            %
            %   composite = blendOn(obj, foregroundRGBA, newBackground)
            
            arguments
                obj
                foregroundRGBA (:,:,4) double
                newBackground (:,:,3) double
            end
            
            if isa(newBackground, 'uint8')
                newBackground = double(newBackground) / 255.0;
            end
            
            [H, W, ~] = size(foregroundRGBA);
            if size(newBackground, 1) ~= H || size(newBackground, 2) ~= W
                newBackground = imresize(newBackground, [H, W]);
            end
            
            alpha = foregroundRGBA(:,:,4);
            composite = foregroundRGBA(:,:,1:3) + (1.0 - alpha) .* newBackground;
            composite = max(0.0, min(1.0, composite));
        end
    end
end
