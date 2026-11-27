classdef MeanMatchStretch < OceanMath.Image.Pixels.IPixelTransform
    % MEANMATCHSTRETCH  Shifts image mean to middle gray and scales contrast.
    %
    %   Centers the pixel intensities around the dynamic mean, applies a
    %   contrast scale factor, and translates the mean to the target gray value.
    %
    %   Constructor:
    %       MeanMatchStretch('TargetMean', target, 'Scale', scale)
    %
    %   Inputs:
    %       TargetMean - Target average brightness.
    %                    Default: -1 (auto-detects: 127.5 for uint8, 0.5 for double).
    %       Scale      - Contrast expansion factor (default: 1.0).
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-11
    
    properties (Access = private)
        TargetMean (1,1) double = -1.0
        Scale      (1,1) double = 1.0
    end
    
    methods
        function obj = MeanMatchStretch(options)
            % MEANMATCHSTRETCH Construct a MeanMatchStretch operator.
            arguments
                options.TargetMean (1,1) double = -1.0
                options.Scale      (1,1) double = 1.0
            end
            obj.TargetMean = options.TargetMean;
            obj.Scale = options.Scale;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Center around dynamic mean, scale, and shift to target.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            is_uint8 = isa(I_in, 'uint8');
            I_double = double(I_in);
            
            % Compute mean dynamically
            mean_val = mean(I_double(:));
            
            % Determine target middle gray and limits for clipping
            if obj.TargetMean == -1.0
                if is_uint8
                    target_gray = 127.5;
                    t_min = 0.0;
                    t_max = 255.0;
                else
                    target_gray = 0.5;
                    t_min = 0.0;
                    t_max = 1.0;
                end
            else
                target_gray = obj.TargetMean;
                if is_uint8
                    t_min = 0.0;
                    t_max = 255.0;
                else
                    % For double, assume target_gray's bounds
                    if target_gray <= 1.0
                        t_min = 0.0;
                        t_max = 1.0;
                    else
                        t_min = 0.0;
                        t_max = 255.0;
                    end
                end
            end
            
            % Center, scale, and shift
            I_centered = I_double - mean_val;
            I_res = obj.Scale * I_centered + target_gray;
            
            % Clip to valid range
            I_res = max(t_min, min(t_max, I_res));
            
            if is_uint8
                I_out = uint8(I_res);
            else
                I_out = I_res;
            end
        end
    end
end
