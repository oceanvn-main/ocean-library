classdef MinMaxStretch < OceanMath.Image.Pixels.IPixelTransform
    % MINMAXSTRETCH  Stretches input dynamic range to fill a target range.
    %
    %   Maps the darkest pixel to target_min and the brightest pixel to
    %   target_max, linearly interpolating intermediate values.
    %
    %   Constructor:
    %       MinMaxStretch('TargetRange', bounds)
    %
    %   Inputs:
    %       TargetRange - [1 x 2] double bounds.
    %                     Default: [-1, -1] (auto-detects based on input class:
    %                     [0, 255] for uint8, [0, 1] for double).
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-11
    
    properties (Access = private)
        TargetRange (1,2) double = [-1, -1]
    end
    
    methods
        function obj = MinMaxStretch(options)
            % MINMAXSTRETCH Construct a MinMaxStretch operator.
            arguments
                options.TargetRange (1,2) double = [-1, -1]
            end
            obj.TargetRange = options.TargetRange;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Dynamically compute min/max and stretch intensities.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            is_uint8 = isa(I_in, 'uint8');
            I_double = double(I_in);
            
            min_val = min(I_double(:));
            max_val = max(I_double(:));
            
            % If the image is completely uniform, no stretching is possible
            if (max_val - min_val) < 1e-12
                I_out = I_in;
                return;
            end
            
            % Determine target range bounds
            if all(obj.TargetRange == -1)
                if is_uint8
                    t_min = 0.0;
                    t_max = 255.0;
                else
                    t_min = 0.0;
                    t_max = 1.0;
                end
            else
                t_min = obj.TargetRange(1);
                t_max = obj.TargetRange(2);
            end
            
            % Apply contrast stretching formula
            I_res = (I_double - min_val) * (t_max - t_min) / (max_val - min_val) + t_min;
            
            % Squeeze values to stay in target range
            I_res = max(t_min, min(t_max, I_res));
            
            if is_uint8
                I_out = uint8(I_res);
            else
                I_out = I_res;
            end
        end
    end
end
