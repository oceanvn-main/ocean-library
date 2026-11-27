classdef GammaCorrection < OceanMath.Image.Pixels.IPixelTransform
    % GAMMACORRECTION  Applies power-law gamma correction to image pixels.
    %
    %   I_out = C * (I_in)^gamma
    %
    %   Normalizes image values to [0, 1] internally to apply the power-law
    %   mapping, then scales back to the original dynamic range.
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-11
    
    properties (Access = private)
        Gamma (1,1) double = 1.0
        C     (1,1) double = 1.0
    end
    
    methods
        function obj = GammaCorrection(options)
            % GAMMACORRECTION Construct a gamma correction transform.
            arguments
                options.Gamma (1,1) double = 1.0
                options.C     (1,1) double = 1.0
            end
            obj.Gamma = options.Gamma;
            obj.C = options.C;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Normalizes and applies the power-law transformation.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            is_uint8 = isa(I_in, 'uint8');
            I_double = double(I_in);
            
            % Determine normalization scale
            if is_uint8
                max_range = 255.0;
            else
                % Auto-detect range: [0, 1] vs [0, 255]
                if max(I_double(:)) <= 1.0
                    max_range = 1.0;
                else
                    max_range = 255.0;
                end
            end
            
            % Normalize to [0, 1]
            I_norm = I_double / max_range;
            
            % Apply gamma mapping: T(r) = c * r^gamma
            I_res = obj.C * (I_norm .^ obj.Gamma);
            
            % Clip to [0, 1] to prevent overflow/saturation issues
            I_res = max(0.0, min(1.0, I_res));
            
            % Scale back to original range
            I_res = I_res * max_range;
            
            if is_uint8
                I_out = uint8(I_res);
            else
                I_out = I_res;
            end
        end
    end
end
