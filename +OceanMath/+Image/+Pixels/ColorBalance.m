classdef ColorBalance < OceanMath.Image.Pixels.IPixelTransform
    % COLORBALANCE  Applies channel-wise multiplier gains for color balancing.
    %
    %   Performs channel-wise scaling on Red, Green, and Blue channels.
    %   Supports linearizing the image using gamma correction before applying
    %   gains, and then encoding it back to the original gamma space.
    %
    %   Author: Ocean Library System
    %   Date: 2026-07-01
    
    properties (Access = public)
        RedGain         (1,1) double = 1.0
        GreenGain       (1,1) double = 1.0
        BlueGain        (1,1) double = 1.0
        Gamma           (1,1) double = 1.0
        ApplyGammaFirst (1,1) logical = false
    end
    
    methods
        function obj = ColorBalance(options)
            % COLORBALANCE Construct a color balance transform.
            arguments
                options.RedGain         (1,1) double = 1.0
                options.GreenGain       (1,1) double = 1.0
                options.BlueGain        (1,1) double = 1.0
                options.Gamma           (1,1) double = 1.0
                options.ApplyGammaFirst (1,1) logical = false
            end
            obj.RedGain = options.RedGain;
            obj.GreenGain = options.GreenGain;
            obj.BlueGain = options.BlueGain;
            obj.Gamma = options.Gamma;
            obj.ApplyGammaFirst = options.ApplyGammaFirst;
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Multiplies color channels by specified gain constants.
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            [H, W, C] = size(I_in);
            if C < 3
                % Not a color image, return unchanged or apply average scale
                I_out = I_in;
                return;
            end
            
            is_uint8 = isa(I_in, 'uint8');
            I_double = double(I_in);
            
            % Determine range
            if is_uint8
                max_range = 255.0;
            else
                if max(I_double(:)) <= 1.0
                    max_range = 1.0;
                else
                    max_range = 255.0;
                end
            end
            
            % Normalize to [0, 1]
            I_norm = I_double / max_range;
            
            if obj.ApplyGammaFirst
                % 1. Gamma decode (to linear space)
                I_linear = I_norm .^ obj.Gamma;
                
                % 2. Apply channel-wise gains in linear space
                I_linear(:,:,1) = I_linear(:,:,1) * obj.RedGain;
                I_linear(:,:,2) = I_linear(:,:,2) * obj.GreenGain;
                I_linear(:,:,3) = I_linear(:,:,3) * obj.BlueGain;
                
                % 3. Gamma encode (back to sRGB/encoded space)
                I_res = I_linear .^ (1.0 / obj.Gamma);
            else
                % Apply channel-wise gains directly in encoded space
                I_res = I_norm;
                I_res(:,:,1) = I_res(:,:,1) * obj.RedGain;
                I_res(:,:,2) = I_res(:,:,2) * obj.GreenGain;
                I_res(:,:,3) = I_res(:,:,3) * obj.BlueGain;
            end
            
            % Clip to [0, 1] to prevent overflow/underflow
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
