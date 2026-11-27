classdef LinearOperator < OceanMath.Image.Pixels.IPixelTransform
    % LINEAROPERATOR  Applies linear scaling and bias to image pixels.
    %
    %   I_out = scale * I_in + bias
    %
    %   Used for brightness adjustment (changing bias) and contrast adjustment
    %   (changing scale). Automatically handles clipping and maintains input
    %   class type (uint8/double).
    %
    %   Author: Ocean Library System
    %   Date: 2026-06-11
    
    properties (Access = private)
        Scale (1,1) double = 1.0
        Bias  (1,1) double = 0.0
        Clip  (1,1) logical = true
    end
    
    methods
        function obj = LinearOperator(options)
            % LINEAROPERATOR Construct a linear operator.
            arguments
                options.Scale (1,1) double = 1.0
                options.Bias  (1,1) double = 0.0
                options.Clip  (1,1) logical = true
            end
            obj.Scale = options.Scale;
            obj.Bias = options.Bias;
            obj.Clip = options.Clip;
            obj.Description = 'Input: Spatial image array (Grayscale/RGB). Output: Spatial image array. Safely scales contrast and adds brightness bias without breaking the pipeline.';
        end
        
        function I_out = apply(obj, I_in)
            % APPLY Performs the linear mapping: I_out = Scale * I_in + Bias
            arguments
                obj
                I_in {mustBeNumeric}
            end
            
            % Save original class type
            is_uint8 = isa(I_in, 'uint8');
            
            % Compute double representation for accuracy
            I_double = double(I_in);
            I_res = obj.Scale * I_double + obj.Bias;
            
            % Apply clipping if requested
            if obj.Clip
                if is_uint8
                    I_res = max(0.0, min(255.0, I_res));
                else
                    % Auto-detect range: [0, 1] vs [0, 255]
                    max_val = max(I_double(:));
                    if max_val <= 1.0
                        I_res = max(0.0, min(1.0, I_res));
                    else
                        I_res = max(0.0, min(255.0, I_res));
                    end
                end
            end
            
            % Restore original class type
            if is_uint8
                I_out = uint8(I_res);
            else
                I_out = I_res;
            end
        end
    end
end
