classdef AmbientLight < OceanMath.Graphics.Lighting.ILight
    % AMBIENTLIGHT  Uniform ambient light source.
    
    properties
        Intensity (3,1) double = [0.2; 0.2; 0.2]  % RGB color intensity
    end
    
    methods
        function obj = AmbientLight(options)
            arguments
                options.Intensity (3,1) double = [0.2; 0.2; 0.2]
            end
            obj.Intensity = options.Intensity;
        end
        
        function [intensity, dirToLight, distance] = getLightAt(obj, ~)
            intensity = obj.Intensity;
            dirToLight = [0; 0; 0];
            distance = Inf;
        end
    end
end
