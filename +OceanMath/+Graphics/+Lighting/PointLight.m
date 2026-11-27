classdef PointLight < OceanMath.Graphics.Lighting.ILight
    % POINTLIGHT  Positional light source radiating in all directions.
    
    properties
        Position (3,1) double = [0; 0; 0]
        Intensity (3,1) double = [1.0; 1.0; 1.0]
        ConstantAttenuation (1,1) double = 1.0
        LinearAttenuation (1,1) double = 0.0
        QuadraticAttenuation (1,1) double = 0.0
    end
    
    methods
        function obj = PointLight(options)
            arguments
                options.Position (3,1) double = [0; 0; 0]
                options.Intensity (3,1) double = [1.0; 1.0; 1.0]
                options.ConstantAttenuation (1,1) double = 1.0
                options.LinearAttenuation (1,1) double = 0.0
                options.QuadraticAttenuation (1,1) double = 0.0
            end
            obj.Position = options.Position;
            obj.Intensity = options.Intensity;
            obj.ConstantAttenuation = options.ConstantAttenuation;
            obj.LinearAttenuation = options.LinearAttenuation;
            obj.QuadraticAttenuation = options.QuadraticAttenuation;
        end
        
        function [intensity, dirToLight, distance] = getLightAt(obj, point)
            % GETLIGHTAT Calculate light direction and distance-attenuated intensity.
            vec = obj.Position(:) - point(:);
            distance = norm(vec);
            
            if distance < 1e-12
                dirToLight = [0; 0; 1];
            else
                dirToLight = vec / distance;
            end
            
            % Distance attenuation equation
            denom = obj.ConstantAttenuation + ...
                    obj.LinearAttenuation * distance + ...
                    obj.QuadraticAttenuation * distance^2;
            
            if denom < 1e-12
                denom = 1e-12;
            end
            
            intensity = obj.Intensity / denom;
        end
    end
end
