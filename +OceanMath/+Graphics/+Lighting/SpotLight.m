classdef SpotLight < OceanMath.Graphics.Lighting.ILight
    % SPOTLIGHT  Cone-shaped positional light source aiming in a specific direction.
    
    properties
        Position (3,1) double = [0; 0; 0]
        Direction (3,1) double = [0; 0; -1]       % Unit direction light is aiming
        Intensity (3,1) double = [1.0; 1.0; 1.0]
        ConeAngle (1,1) double = 30.0             % Cutoff angle in degrees
        Exponent (1,1) double = 2.0               % Angular concentration falloff exponent
        ConstantAttenuation (1,1) double = 1.0
        LinearAttenuation (1,1) double = 0.0
        QuadraticAttenuation (1,1) double = 0.0
    end
    
    methods
        function obj = SpotLight(options)
            arguments
                options.Position (3,1) double = [0; 0; 0]
                options.Direction (3,1) double = [0; 0; -1]
                options.Intensity (3,1) double = [1.0; 1.0; 1.0]
                options.ConeAngle (1,1) double = 30.0
                options.Exponent (1,1) double = 2.0
                options.ConstantAttenuation (1,1) double = 1.0
                options.LinearAttenuation (1,1) double = 0.0
                options.QuadraticAttenuation (1,1) double = 0.0
            end
            obj.Position = options.Position;
            
            dVal = options.Direction;
            dNorm = norm(dVal);
            if dNorm < 1e-12
                dVal = [0; 0; -1];
                dNorm = 1.0;
            end
            obj.Direction = dVal / dNorm;
            obj.Intensity = options.Intensity;
            obj.ConeAngle = options.ConeAngle;
            obj.Exponent = options.Exponent;
            obj.ConstantAttenuation = options.ConstantAttenuation;
            obj.LinearAttenuation = options.LinearAttenuation;
            obj.QuadraticAttenuation = options.QuadraticAttenuation;
        end
        
        function [intensity, dirToLight, distance] = getLightAt(obj, point)
            vec = obj.Position(:) - point(:);
            distance = norm(vec);
            
            if distance < 1e-12
                dirToLight = -obj.Direction;
            else
                dirToLight = vec / distance;
            end
            
            % Direction from light to surface point is -dirToLight
            cosAngle = dot(-dirToLight, obj.Direction);
            cosCutoff = cos(deg2rad(obj.ConeAngle));
            
            if cosAngle < cosCutoff
                % Outside the spotlight cone
                intensity = [0; 0; 0];
                return;
            end
            
            % Angular decay concentration factor
            angularFalloff = cosAngle^obj.Exponent;
            
            % Distance attenuation equation
            denom = obj.ConstantAttenuation + ...
                    obj.LinearAttenuation * distance + ...
                    obj.QuadraticAttenuation * distance^2;
            
            if denom < 1e-12
                denom = 1e-12;
            end
            
            intensity = (obj.Intensity * angularFalloff) / denom;
        end
    end
end
