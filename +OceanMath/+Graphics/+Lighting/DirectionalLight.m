classdef DirectionalLight < OceanMath.Graphics.Lighting.ILight
    % DIRECTIONALLIGHT  Distant light source emitting parallel rays.
    
    properties
        Direction (3,1) double = [0; 0; -1]  % Unit vector showing direction light is travelling
        Intensity (3,1) double = [1.0; 1.0; 1.0]
    end
    
    methods
        function obj = DirectionalLight(options)
            arguments
                options.Direction (3,1) double = [0; 0; -1]
                options.Intensity (3,1) double = [1.0; 1.0; 1.0]
            end
            
            dVal = options.Direction;
            dNorm = norm(dVal);
            if dNorm < 1e-12
                dVal = [0; 0; -1];
                dNorm = 1.0;
            end
            obj.Direction = dVal / dNorm;
            obj.Intensity = options.Intensity;
        end
        
        function [intensity, dirToLight, distance] = getLightAt(obj, ~)
            intensity = obj.Intensity;
            dirToLight = -obj.Direction; % Direction towards light is opposite of travel
            distance = Inf;
        end
    end
end
