classdef Scene < handle
    % SCENE  Container for all 3D objects and light sources in the simulation.
    
    properties
        Objects = {}      % Cell array of SceneObject instances
        Lights = {}       % Cell array of ILight instances
    end
    
    methods
        function obj = Scene()
            % SCENE  Construct an empty scene.
        end
        
        function addObject(obj, sceneObj)
            % ADDOBJECT Add a SceneObject to the scene.
            arguments
                obj
                sceneObj OceanMath.Graphics.Scene.SceneObject
            end
            obj.Objects{end+1} = sceneObj;
        end
        
        function addLight(obj, lightSource)
            % ADDLIGHT Add a light source to the scene.
            obj.Lights{end+1} = lightSource;
        end
        
        function [closestObj, tMin, normal] = intersect(obj, rayOrigin, rayDir)
            % INTERSECT Find the nearest object intersection for a given ray.
            %
            % Loops through all objects, returning the closest hit object, its
            % intersection parameter t, and the surface normal at the hit point.
            
            arguments
                obj
                rayOrigin (3,1) double
                rayDir (3,1) double
            end
            
            closestObj = [];
            tMin = Inf;
            normal = [0; 0; 0];
            
            % Normalize ray direction
            d = rayDir / norm(rayDir);
            
            for i = 1:numel(obj.Objects)
                [t, N] = obj.Objects{i}.intersect(rayOrigin, d);
                if ~isnan(t) && t < tMin && t > 1e-4
                    tMin = t;
                    closestObj = obj.Objects{i};
                    normal = N;
                end
            end
            
            if isinf(tMin)
                tMin = NaN;
            end
        end
    end
end
