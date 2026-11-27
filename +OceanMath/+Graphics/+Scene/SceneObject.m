classdef SceneObject < handle
    % SCENEOBJECT  Instantiated 3D object in the scene.
    
    properties
        Geometry                         % Instance of IGeometry
        Material                         % Instance of Material
        Pose (4,4) double = eye(4)       % T_local2world: Local-to-world transform
    end
    
    methods
        function obj = SceneObject(geometry, material, options)
            arguments
                geometry
                material
                options.Pose (4,4) double = eye(4)
            end
            obj.Geometry = geometry;
            obj.Material = material;
            obj.Pose = options.Pose;
        end
        
        function [t, normal] = intersect(obj, rayOrigin, rayDir)
            % INTERSECT Intersects world-space ray with this transformed object.
            
            % Compute world-to-local transform
            T_inv = inv(obj.Pose);
            
            % Transform ray origin (homogeneous point)
            O_hom = T_inv * [rayOrigin(:); 1];
            O_local = O_hom(1:3);
            
            % Transform ray direction (homogeneous direction vector)
            D_hom = T_inv * [rayDir(:); 0];
            D_local = D_hom(1:3);
            
            % Normalize direction in local space
            d_norm = norm(D_local);
            if d_norm > 1e-12
                D_local = D_local / d_norm;
            end
            
            % Intersect in local coordinate space
            [t_local, N_local] = obj.Geometry.intersect(O_local, D_local);
            
            t = NaN;
            normal = [0; 0; 0];
            if isnan(t_local)
                return;
            end
            
            % Transform local intersection point back to world space
            P_local = O_local + t_local * D_local;
            P_hom = obj.Pose * [P_local; 1];
            P_world = P_hom(1:3);
            
            % Compute actual distance in world coordinate space
            t = norm(P_world - rayOrigin(:));
            
            % Transform surface normal using H^(-T) (co-vector transform rule)
            R_sub = obj.Pose(1:3, 1:3);
            N_world = (inv(R_sub)') * N_local(:);
            
            % Normalize normal vector
            n_norm = norm(N_world);
            if n_norm > 1e-12
                normal = N_world / n_norm;
            else
                normal = [0; 0; 1];
            end
        end
        
        function [vertices, faces] = toMesh(obj)
            % TOMESH Generate mesh vertices in world coordinates.
            [V_local, faces] = obj.Geometry.toMesh();
            vertices = OceanMath.Geometry.Transform3D.transformPoints(obj.Pose, V_local);
        end
    end
end
