classdef SphereGeometry < OceanMath.Vision.Geometry.IGeometry
    % SPHEREGEOMETRY  Models a 3D sphere geometry.
    
    properties
        Center (3,1) double = [0; 0; 0]      % Center of the sphere
        Radius (1,1) double = 1.0            % Radius of the sphere
    end
    
    methods
        function obj = SphereGeometry(options)
            arguments
                options.Center (3,1) double = [0; 0; 0]
                options.Radius (1,1) double = 1.0
            end
            obj.Center = options.Center;
            obj.Radius = options.Radius;
        end
        
        function [t, normal] = intersect(obj, rayOrigin, rayDir)
            % INTERSECT Find the ray-sphere intersection.
            %
            % Ray equation: P(t) = O + t * D
            % Sphere equation: norm(P - C)^2 = R^2
            % Substituting: a*t^2 + b*t + c = 0
            %   a = dot(D, D) = 1 (unit vector)
            %   b = 2 * dot(D, O - C)
            %   c = dot(O - C, O - C) - R^2
            
            arguments
                obj
                rayOrigin (3,1) double
                rayDir (3,1) double
            end
            
            t = NaN;
            normal = [0; 0; 0];
            
            % Normalize ray direction
            d = rayDir / norm(rayDir);
            
            oc = rayOrigin - obj.Center;
            
            a = 1.0; % since d is normalized
            b = 2.0 * dot(d, oc);
            c = dot(oc, oc) - obj.Radius^2;
            
            disc = b^2 - 4.0 * a * c;
            
            if disc < 0
                return; % No intersection
            end
            
            % Find closest positive intersection point
            t1 = (-b - sqrt(disc)) / (2.0 * a);
            t2 = (-b + sqrt(disc)) / (2.0 * a);
            
            if t1 > 1e-4 && t2 > 1e-4
                t_val = min(t1, t2);
            elseif t1 > 1e-4
                t_val = t1;
            elseif t2 > 1e-4
                t_val = t2;
            else
                return; % Both intersections are behind ray origin
            end
            
            t = t_val;
            p_int = rayOrigin + t * d;
            normal = (p_int - obj.Center) / obj.Radius;
        end
        
        function [vertices, faces] = toMesh(obj)
            % TOMESH Generate a latitude-longitude mesh for the sphere.
            
            nTheta = 24; % Latitude segments
            nPhi = 24;   % Longitude segments
            
            theta = linspace(0, pi, nTheta);
            phi = linspace(0, 2*pi, nPhi);
            
            [Theta, Phi] = meshgrid(theta, phi);
            
            X = obj.Center(1) + obj.Radius * sin(Theta) .* cos(Phi);
            Y = obj.Center(2) + obj.Radius * sin(Theta) .* sin(Phi);
            Z = obj.Center(3) + obj.Radius * cos(Theta);
            
            vertices = [X(:)'; Y(:)'; Z(:)'];
            
            % Generate triangular faces
            faces = [];
            for c = 1:(nTheta-1)
                for r = 1:(nPhi-1)
                    % Corner indices of the quad cell
                    p1 = (c-1)*nPhi + r;
                    p2 = (c-1)*nPhi + r + 1;
                    p3 = c*nPhi + r;
                    p4 = c*nPhi + r + 1;
                    
                    % Triangulate quad
                    faces = [faces, [p1; p3; p2], [p2; p3; p4]]; %#ok<AGROW>
                end
            end
        end
        
        function normal = getNormal(obj, point)
            % GETNORMAL Get unit normal at a point on the surface.
            normal = (point - obj.Center) / obj.Radius;
        end
    end
end
