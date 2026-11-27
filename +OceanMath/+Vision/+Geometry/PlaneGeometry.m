classdef PlaneGeometry < OceanMath.Vision.Geometry.IGeometry
    % PLANEGEOMETRY  Models an infinite plane or a finite rectangular plane segment.
    
    properties
        Point (3,1) double = [0; 0; 0]       % Center or anchor point of the plane
        Normal (3,1) double = [0; 0; 1]      % Unit normal vector of the plane
        IsFinite (1,1) logical = false       % If true, bounds intersection to a rectangle
        UDirection (3,1) double              % Unit tangent along width dimension
        VDirection (3,1) double              % Unit tangent along height dimension
        Width (1,1) double = 1.0             % Half-width along UDirection (for finite planes)
        Height (1,1) double = 1.0            % Half-height along VDirection (for finite planes)
    end
    
    methods
        function obj = PlaneGeometry(options)
            arguments
                options.Point (3,1) double = [0; 0; 0]
                options.Normal (3,1) double = [0; 0; 1]
                options.IsFinite (1,1) logical = false
                options.UDirection double = []
                options.VDirection double = []
                options.Width (1,1) double = 5.0
                options.Height (1,1) double = 5.0
            end
            
            obj.Point = options.Point;
            
            % Normalize normal
            nVal = options.Normal;
            nNorm = norm(nVal);
            if nNorm < 1e-12
                nVal = [0; 0; 1];
                nNorm = 1.0;
            end
            obj.Normal = nVal / nNorm;
            
            obj.IsFinite = options.IsFinite;
            obj.Width = options.Width;
            obj.Height = options.Height;
            
            % Compute tangent axes U and V if empty
            if isempty(options.UDirection) || isempty(options.VDirection)
                if abs(obj.Normal(1)) < 0.9
                    ref = [1; 0; 0];
                else
                    ref = [0; 1; 0];
                end
                uVal = cross(obj.Normal, ref);
                obj.UDirection = uVal / norm(uVal);
                obj.VDirection = cross(obj.Normal, obj.UDirection);
            else
                uVal = options.UDirection;
                vVal = options.VDirection;
                obj.UDirection = uVal / norm(uVal);
                obj.VDirection = vVal / norm(vVal);
            end
        end
        
        function [t, normal] = intersect(obj, rayOrigin, rayDir)
            % INTERSECT Find the ray-plane intersection.
            %
            % Formulas:
            %   t = dot(Point - rayOrigin, Normal) / dot(rayDir, Normal)
            
            arguments
                obj
                rayOrigin (3,1) double
                rayDir (3,1) double
            end
            
            t = NaN;
            normal = [0; 0; 0];
            
            % Normalize ray direction
            d = rayDir / norm(rayDir);
            
            denom = dot(d, obj.Normal);
            if abs(denom) < 1e-8
                return; % Ray is parallel to the plane
            end
            
            t_val = dot(obj.Point - rayOrigin, obj.Normal) / denom;
            if t_val < 1e-4
                return; % Intersection is behind or at ray origin
            end
            
            % If finite, check if intersection point is within rectangle bounds
            if obj.IsFinite
                p_int = rayOrigin + t_val * d;
                diffVec = p_int - obj.Point;
                
                u_proj = dot(diffVec, obj.UDirection);
                v_proj = dot(diffVec, obj.VDirection);
                
                if abs(u_proj) > obj.Width || abs(v_proj) > obj.Height
                    return; % Outside rectangular boundaries
                end
            end
            
            t = t_val;
            
            % Oriented normal facing towards incoming ray
            if denom > 0
                normal = -obj.Normal;
            else
                normal = obj.Normal;
            end
        end
        
        function [vertices, faces] = toMesh(obj)
            % TOMESH Generate a mesh representation of the plane.
            %
            % Returns a 4-vertex quadrilateral (2-triangle faces).
            
            w = obj.Width;
            h = obj.Height;
            if ~obj.IsFinite
                % Default mesh size for visualization of infinite plane
                w = 10.0;
                h = 10.0;
            end
            
            % Calculate 4 corners
            c1 = obj.Point - w * obj.UDirection - h * obj.VDirection;
            c2 = obj.Point + w * obj.UDirection - h * obj.VDirection;
            c3 = obj.Point + w * obj.UDirection + h * obj.VDirection;
            c4 = obj.Point - w * obj.UDirection + h * obj.VDirection;
            
            vertices = [c1, c2, c3, c4];
            
            % Two triangular faces: [c1, c2, c3] and [c1, c3, c4]
            faces = [1, 2, 3; ...
                     1, 3, 4]';
        end
        
        function normal = getNormal(obj, ~)
            % GETNORMAL Get normal at a given point (always constant for plane)
            normal = obj.Normal;
        end
    end
end
