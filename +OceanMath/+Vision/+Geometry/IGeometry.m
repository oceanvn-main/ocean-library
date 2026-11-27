classdef IGeometry < handle
    % IGEOMETRY  Abstract base interface for 3D simulation geometries.
    
    methods (Abstract)
        % INTERSECT Intersect ray starting at rayOrigin with direction rayDir.
        %
        % INPUTS:
        %   rayOrigin - [3 x 1] coordinate point of ray origin
        %   rayDir    - [3 x 1] unit vector representing ray direction
        %
        % OUTPUTS:
        %   t      - distance from origin to intersection (NaN if no intersection)
        %   normal - [3 x 1] unit surface normal at the intersection point
        [t, normal] = intersect(obj, rayOrigin, rayDir)
        
        % TOMESH Generate mesh representation of this geometry.
        %
        % OUTPUTS:
        %   vertices - [3 x V] matrix of vertex coordinates
        %   faces    - [3 x F] matrix of face indices (triangular faces)
        [vertices, faces] = toMesh(obj)
        
        % GETNORMAL Compute surface normal at a specific 3D point on the surface.
        %
        % INPUTS:
        %   point - [3 x 1] coordinate point on the surface
        %
        % OUTPUTS:
        %   normal - [3 x 1] unit surface normal vector
        normal = getNormal(obj, point)
    end
end
