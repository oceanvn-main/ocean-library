classdef ILight < handle
    % ILIGHT  Abstract base interface for light sources in 3D simulation.
    
    methods (Abstract)
        % GETLIGHTAT Calculate light properties at a specific 3D surface point.
        %
        % INPUTS:
        %   point - [3 x 1] coordinate of point on surface
        %
        % OUTPUTS:
        %   intensity  - [3 x 1] light color/intensity reaching the point (RGB)
        %   dirToLight - [3 x 1] unit vector pointing from the surface point towards the light source
        %   distance   - distance from point to light source (Inf for directional light)
        [intensity, dirToLight, distance] = getLightAt(obj, point)
    end
end
