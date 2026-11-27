classdef ILightingStrategy < handle
    % ILIGHTINGSTRATEGY  Abstract strategy for custom lighting equations.
    
    methods (Abstract)
        % COMPUTELIGHTING Compute final reflected color at a surface point.
        %
        % INPUTS:
        %   point     - [3 x 1] coordinates of surface point
        %   normal    - [3 x 1] unit surface normal at point
        %   viewDir   - [3 x 1] unit view direction vector (pointing from surface towards camera)
        %   material  - Material object containing surface reflection coefficients
        %   lights    - cell array of ILight instances
        %   scene     - Scene instance (used for ray tracing occlusion/shadows, optional)
        %
        % OUTPUTS:
        %   color - [3 x 1] computed RGB color [R; G; B]
        color = computeLighting(obj, point, normal, viewDir, material, lights, scene)
    end
end
