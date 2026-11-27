classdef IRenderer < handle
    % IRENDERER  Abstract base interface for 3D room light renderers.
    
    methods (Abstract)
        % RENDER Render the scene from the camera's perspective.
        %
        % INPUTS:
        %   scene            - Scene instance containing objects and lights
        %   cameraProjection - CameraProjection instance containing intrinsics/extrinsics
        %   varargin         - Additional options (axes handle, resolution, etc.)
        %
        % OUTPUTS:
        %   output - Rendered result (e.g. RGB image matrix or graphical handle array)
        output = render(obj, scene, cameraProjection, varargin)
    end
end
