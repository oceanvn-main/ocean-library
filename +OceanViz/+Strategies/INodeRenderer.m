classdef (Abstract) INodeRenderer < handle
    % INODERENDERER  Abstract interface for drawing a single NN node on axes.
    %
    %   Implementations decide how a node is visually represented:
    %   a circle, an icon, an image patch, or any MATLAB graphics object.
    %
    %   Every concrete renderer must implement draw(), which places exactly
    %   one visual element per call at the specified (x, y) position.
    %
    %   Usage:
    %     strat = OceanViz.Strategies.NetworkGraphStrategy( ...
    %                 'NodeRenderer', MyCustomRenderer());
    %
    %   Built-in renderers:
    %     CircleNodeRenderer  — filled scatter circles (default)
    %
    % See also: OceanViz.Strategies.CircleNodeRenderer
    %           OceanViz.Strategies.NetworkGraphStrategy

    methods (Abstract)
        draw(obj, ax, x, y, sz, color, state, isDead)
        % DRAW  Render one node at (x, y) on axes ax.
        %
        % INPUTS:
        %   ax    - MATLAB axes handle (uiaxes or regular axes)
        %   x     - scalar double, X coordinate
        %   y     - scalar double, Y coordinate
        %   sz    - scalar double, marker area (scatter units) or display size
        %   color - (1×3) double RGB color in [0, 1]
        %   state - string: "Active" | "Disabled" | "Highlighted"
    end
end
