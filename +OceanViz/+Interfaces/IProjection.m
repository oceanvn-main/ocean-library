classdef (Abstract) IProjection < handle
    % IPROJECTION Contract for pluggable 2D/3D rendering logic.
    %
    % This interface defines how data is projected onto a 2D or 3D coordinate 
    % system. It uses a "Double-Dispatch" pattern:
    %   1. The Strategy (e.g. Surface) receives the plot call.
    %   2. The Strategy then delegates the specific geometry to an IProjection.
    %
    % This allows for extreme flexibility: one can add "Polar", "Volumetric", 
    % or "Hyperbolic" projections without changing the core strategy logic.
    
    methods (Abstract)
        % PROJECT Renders the data onto the axes using a specific geometry.
        %
        % Args:
        %   obj     : The concrete Projection instance (e.g. Planar, Manifold).
        %   strategy: The parent Surface strategy. This is passed back so the 
        %             projection can use shared utilities like 'drawScatter'.
        %   ax      : Target Axes handle.
        %   X, Y, Z : Normalized grid data (e.g. meshgrids).
        %   data    : The full data struct (contains raw points, labels, etc.).
        %   options : UI options (Title, Alpha, ViewAngle).
        %   colors  : The resolved theme colors from the Parent Core.
        project(obj, strategy, ax, X, Y, Z, data, options, colors)
    end
end
