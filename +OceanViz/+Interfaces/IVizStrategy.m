classdef (Abstract) IVizStrategy < handle
    % IVIZSTRATEGY The Contract for all Visualization Logic.
    %
    % NAMESPACE: OceanViz.Interfaces
    % FILE LOCATION: +OceanViz/+Interfaces/IVizStrategy.m
    %
    % OVERVIEW:
    %   This abstract class defines the standard interface for any class that
    %   wants to draw results from the OceanStats Audit Engine.
    %
    % USAGE:
    %   1. Subclass this file.
    %   2. Implement the 'draw' method in the subclass.
    %
    % AUTHOR: OceanStats Team
    % DATE: 2026-02-04
    
    methods (Abstract)
        
        % DRAW Renders the specific visualization onto the provided axes.
        %
        % INPUTS:
        %   obj     : The strategy instance
        %   ax      : Target axes handle
        %   data    : Struct containing .X, .Y, .Model, .Level, .Meta
        %   options : Struct containing flags (.ShowGradient, etc.)
        %   theme   : Struct containing colors (.Primary, .Background, etc.)
        draw(obj, ax, data, options, theme)
        
    end
end