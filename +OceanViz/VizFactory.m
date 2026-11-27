classdef VizFactory
    % VIZFACTORY Static Factory to dispense Visualization Strategies.
    %
    % NAMESPACE: OceanStats.Viz
    %
    % OVERVIEW:
    %   This factory centralizes the logic of "Which plotter do I use?".
    %   It analyzes the data dimensions and the task type to hand back
    %   the correct IVizStrategy.
    %
    % USAGE:
    %   strategy = OceanStats.Viz.VizFactory.getStrategy(data);
    %   strategy.draw(ax, data, options, theme);
    %
    % AUTHOR: OceanStats Team
    % DATE: 2026-02-04
    
    methods (Static)
        
        function strategy = getStrategy(data)
            % GETSTRATEGY Returns the appropriate IVizStrategy instance.
            %
            % LOGIC:
            %   1. IF Data is 1D (Feature size == 1) -> Use ErrorRibbonStrategy.
            %   2. IF Data is Multi-Dim (>1)         -> Use SurfaceManifoldStrategy.
            %   (Future: Add checks for 'Heatmap' or 'Contour' specific requests)
            
            import OceanViz.Strategies.SurfaceManifoldStrategy
            import OceanViz.Strategies.ErrorRibbonStrategy
            
            % Check Dimensionality of Input X
            [~, P] = size(data.X);
            
            if P == 1
                % Case A: Simple 1D Regression (Line + Ribbons)
                strategy = ErrorRibbonStrategy();
            else
                % Case B: High-Dim Surface / Decision Boundary
                strategy = SurfaceManifoldStrategy();
            end
        end
        
    end
end