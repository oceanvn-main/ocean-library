classdef ErrorRibbonStrategy < OceanViz.Interfaces.IVizStrategy
    % ERRORRIBBONSTRATEGY Concrete implementation for 1D Regression Plots.
    %
    % NAMESPACE: OceanViz.Strategies
    % INHERITS: OceanViz.Interfaces.IVizStrategy
    %
    % OVERVIEW:
    %   This strategy handles the visualization of 1D Regression models.
    %   It renders:
    %     1. The Raw Data (Scatter points).
    %     2. The Best Fit Model (Line).
    %     3. (Optional) The Ground Truth (Dashed Line).
    %     4. (Future) Confidence Interval Ribbons (if SE is available).
    %
    % ALGORITHM:
    %   1. Sorts the Input X (essential for line plotting).
    %   2. Predicts Y using the provided Strategy and Model.
    %   3. Renders the Scatter layer.
    %   4. Renders the Fit Line layer on top.
    %
    % AUTHOR: OceanStats Team
    % DATE: 2026-02-04
    
    methods
        function draw(obj, ax, data, options, theme)
            % DRAW Implementation of the IVizStrategy contract.
            
            % 1. Validation: This strategy only works for 1D inputs
            if size(data.X, 2) > 1
                warning('ErrorRibbonStrategy:DimMismatch', ...
                    'This strategy expects 1D input. Use SurfaceManifoldStrategy for High-Dim.');
                return;
            end
            
            % 2. Prepare Data (Sort X for clean line plotting)
            [X_sort, idx_sort] = sort(data.X);
            
            % 3. Model Prediction
            % Note: In the future, if PredictFcn returns [Y, SE], capture it here.
            Y_pred = data.Strategy.PredictFcn(data.Model, X_sort, data.Level);
            
            % 4. Render Layers
            obj.renderScatter(ax, data.X, data.Y, theme);
            obj.renderRibbon(ax, X_sort, Y_pred, [], theme); % SE is empty for now
            obj.renderFitLine(ax, X_sort, Y_pred, data.Level, theme);
            obj.renderGroundTruth(ax, X_sort, options, theme);
            
            % 5. Styling
            obj.styleAxes(ax, options);
            
            % 6. Legend
            legend(ax, 'Location', 'best', ...
                'Interpreter', 'latex', ...
                'TextColor', theme.Text, ...
                'EdgeColor', 'none', ...
                'Color', [theme.Background 0.8]);
        end
    end
    
    methods (Access = private)
        
        function renderScatter(~, ax, X, Y, theme)
            scatter(ax, X, Y, 25, ...
                'MarkerEdgeColor', 'none', ...
                'MarkerFaceColor', theme.Scatter, ...
                'MarkerFaceAlpha', 0.5, ...
                'DisplayName', 'Data'); 
            hold(ax, 'on');
        end
        
        function renderRibbon(~, ax, X, Y_pred, SE, theme)
            % Future-Proofing: This is where we would plot the shaded area
            % using fill() or patch() if SE (Standard Error) were provided.
            if isempty(SE), return; end
            
            % Example logic for future implementation:
            % upper = Y_pred + 1.96 * SE;
            % lower = Y_pred - 1.96 * SE;
            % fill(ax, [X; flipud(X)], [upper; flipud(lower)], ...
            %      theme.Primary, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
        
        function renderFitLine(~, ax, X, Y_pred, level, theme)
            plot(ax, X, Y_pred, ...
                'LineWidth', 2, ...
                'Color', theme.Primary, ...
                'DisplayName', sprintf('Best Fit (d=%d)', level));
        end
        
        function renderGroundTruth(~, ax, X, options, theme)
            if isfield(options, 'GroundTruthFcn') && ~isempty(options.GroundTruthFcn)
                Y_true = options.GroundTruthFcn(X);
                plot(ax, X, Y_true, '--', ...
                    'Color', theme.Secondary, ...
                    'LineWidth', 1.5, ...
                    'DisplayName', 'Truth'); 
            end
        end
        
        function styleAxes(~, ax, options)
            grid(ax, 'on');
            
            titleStr = '';
            if isfield(options, 'Title'), titleStr = options.Title; end
            title(ax, titleStr, 'Interpreter', 'latex');
            
            xlabel(ax, 'Input X', 'Interpreter', 'latex');
            ylabel(ax, 'Target Y', 'Interpreter', 'latex');
        end
        
    end
end