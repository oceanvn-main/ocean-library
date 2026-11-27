classdef Planar < OceanViz.Interfaces.IProjection
    % PLANAR High-contrast 2D top-down projection strategy.
    %
    % This class implements a specific "Decision Map" architecture:
    %   - Sharp boundaries for clear category separation.
    %   - Ghosted background shading for context.
    %   - Perfect top-down orientation.
    
    properties
        % ZMode controls how the vertical axis is interpreted:
        %   'Probability' : Smooth heatmaps (Ideal for Softmax/Overlapping classes).
        %   'Label'       : Sharp discrete territories (Standard top-down boundaries).
        ZMode (1,1) string {mustBeMember(ZMode, ["Probability", "Label"])} = "Label"
    end
    
    methods
        function obj = Planar(varargin)
            % Constructor allows for property-value pairing
            if nargin > 0
                for i = 1:2:length(varargin)
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end

        function project(obj, strategy, ax, X, Y, Z, data, options, colors) %#ok<INUSL>
            % HIGH-FIDELITY TOP-DOWN HEATMAP OR DECISION MAP
            
            % 1. Intelligent Z-Handling
            if ndims(Z) == 3
                if strcmp(obj.ZMode, "Probability")
                    % Soft Manifold: Weighted average of class indices
                    K = size(Z, 3);
                    weights = reshape(1:K, 1, 1, K);
                    Z_vis = sum(Z .* weights, 3);
                else
                    % Hard Manifold: Discrete Argmax
                    [~, Z_vis] = max(Z, [], 3);
                end
            else
                Z_vis = Z;
            end
            uniqueLevels = unique(Z_vis(:));
            
            % 2. Render Base Layer
            if isfield(strategy.Parent.Colors, 'Gradient')
                cmap = strategy.Parent.Colors.Gradient;
            else
                cmap = turbo(256);
            end
            
            if strcmp(obj.ZMode, "Probability")
                % Professional Heatmap: Smooth intensity surface
                h = pcolor(ax, X, Y, Z_vis);
                set(h, 'EdgeColor', 'none', 'FaceColor', 'interp');
                shading(ax, 'interp'); 
                
                % Add Colorbar for Probability Interpretation
                cb = colorbar(ax);
                cb.Label.String = 'Probability P(C1|X)';
                cb.Label.Color = colors.Text;
                cb.Color = colors.Text;
                clim(ax, [0 1]); % FORCE scale to [0, 1] for probability
            else
                % High-Contrast Discrete Map: Hard territories
                [~, hSurf] = contourf(ax, X, Y, Z_vis, (uniqueLevels' - 0.5));
                set(hSurf, 'EdgeColor', 'none', 'FaceAlpha', 0.25);
            end
            colormap(ax, cmap);
            
            % 3. Hybrid Overlay: Decision Boundaries
            % This allows user to see "Hard Decisions" on top of "Soft Heat"
            if options.ShowContour || strcmp(obj.ZMode, "Label")
                hold(ax, 'on');
                if length(uniqueLevels) > 1
                    if strcmp(obj.ZMode, "Label") || ndims(Z) == 3
                        boundaryLevels = uniqueLevels(1:end-1) + 0.5;
                        color = colors.Boundary;
                    else
                        % In Probability mode, we show the 0.5 Decision Boundary
                        boundaryLevels = [0.5 0.5];
                        color = [1 1 1]; % High-contrast white for soft maps
                    end
                    [~, hC] = contour(ax, X, Y, Z_vis, boundaryLevels, ...
                        'LineWidth', 2.0, 'LineColor', color);
                    set(hC, 'LineStyle', '--');
                end
            end
            
            % 4. Multi-Dimensional Scatter
            strategy.drawScatter(ax, data, 2);
            
            % Perspective & Labels
            xlabel(ax, options.XLabel, 'Interpreter', 'latex');
            ylabel(ax, options.YLabel, 'Interpreter', 'latex');
            title(ax, options.Title, 'Interpreter', 'latex', 'Color', colors.Text);
            
            view(ax, 0, 90);
            axis(ax, 'tight');
            axis(ax, 'equal'); 
            grid(ax, 'on');
            ax.GridColor = [0.3 0.3 0.3];
            ax.GridLineStyle = ':';
            ax.GridAlpha = 0.5;
        end
    end
end
