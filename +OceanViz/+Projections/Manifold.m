classdef Manifold < OceanViz.Interfaces.IProjection
    % MANIFOLD Professional 3D geometric projection strategy.
    %
    % This class transforms model outcomes into physical 3D manifolds.
    % It supports different "Z-Modes" to handle either probability surfaces 
    % or discrete classification cliffs.
    
    properties
        % ZMode controls how the vertical axis is interpreted:
        %   'Probability' : Smooth ramps / gradients (Ideal for Softmax/PolyMARS).
        %   'Label'       : Sharp integer cliffs (Standard discrete decisions).
        ZMode (1,1) string {mustBeMember(ZMode, ["Probability", "Label"])} = "Probability"
    end
    
    methods
        function obj = Manifold(varargin)
            % Constructor allows for property-value pairing (e.g. Manifold('ZMode', 'Label'))
            if nargin > 0
                for i = 1:2:length(varargin)
                    obj.(varargin{i}) = varargin{i+1};
                end
            end
        end
        
        function project(obj, strategy, ax, X, Y, Z, data, options, colors)
            % GLOWING 3D MANIFOLD
            
            % 1. Intelligent Z-Lifting
            % If Z is a 3D Tensor (Height x Width x Classes), we condense it
            if ndims(Z) == 3
                if strcmp(obj.ZMode, 'Probability')
                    % Soft Manifold: Weighted average of class indices
                    K = size(Z, 3);
                    weights = reshape(1:K, 1, 1, K);
                    Z_lift = sum(Z .* weights, 3);
                else
                    % Hard Manifold: Discrete Argmax heights
                    [~, Z_lift] = max(Z, [], 3);
                end
            else
                % Fallback: Use Z as-is if already flattened
                Z_lift = Z;
            end
            
            % 2. Render Primary Surface
            surf(ax, X, Y, Z_lift, ...
                'EdgeColor', 'none', 'FaceColor', 'interp', ...
                'FaceAlpha', options.FaceAlpha, ...
                'AmbientStrength', 0.6, 'DiffuseStrength', 0.8);
            
            if isfield(strategy.Parent.Colors, 'Gradient')
                colormap(ax, strategy.Parent.Colors.Gradient);
            end
            
            % 3. 3D Boundaries
            if options.ShowContour
                uniqueLevels = unique(Z_lift(:));
                if length(uniqueLevels) > 1
                    if strcmp(obj.ZMode, 'Label') || ndims(Z) == 3
                        boundaryLevels = uniqueLevels(1:end-1) + 0.5;
                    else
                        boundaryLevels = linspace(min(Z_lift(:)), max(Z_lift(:)), 5);
                    end
                    contour3(ax, X, Y, Z_lift, boundaryLevels, ...
                        'LineWidth', 2.5, 'LineColor', colors.Boundary);
                end
            end
            
            % 4. Vector Fields (Gradients)
            if options.ShowGradient
                [FX, FY] = gradient(Z_lift);
                stride = 5;
                qx = X(1:stride:end, 1:stride:end);
                qy = Y(1:stride:end, 1:stride:end);
                qz = Z_lift(1:stride:end, 1:stride:end);
                qfx = FX(1:stride:end, 1:stride:end);
                qfy = FY(1:stride:end, 1:stride:end);
                
                quiver3(ax, qx, qy, qz, qfx, qfy, zeros(size(qfx)), ...
                    'Color', colors.Secondary, 'LineWidth', 1.2);
            end
            
            % 4. Multi-Dimensional Scatter
            strategy.drawScatter(ax, data, 3);
            
            % 5. Professional Lighting & Environment
            % Uses Gouraud lighting for smoother depth perception.
            view(ax, options.ViewAngle);
            camlight(ax, 'headlight');
            lighting(ax, 'gouraud');
            
            xlabel(ax, options.XLabel, 'Interpreter', 'latex');
            ylabel(ax, options.YLabel, 'Interpreter', 'latex');
            zlabel(ax, options.ZLabel, 'Interpreter', 'latex');
            title(ax, options.Title, 'Interpreter', 'latex', 'Color', colors.Text);
        end
    end
end
