classdef TriMeshStrategy < OceanViz.Strategies.BaseStrategy
    % TRIMESHSTRATEGY  OceanViz strategy for 3-D triangle meshes and point clouds
    %   (mesh-prediction pipelines: cubified voxels, refined meshes, GT surfaces).
    %
    %   Data Contract (all optional except one of Vertices/Faces or Points):
    %     data.Vertices   (V x 3)  mesh vertices
    %     data.Faces      (F x 3)  1-based triangle indices
    %     data.FaceColor  (1 x 3)  base RGB of the mesh (default: theme Primary)
    %     data.ShowEdges  logical  draw triangle edges (default false)
    %     data.Points     (P x 3)  point cloud drawn on top (e.g. surface samples)
    %     data.PointColor (P x 1) scalar colour per point, or (1 x 3) RGB
    %     data.YUp        logical  true (default): data are y-up (object frame);
    %                              false: y-DOWN camera frame (image rows), flipped for display
    %
    %   Shading is flat Lambertian with a fixed headlight-style direction,
    %   computed here per face (|n . l|, two-sided) so the result is identical
    %   in every renderer and export, without relying on MATLAB light objects.
    %
    %   Uses Core.plot options: Title, FaceAlpha, EdgeAlpha, ViewAngle.
    %
    %   Author: Ocean Library  |  2026-09-23

    methods
        function plot(obj, ax, data, options)
            arguments
                obj
                ax
                data    struct
                options struct
            end
            colors = obj.Parent.Colors;
            yUp = ~isfield(data, 'YUp') || data.YUp;
            toDisplay = @(X) OceanViz.Strategies.TriMeshStrategy.displayCoords(X, yUp);

            cla(ax);
            hold(ax, 'on');
            if isfield(data, 'Vertices') && isfield(data, 'Faces') && ~isempty(data.Faces)
                V = toDisplay(data.Vertices);
                F = data.Faces;
                base = colors.Primary;
                if isfield(data, 'FaceColor')
                    base = data.FaceColor;
                end
                n = cross(V(F(:,2),:) - V(F(:,1),:), V(F(:,3),:) - V(F(:,1),:), 2);
                n = n ./ max(vecnorm(n, 2, 2), eps);
                light = [-0.35, -0.55, 0.75] / norm([-0.35, -0.55, 0.75]);
                shade = 0.30 + 0.70 * abs(n * light.');
                edgeColor = 'none';
                if isfield(data, 'ShowEdges') && data.ShowEdges
                    edgeColor = 0.35 * base;
                end
                patch(ax, 'Vertices', V, 'Faces', F, 'FaceVertexCData', shade .* base, ...
                    'FaceColor', 'flat', 'EdgeColor', edgeColor, 'EdgeAlpha', options.EdgeAlpha, ...
                    'FaceAlpha', options.FaceAlpha, 'LineWidth', 0.25);
            end
            if isfield(data, 'Points') && ~isempty(data.Points)
                P = toDisplay(data.Points);
                c = colors.Secondary;
                if isfield(data, 'PointColor')
                    c = data.PointColor;
                end
                scatter3(ax, P(:,1), P(:,2), P(:,3), 4, c, 'filled');
            end
            hold(ax, 'off');

            % Fixed, equal limits: every panel shows the same metric cube, so
            % meshes in neighbouring tiles are directly comparable and the axes
            % never grow out of their tile (as 'vis3d' + tick labels can).
            lim = [-1 1];
            if isfield(data, 'Limits')
                lim = data.Limits;
            end
            set(ax, 'XLim', lim, 'YLim', lim, 'ZLim', lim, 'DataAspectRatio', [1 1 1], ...
                'XTickLabel', {}, 'YTickLabel', {}, 'ZTickLabel', {}, ...
                'XColor', colors.Grid, 'YColor', colors.Grid, 'ZColor', colors.Grid, 'Box', 'on');
            view(ax, options.ViewAngle(1), options.ViewAngle(2));
            title(ax, options.Title, 'Color', colors.Text, 'Interpreter', 'none', ...
                'FontWeight', 'normal', 'FontSize', 10);
            grid(ax, 'on');
        end
    end

    methods (Static)
        function D = displayCoords(X, yUp)
            % Plot (x, z, y): MATLAB's 3-D view has z up, our data have y up.
            if yUp
                D = [X(:,1), X(:,3), X(:,2)];
            else
                D = [X(:,1), X(:,3), -X(:,2)];
            end
        end
    end
end
