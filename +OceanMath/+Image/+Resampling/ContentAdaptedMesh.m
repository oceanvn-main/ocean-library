classdef ContentAdaptedMesh
    % CONTENTADAPTEDMESH  Generates non-uniform meshes adapted to image features.
    %
    %   ContentAdaptedMesh computes the gradient magnitude of an image and
    %   distributes vertices adaptively along high-contrast edges/boundaries,
    %   using a minimum-distance spacing constraint. A Delaunay triangulation
    %   is then computed over the vertices.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-30

    methods (Static)
        function [pts, tri] = generate(I, num_vertices, options)
            % GENERATE  Generates the content-adapted mesh vertices and triangulation.
            %
            %   [pts, tri] = generate(I, num_vertices, options)
            %
            %   Inputs:
            %     I            - Source image (H x W or H x W x C)
            %     num_vertices - Target number of vertices in the mesh
            %     options      - Name-value options:
            %                      MinDistance: Minimum distance (in pixels)
            %                                   between vertices (default: 12)
            %
            %   Outputs:
            %     pts - N_pts x 2 matrix of vertex coordinates [X, Y]
            %     tri - N_tri x 3 connectivity matrix

            arguments
                I {mustBeNumeric}
                num_vertices (1,1) double {mustBeInteger, mustBePositive}
                options.MinDistance (1,1) double = 12
            end

            % 1. Convert image to grayscale and double
            if size(I, 3) > 1
                % Simple RGB to Grayscale conversion
                I_gray = 0.2989 * double(I(:,:,1)) + 0.5870 * double(I(:,:,2)) + 0.1140 * double(I(:,:,3));
            else
                I_gray = double(I);
            end

            [H, W] = size(I_gray);

            % 2. Compute gradient magnitude using Sobel kernels
            Gx = conv2(I_gray, [-1, 0, 1; -2, 0, 2; -1, 0, 1] / 8, 'same');
            Gy = conv2(I_gray, [-1, -2, -1; 0, 0, 0; 1, 2, 1] / 8, 'same');
            G = sqrt(Gx.^2 + Gy.^2);

            % 3. Initialize vertices with boundary points
            % This ensures Delaunay triangulation covers the entire canvas.
            n_boundary = 8;
            edge_x = round(linspace(1, W, n_boundary));
            edge_y = round(linspace(1, H, n_boundary));
            boundary_pts = [
                edge_x', ones(length(edge_x), 1);
                edge_x', H * ones(length(edge_x), 1);
                ones(length(edge_y), 1), edge_y';
                W * ones(length(edge_y), 1), edge_y'
            ];
            pts = unique(boundary_pts, 'rows');

            % 4. Select vertices at local maxima of gradient magnitude
            [~, sort_idx] = sort(G(:), 'descend');
            [sorted_r, sorted_c] = ind2sub([H, W], sort_idx);

            min_dist = options.MinDistance;

            for i = 1:length(sort_idx)
                if size(pts, 1) >= num_vertices
                    break;
                end

                candidate = [sorted_c(i), sorted_r(i)];

                % Verify minimum distance constraint from already selected points
                dx = pts(:, 1) - candidate(1);
                dy = pts(:, 2) - candidate(2);
                dist = sqrt(dx.^2 + dy.^2);

                if all(dist >= min_dist)
                    pts = [pts; candidate];
                end
            end

            % 5. Compute Delaunay Triangulation
            tri = delaunay(pts(:, 1), pts(:, 2));
        end
    end
end
