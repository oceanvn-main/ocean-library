classdef BilinearQuadWarp
    % BILINEARQUADWARP  Quadrilateral grid image warping using triangulation.
    %
    %   BilinearQuadWarp takes a grid of source control points and target
    %   control points, decomposes the quadrilateral elements into triangles,
    %   and warps the image using PiecewiseAffineWarp.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-30

    methods (Static)
        function I_warped = warp(I_src, grid_src_x, grid_src_y, grid_tgt_x, grid_tgt_y, out_size)
            % WARP  Warps the source image using quadrilateral grid coordinates.
            %
            %   I_warped = warp(I_src, grid_src_x, grid_src_y, grid_tgt_x, grid_tgt_y, out_size)
            %
            %   Inputs:
            %     I_src      - Source image (H x W x C)
            %     grid_src_x - Grid matrix (R x C) of source X coordinates
            %     grid_src_y - Grid matrix (R x C) of source Y coordinates
            %     grid_tgt_x - Grid matrix (R x C) of target X coordinates
            %     grid_tgt_y - Grid matrix (R x C) of target Y coordinates
            %     out_size   - Optional [H_tgt, W_tgt] vector specifying output size
            %                  (defaults to source size)

            arguments
                I_src {mustBeNumeric}
                grid_src_x (:, :) double
                grid_src_y (:, :) double
                grid_tgt_x (:, :) double
                grid_tgt_y (:, :) double
                out_size (1, 2) double = [size(I_src, 1), size(I_src, 2)]
            end

            % Check size parity of grid inputs
            sz = size(grid_src_x);
            if ~isequal(sz, size(grid_src_y)) || ~isequal(sz, size(grid_tgt_x)) || ~isequal(sz, size(grid_tgt_y))
                error('BilinearQuadWarp:DimensionMismatch', 'All grid coordinate matrices must have the same dimensions.');
            end

            [rows, cols] = size(grid_src_x);
            if rows < 2 || cols < 2
                error('BilinearQuadWarp:GridTooSmall', 'Grid dimensions must be at least 2x2 to define quadrilateral patches.');
            end

            % 1. Flatten grids to vertices matrix (N x 2)
            pts_src = [grid_src_x(:), grid_src_y(:)];
            pts_tgt = [grid_tgt_x(:), grid_tgt_y(:)];

            % 2. Triangulate quadrilateral elements
            % Each quad (r, c) is split along its TL-BR diagonal into 2 triangles:
            % Triangle 1: Top-Left, Top-Right, Bottom-Left
            % Triangle 2: Top-Right, Bottom-Right, Bottom-Left
            num_quads = (rows - 1) * (cols - 1);
            tri = zeros(2 * num_quads, 3);
            t_idx = 1;

            for c = 1:(cols - 1)
                for r = 1:(rows - 1)
                    % Flat column-major indices for corner vertices
                    idx_TL = r + (c - 1) * rows;
                    idx_TR = r + c * rows;
                    idx_BL = r + 1 + (c - 1) * rows;
                    idx_BR = r + 1 + c * rows;

                    % Triangle 1: TL, TR, BL
                    tri(t_idx, :) = [idx_TL, idx_TR, idx_BL];
                    t_idx = t_idx + 1;

                    % Triangle 2: TR, BR, BL
                    tri(t_idx, :) = [idx_TR, idx_BR, idx_BL];
                    t_idx = t_idx + 1;
                end
            end

            % 3. Delegate to PiecewiseAffineWarp
            I_warped = OceanMath.Image.Resampling.PiecewiseAffineWarp.warp(...
                I_src, pts_src, pts_tgt, tri, out_size);
        end
    end
end
