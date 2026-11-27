classdef PiecewiseAffineWarp
    % PIECEWISEAFFINEWARP  Piecewise affine image warping over a triangle mesh.
    %
    %   Piecewise affine warping partitions the target image into a set of
    %   triangles, mapping each triangle to a corresponding triangle in the
    %   source image via a local affine transformation. Within each triangle,
    %   barycentric coordinates are used to perform the backward mapping and
    %   reconstruct the warped image.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-30

    methods (Static)
        function I_warped = warp(I_src, pts_src, pts_tgt, tri, out_size)
            % WARP  Warps the source image using piecewise affine mapping.
            %
            %   I_warped = warp(I_src, pts_src, pts_tgt, tri, out_size)
            %
            %   Inputs:
            %     I_src    - Source image (H x W x C, double or uint8)
            %     pts_src  - Source vertices (N x 2 matrix, each row is [x, y])
            %     pts_tgt  - Target vertices (N x 2 matrix, each row is [x, y])
            %     tri      - Triangle connectivity (M x 3 matrix of 1-based indices)
            %     out_size - Optional [H_tgt, W_tgt] vector specifying output size
            %                (defaults to source size)

            arguments
                I_src {mustBeNumeric}
                pts_src (:, 2) double
                pts_tgt (:, 2) double
                tri (:, 3) double
                out_size (1, 2) double = [size(I_src, 1), size(I_src, 2)]
            end

            I_dbl = double(I_src);
            [H_src, W_src, C] = size(I_dbl, 1:3);
            H_tgt = out_size(1);
            W_tgt = out_size(2);

            % Initialize output image and accumulator/weight map
            I_warped = zeros(H_tgt, W_tgt, C);
            
            num_tri = size(tri, 1);

            for t_idx = 1:num_tri
                % 1. Get triangle vertices
                idx = tri(t_idx, :);
                
                % Target vertices (x, y)
                x0 = pts_tgt(idx(1), 1); y0 = pts_tgt(idx(1), 2);
                x1 = pts_tgt(idx(2), 1); y1 = pts_tgt(idx(2), 2);
                x2 = pts_tgt(idx(3), 1); y2 = pts_tgt(idx(3), 2);

                % Source vertices (u, v)
                u0 = pts_src(idx(1), 1); v0 = pts_src(idx(1), 2);
                u1 = pts_src(idx(2), 1); v1 = pts_src(idx(2), 2);
                u2 = pts_src(idx(3), 1); v2 = pts_src(idx(3), 2);

                % 2. Bounding Box in Target Space
                min_x = max(1, floor(min([x0, x1, x2])));
                max_x = min(W_tgt, ceil(max([x0, x1, x2])));
                min_y = max(1, floor(min([y0, y1, y2])));
                max_y = min(H_tgt, ceil(max([y0, y1, y2])));

                if min_x > max_x || min_y > max_y
                    continue;
                end

                % Create coordinate grids for target bounding box
                [X_box, Y_box] = meshgrid(min_x:max_x, min_y:max_y);
                x_pts = X_box(:);
                y_pts = Y_box(:);

                % 3. Barycentric Coordinates Transformation Matrix
                % Matrix maps [alpha; beta; gamma] to [x; y; 1]
                T = [x0, x1, x2;
                     y0, y1, y2;
                     1,   1,   1];

                det_T = det(T);
                if abs(det_T) < 1e-9
                    continue;
                end

                % Solve for barycentric coordinates for all pixels in box:
                % T_inv * [x_pts'; y_pts'; ones(1, len)]
                T_inv = inv(T);
                
                % Unrolled matrix-vector multiplication for speed
                alpha = T_inv(1,1)*x_pts + T_inv(1,2)*y_pts + T_inv(1,3);
                beta  = T_inv(2,1)*x_pts + T_inv(2,2)*y_pts + T_inv(2,3);
                gamma = T_inv(3,1)*x_pts + T_inv(3,2)*y_pts + T_inv(3,3);

                % Point is inside if all barycentric coordinates are non-negative
                % We use a small epsilon tolerance to cover edges cleanly
                eps_tol = -1e-9;
                inside = (alpha >= eps_tol) & (beta >= eps_tol) & (gamma >= eps_tol);

                if ~any(inside)
                    continue;
                end

                % Keep only pixels inside the target triangle
                x_in = x_pts(inside);
                y_in = y_pts(inside);
                alpha_in = alpha(inside);
                beta_in  = beta(inside);
                gamma_in = gamma(inside);

                % 4. Map to Source Space
                % Since barycentric coordinates are invariant under affine warping:
                u_src = alpha_in * u0 + beta_in * u1 + gamma_in * u2;
                v_src = alpha_in * v0 + beta_in * v1 + gamma_in * v2;

                % 5. Bilinear Interpolation
                u0_s = floor(u_src);
                v0_s = floor(v_src);
                u1_s = u0_s + 1;
                v1_s = v0_s + 1;

                % Clamp to source image boundaries
                u0_c = max(1, min(W_src, u0_s));
                u1_c = max(1, min(W_src, u1_s));
                v0_c = max(1, min(H_src, v0_s));
                v1_c = max(1, min(H_src, v1_s));

                du = u_src - u0_s;
                dv = v_src - v0_s;

                w00 = (1 - du) .* (1 - dv);
                w01 = du .* (1 - dv);
                w10 = (1 - du) .* dv;
                w11 = du .* dv;

                % Compute linear indices
                idx00 = v0_c + (u0_c - 1) * H_src;
                idx01 = v0_c + (u1_c - 1) * H_src;
                idx10 = v1_c + (u0_c - 1) * H_src;
                idx11 = v1_c + (u1_c - 1) * H_src;

                % Write to target canvas
                tgt_idx = y_in + (x_in - 1) * H_tgt;

                for c = 1:C
                    Ic = I_dbl(:, :, c);
                    val = w00 .* Ic(idx00) + w01 .* Ic(idx01) + w10 .* Ic(idx10) + w11 .* Ic(idx11);
                    I_warped(tgt_idx + (c - 1) * H_tgt * W_tgt) = val;
                end
            end
            
            % Cast back to original type if uint8
            if isa(I_src, 'uint8')
                I_warped = uint8(max(0, min(255, round(I_warped))));
            end
        end
    end
end
