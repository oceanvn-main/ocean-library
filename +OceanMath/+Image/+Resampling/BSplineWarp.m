classdef BSplineWarp
    % BSPLINEWARP  Cubic B-Spline Free-Form Deformation (FFD) image warping.
    %
    %   BSplineWarp performs local image warping using a grid of control points.
    %   Coordinates are mapped using tensor-product cubic B-spline basis functions,
    %   achieving C2-continuous smooth deformations.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-15

    methods (Static)
        function I_warped = warp(I_src, pts_src, pts_tgt, grid_sz, out_size)
            % WARP  Warp source image using Cubic B-Spline FFD mapping.
            %
            %   I_warped = warp(I_src, pts_src, pts_tgt, grid_sz, out_size)
            %
            %   Inputs:
            %     I_src    - Source image (H x W x C)
            %     pts_src  - Original grid points (N x 2, [x, y])
            %     pts_tgt  - Target grid points (N x 2, [x, y])
            %     grid_sz  - Grid size [Ry, Rx]
            %     out_size - Optional output size [H_tgt, W_tgt] (defaults to source size)

            arguments
                I_src {mustBeNumeric}
                pts_src (:, 2) double
                pts_tgt (:, 2) double
                grid_sz (1, 2) double
                out_size (1, 2) double = [size(I_src, 1), size(I_src, 2)]
            end

            I_dbl = double(I_src);
            [H_src, W_src, C] = size(I_dbl, 1:3);
            H_tgt = out_size(1);
            W_tgt = out_size(2);

            Ry = grid_sz(1);
            Rx = grid_sz(2);

            % Target-to-source displacements at control points
            disp_x = reshape(pts_src(:, 1) - pts_tgt(:, 1), Ry, Rx);
            disp_y = reshape(pts_src(:, 2) - pts_tgt(:, 2), Ry, Rx);

            % Grid cell sizes
            Delta_x = (W_tgt - 1) / (Rx - 1);
            Delta_y = (H_tgt - 1) / (Ry - 1);

            % Coordinate grid of target pixels
            [X_tgt, Y_tgt] = meshgrid(1:W_tgt, 1:H_tgt);

            % Map target pixel coordinates to grid units
            px = (X_tgt - 1) / Delta_x + 1;
            py = (Y_tgt - 1) / Delta_y + 1;

            ix = floor(px);
            iy = floor(py);

            u = px - ix;
            v = py - iy;

            % Cubic B-Spline 1D basis functions
            B0_u = (1 - u).^3 / 6;
            B1_u = (3*u.^3 - 6*u.^2 + 4) / 6;
            B2_u = (-3*u.^3 + 3*u.^2 + 3*u + 1) / 6;
            B3_u = u.^3 / 6;

            B0_v = (1 - v).^3 / 6;
            B1_v = (3*v.^3 - 6*v.^2 + 4) / 6;
            B2_v = (-3*v.^3 + 3*v.^2 + 3*v + 1) / 6;
            B3_v = v.^3 / 6;

            % Accumulate dense displacement fields
            Dx = zeros(H_tgt, W_tgt);
            Dy = zeros(H_tgt, W_tgt);

            % Tensor-product accumulation over the 4x4 neighbor control points
            for m = 0:3
                for n = 0:3
                    jx = ix + m - 1;
                    jy = iy + n - 1;

                    % Clamp indices to handle grid boundaries cleanly
                    jx_c = max(1, min(Rx, jx));
                    jy_c = max(1, min(Ry, jy));

                    % Linear index inside grid arrays
                    cp_idx = jy_c + (jx_c - 1) * Ry;

                    % Displacements at the neighbor control point
                    val_x = disp_x(cp_idx);
                    val_y = disp_y(cp_idx);

                    % 1D basis values
                    switch m
                        case 0, w_u = B0_u;
                        case 1, w_u = B1_u;
                        case 2, w_u = B2_u;
                        case 3, w_u = B3_u;
                    end
                    switch n
                        case 0, w_v = B0_v;
                        case 1, w_v = B1_v;
                        case 2, w_v = B2_v;
                        case 3, w_v = B3_v;
                    end

                    weight = w_u .* w_v;

                    Dx = Dx + weight .* val_x;
                    Dy = Dy + weight .* val_y;
                end
            end

            % Map coordinates back to source space
            u_src = X_tgt + Dx;
            v_src = Y_tgt + Dy;

            % Backward Bilinear Interpolation
            u0 = floor(u_src);
            v0 = floor(v_src);
            u1 = u0 + 1;
            v1 = v0 + 1;

            u0_c = max(1, min(W_src, u0));
            u1_c = max(1, min(W_src, u1));
            v0_c = max(1, min(H_src, v0));
            v1_c = max(1, min(H_src, v1));

            du = u_src - u0;
            dv = v_src - v0;

            w00 = (1 - du) .* (1 - dv);
            w01 = du .* (1 - dv);
            w10 = (1 - du) .* dv;
            w11 = du .* dv;

            idx00 = v0_c + (u0_c - 1) * H_src;
            idx01 = v0_c + (u1_c - 1) * H_src;
            idx10 = v1_c + (u0_c - 1) * H_src;
            idx11 = v1_c + (u1_c - 1) * H_src;

            I_warped = zeros(H_tgt, W_tgt, C);
            for c = 1:C
                Ic = I_dbl(:, :, c);
                val = w00 .* Ic(idx00) + w01 .* Ic(idx01) + w10 .* Ic(idx10) + w11 .* Ic(idx11);
                I_warped(:, :, c) = val;
            end

            % Cast back to original type if uint8
            if isa(I_src, 'uint8')
                I_warped = uint8(max(0, min(255, round(I_warped))));
            end
        end
    end
end
