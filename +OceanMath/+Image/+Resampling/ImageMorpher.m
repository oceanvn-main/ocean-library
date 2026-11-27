classdef ImageMorpher
    % IMAGEMORPHER  Feature-based cross-dissolve image morphing engine (Ex 3.25).
    %
    %   Implements the 4-step morphing pipeline (Szeliski Sec 3.6.3):
    %     Step 1 - Build Delaunay triangulation from control points
    %     Step 2 - Interpolate control points -> in-between "skeleton" at t
    %     Step 3 - Inverse-map BOTH images toward the intermediate skeleton
    %     Step 4 - Cross-dissolve: (1-t)*I_A_warp + t*I_B_warp
    %
    %   All warp engines from Ex 3.23 are supported as backends.
    %
    %   Usage:
    %     % Point-based (grid or sparse landmarks), Piecewise Affine backend:
    %     I_morph = OceanMath.Image.Resampling.ImageMorpher.morphFrame( ...
    %         I_A, I_B, pts_A, pts_B, 0.5, 'WarpMode', 'PiecewiseAffine');
    %
    %     % Generate 9-frame animation:
    %     frames = OceanMath.Image.Resampling.ImageMorpher.morphSequence( ...
    %         I_A, I_B, pts_A, pts_B, 9, 'WarpMode', 'ThinPlateSpline');
    %
    %   Reference:
    %     Szeliski "Computer Vision: Algorithms and Applications" Sec 3.6.3
    %     Beier, T. & Neely, S. (1992). Feature-based image metamorphosis.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-16

    methods (Static)

        function I_morph = morphFrame(I_A, I_B, pts_A, pts_B, t, options)
            % MORPHFRAME  Compute one cross-dissolve morph frame at parameter t.
            %
            %   I_morph = morphFrame(I_A, I_B, pts_A, pts_B, t)
            %   I_morph = morphFrame(..., 'WarpMode', 'ThinPlateSpline')
            %   I_morph = morphFrame(..., 'WarpMode', 'BSplineGrid', 'GridRes', [4, 4])
            %
            %   Inputs:
            %     I_A, I_B    - Source images (same H x W x C, double [0,1] or uint8)
            %     pts_A       - Source control points (N x 2, [x, y])
            %     pts_B       - Target control points (N x 2, [x, y])
            %     t           - Morph parameter in [0, 1]  (0 = I_A, 1 = I_B)
            %
            %   Name-value options:
            %     WarpMode    - 'PiecewiseAffine' (default) | 'ThinPlateSpline' |
            %                   'BSplineGrid'
            %     GridRes     - [Ry, Rx] grid size for BSplineGrid (default [4, 4])
            %
            %   Output:
            %     I_morph     - Morphed image (double [0,1], same H x W x C)

            arguments
                I_A          {mustBeNumeric}
                I_B          {mustBeNumeric}
                pts_A        (:, 2) double
                pts_B        (:, 2) double
                t            (1, 1) double
                options.WarpMode (1, 1) string = "PiecewiseAffine"
                options.GridRes  (1, 2) double = [4, 4]
            end

            t = max(0, min(1, t));  % clamp

            % Convert to double [0,1]
            I_A_d = OceanMath.Image.Resampling.ImageMorpher.toDouble_(I_A);
            I_B_d = OceanMath.Image.Resampling.ImageMorpher.toDouble_(I_B);

            % --- Step 2: Interpolate control points -> intermediate skeleton ---
            pts_mid = (1 - t) * pts_A + t * pts_B;

            % --- Step 3: Inverse-warp both images to the intermediate shape ---
            mode = options.WarpMode;
            switch mode
                case "PiecewiseAffine"
                    % Triangulate on the source grid (most stable connectivity)
                    tri = delaunay(pts_A(:, 1), pts_A(:, 2));
                    % Warp I_A: output shape = pts_mid, sampling source = pts_A
                    I_A_warp = OceanMath.Image.Resampling.PiecewiseAffineWarp.warp( ...
                        I_A_d, pts_A, pts_mid, tri);
                    % Warp I_B: output shape = pts_mid, sampling source = pts_B
                    I_B_warp = OceanMath.Image.Resampling.PiecewiseAffineWarp.warp( ...
                        I_B_d, pts_B, pts_mid, tri);

                case "ThinPlateSpline"
                    tps_A = OceanMath.Image.Resampling.ThinPlateSpline( ...
                        pts_mid, pts_A, 'thinplate');
                    I_A_warp = tps_A.warpImage(I_A_d);
                    tps_B = OceanMath.Image.Resampling.ThinPlateSpline( ...
                        pts_mid, pts_B, 'thinplate');
                    I_B_warp = tps_B.warpImage(I_B_d);

                case "BSplineGrid"
                    res = options.GridRes;
                    I_A_warp = OceanMath.Image.Resampling.BSplineWarp.warp( ...
                        I_A_d, pts_A, pts_mid, res);
                    I_B_warp = OceanMath.Image.Resampling.BSplineWarp.warp( ...
                        I_B_d, pts_B, pts_mid, res);

                otherwise
                    error('ImageMorpher:UnknownMode', ...
                        'Unknown WarpMode "%s". Choose: PiecewiseAffine, ThinPlateSpline, BSplineGrid.', mode);
            end

            % --- Step 4: Cross-dissolve ---
            I_morph = (1 - t) * double(I_A_warp) + t * double(I_B_warp);
            I_morph = max(0, min(1, I_morph));
        end

        % ------------------------------------------------------------------

        function frames = morphSequence(I_A, I_B, pts_A, pts_B, N, options)
            % MORPHSEQUENCE  Generate N morph frames at t = linspace(0,1,N).
            %
            %   frames = morphSequence(I_A, I_B, pts_A, pts_B, N)
            %   frames = morphSequence(..., 'WarpMode', 'ThinPlateSpline')
            %
            %   Output:
            %     frames - 1 x N cell array of double images in [0,1]

            arguments
                I_A          {mustBeNumeric}
                I_B          {mustBeNumeric}
                pts_A        (:, 2) double
                pts_B        (:, 2) double
                N            (1, 1) double {mustBePositive, mustBeInteger}
                options.WarpMode (1, 1) string = "PiecewiseAffine"
                options.GridRes  (1, 2) double = [4, 4]
            end

            t_vals = linspace(0, 1, N);
            frames = cell(1, N);
            for i = 1:N
                frames{i} = OceanMath.Image.Resampling.ImageMorpher.morphFrame( ...
                    I_A, I_B, pts_A, pts_B, t_vals(i), ...
                    'WarpMode', options.WarpMode, ...
                    'GridRes',  options.GridRes);
            end
        end

        % ------------------------------------------------------------------

        function [pts_A, pts_B] = rotationControlPoints(img_sz, theta_deg, res)
            % ROTATIONCONTROLPOINTS  Build (pts_A, pts_B) for a rotation morph test.
            %
            %   [pts_A, pts_B] = rotationControlPoints([H, W], theta_deg, res)
            %
            %   Inputs:
            %     img_sz    - [H, W] image size
            %     theta_deg - Rotation angle in degrees (counter-clockwise)
            %     res       - Grid resolution (points per axis, default = 8)
            %
            %   Notes:
            %     Linear interpolation of Cartesian positions does NOT follow
            %     circular arcs (see Sec 3.6.3 discussion). Pixels travel along
            %     chords rather than arcs, producing an "implosion" artifact at
            %     t = 0.5 for large angles.

            arguments
                img_sz    (1, 2) double
                theta_deg (1, 1) double
                res       (1, 1) double {mustBeInteger, mustBePositive} = 8
            end

            H  = img_sz(1); W = img_sz(2);
            cx = (W + 1) / 2;
            cy = (H + 1) / 2;

            x_grid = linspace(1, W, res);
            y_grid = linspace(1, H, res);
            [X, Y] = meshgrid(x_grid, y_grid);
            pts_A = [X(:), Y(:)];

            theta  = deg2rad(theta_deg);
            dx     = pts_A(:, 1) - cx;
            dy     = pts_A(:, 2) - cy;
            pts_B  = [cx + dx * cos(theta) - dy * sin(theta), ...
                      cy + dx * sin(theta) + dy * cos(theta)];
        end

        % ------------------------------------------------------------------

        function [pts_A, pts_B] = uniformGridPoints(img_sz, res)
            % UNIFORMGRIDPOINTS  Generate a uniform grid of correspondence pairs
            %                    (identical, to be manually offset by the user).
            %
            %   [pts_A, pts_B] = uniformGridPoints([H, W], res)
            %
            %   Returns a res x res grid of points. pts_A == pts_B initially.

            arguments
                img_sz (1, 2) double
                res    (1, 1) double {mustBeInteger, mustBePositive} = 4
            end

            H = img_sz(1); W = img_sz(2);
            x_grid = linspace(max(1, W * 0.05), min(W, W * 0.95), res);
            y_grid = linspace(max(1, H * 0.05), min(H, H * 0.95), res);
            [X, Y] = meshgrid(x_grid, y_grid);
            pts_A = [X(:), Y(:)];
            pts_B = pts_A;
        end

    end

    methods (Static, Access = private)
        function I_d = toDouble_(I)
            % Convert uint8 [0,255] or double [0,1] to double [0,1]
            I_d = double(I);
            if max(I_d(:)) > 1.5
                I_d = I_d / 255;
            end
        end
    end
end
