classdef BeierNeelyWarp
    % BEIERNEELYWARP  Feature-based image warping using line correspondences.
    %
    %   BeierNeelyWarp implements the Beier-Neely morphing algorithm.
    %   Geometric transformations are defined by corresponding directed line
    %   segments in the source and target images.
    %
    %   Reference:
    %     Beier, T., & Neely, S. (1992). Feature-based image metamorphosis.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-30

    methods (Static)
        function I_warped = warp(I_src, pts_tgt_P, pts_tgt_Q, pts_src_P, pts_src_Q, options)
            % WARP  Warp image using line correspondences.
            %
            %   I_warped = warp(I_src, pts_tgt_P, pts_tgt_Q, pts_src_P, pts_src_Q, options)
            %
            %   Inputs:
            %     I_src     - Source image (H x W x C)
            %     pts_tgt_P - L x 2 matrix of target line endpoints P
            %     pts_tgt_Q - L x 2 matrix of target line endpoints Q
            %     pts_src_P - L x 2 matrix of source line endpoints P'
            %     pts_src_Q - L x 2 matrix of source line endpoints Q'
            %     options   - Name-value options:
            %                   a: length weight parameter (default 0.5)
            %                   b: smoothing parameter in denominator (default 1.0)
            %                   p: distance decay parameter (default 1.0)

            arguments
                I_src {mustBeNumeric}
                pts_tgt_P (:, 2) double
                pts_tgt_Q (:, 2) double
                pts_src_P (:, 2) double
                pts_src_Q (:, 2) double
                options.a (1,1) double = 0.5
                options.b (1,1) double = 1.0
                options.p (1,1) double = 1.0
            end

            L = size(pts_tgt_P, 1);
            if L < 1
                error('BeierNeelyWarp:NoLines', 'At least one directed line segment is required.');
            end

            % Check dimension parity of line control structures
            if ~isequal(size(pts_tgt_Q), [L, 2]) || ~isequal(size(pts_src_P), [L, 2]) || ~isequal(size(pts_src_Q), [L, 2])
                error('BeierNeelyWarp:DimensionMismatch', 'All line endpoint matrices must have size L x 2.');
            end

            I_dbl = double(I_src);
            [H, W, C] = size(I_dbl, 1:3);
            N_px = H * W;

            % 1. Construct target pixel coordinate grid
            [X_grid, Y_grid] = meshgrid(1:W, 1:H);
            X = [X_grid(:), Y_grid(:)]; % N_px x 2

            sum_Dx = zeros(N_px, 1);
            sum_Dy = zeros(N_px, 1);
            sum_w = zeros(N_px, 1);

            a = options.a;
            b = options.b;
            p = options.p;

            % 2. Vectorized calculation for each control line
            for i = 1:L
                P = pts_tgt_P(i, :);
                Q = pts_tgt_Q(i, :);
                P_prime = pts_src_P(i, :);
                Q_prime = pts_src_Q(i, :);

                % Line vectors
                v_PQ = Q - P;
                len2_PQ = sum(v_PQ.^2);
                len_PQ = sqrt(len2_PQ);
                if len_PQ < 1e-9, continue; end
                v_PQ_perp = [-v_PQ(2), v_PQ(1)]; % 90 deg rotation

                v_src_PQ = Q_prime - P_prime;
                len_src_PQ = sqrt(sum(v_src_PQ.^2));
                if len_src_PQ < 1e-9, continue; end
                v_src_PQ_perp = [-v_src_PQ(2), v_src_PQ(1)];

                % Projection offsets (X - P)
                XP = X - P;

                % Compute u (projection parameter along line segment)
                u = (XP(:, 1) * v_PQ(1) + XP(:, 2) * v_PQ(2)) / len2_PQ;

                % Compute v (perpendicular distance parameter)
                v = (XP(:, 1) * v_PQ_perp(1) + XP(:, 2) * v_PQ_perp(2)) / len_PQ;

                % Compute mapped coordinate X_prime in source space
                X_prime_x = P_prime(1) + u * v_src_PQ(1) + (v * v_src_PQ_perp(1)) / len_src_PQ;
                X_prime_y = P_prime(2) + u * v_src_PQ(2) + (v * v_src_PQ_perp(2)) / len_src_PQ;

                % Compute shortest distance d of X to segment PQ
                d = zeros(N_px, 1);

                % Case 1: u < 0
                idx1 = u < 0;
                d(idx1) = sqrt(XP(idx1, 1).^2 + XP(idx1, 2).^2);

                % Case 2: u > 1
                idx2 = u > 1;
                XQ = X - Q;
                d(idx2) = sqrt(XQ(idx2, 1).^2 + XQ(idx2, 2).^2);

                % Case 3: 0 <= u <= 1
                idx3 = ~idx1 & ~idx2;
                d(idx3) = abs(v(idx3));

                % Compute line influence weight
                w = ((len_PQ^a) ./ (b + d)).^p;

                % Accumulate weighted displacements (D_i = X'_i - X)
                sum_Dx = sum_Dx + w .* (X_prime_x - X(:, 1));
                sum_Dy = sum_Dy + w .* (X_prime_y - X(:, 2));
                sum_w = sum_w + w;
            end

            % 3. Apply normalized weighted displacements
            u_src = X(:, 1) + sum_Dx ./ max(sum_w, 1e-12);
            v_src = X(:, 2) + sum_Dy ./ max(sum_w, 1e-12);

            % 4. Backward Bilinear Interpolation
            u0 = floor(u_src);
            v0 = floor(v_src);
            u1 = u0 + 1;
            v1 = v0 + 1;

            u0_c = max(1, min(W, u0));
            u1_c = max(1, min(W, u1));
            v0_c = max(1, min(H, v0));
            v1_c = max(1, min(H, v1));

            du = u_src - u0;
            dv = v_src - v0;

            w00 = (1 - du) .* (1 - dv);
            w01 = du .* (1 - dv);
            w10 = (1 - du) .* dv;
            w11 = du .* dv;

            idx00 = v0_c + (u0_c - 1) * H;
            idx01 = v0_c + (u1_c - 1) * H;
            idx10 = v1_c + (u0_c - 1) * H;
            idx11 = v1_c + (u1_c - 1) * H;

            I_warped = zeros(H, W, C);
            for c = 1:C
                Ic = double(I_src(:, :, c));
                val = w00 .* Ic(idx00) + w01 .* Ic(idx01) + w10 .* Ic(idx10) + w11 .* Ic(idx11);
                I_warped(:, :, c) = reshape(val, H, W);
            end

            % Cast back to original type if uint8
            if isa(I_src, 'uint8')
                I_warped = uint8(max(0, min(255, round(I_warped))));
            end
        end
    end
end
