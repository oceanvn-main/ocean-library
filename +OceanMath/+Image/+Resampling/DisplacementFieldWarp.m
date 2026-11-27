classdef DisplacementFieldWarp
    % DISPLACEMENTFIELDWARP  Forward image warping using a displacement field.
    %
    %   DisplacementFieldWarp warps a source image forward using a coordinate
    %   displacement field. It utilizes bilinear splatting to distribute source
    %   pixel values to target coordinates, followed by normalization and
    %   neighborhood hole filling.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-06-30

    methods (Static)
        function I_warped = warp(I_src, Dx, Dy)
            % WARP  Forward warps an image using displacement fields.
            %
            %   I_warped = warp(I_src, Dx, Dy)
            %
            %   Inputs:
            %     I_src - Source image (H x W x C)
            %     Dx    - H x W matrix of horizontal displacements
            %     Dy    - H x W matrix of vertical displacements

            arguments
                I_src {mustBeNumeric}
                Dx (:, :) double
                Dy (:, :) double
            end

            I_dbl = double(I_src);
            [H, W, C] = size(I_dbl, 1:3);

            if ~isequal(size(Dx), [H, W]) || ~isequal(size(Dy), [H, W])
                error('DisplacementFieldWarp:DimensionMismatch', 'Displacement fields Dx and Dy must match the size of I_src.');
            end

            % 1. Initialize accumulators
            I_accum = zeros(H, W, C);
            W_accum = zeros(H, W);

            % 2. Bilinear splatting loop
            % We iterate over all source pixels and splat them to their target coordinates.
            for r = 1:H
                for c = 1:W
                    dx = Dx(r, c);
                    dy = Dy(r, c);

                    % Target coordinates
                    x = c + dx;
                    y = r + dy;

                    % Boundaries check
                    if x >= 1 && x <= W && y >= 1 && y <= H
                        x0 = floor(x);
                        y0 = floor(y);
                        x1 = x0 + 1;
                        y1 = y0 + 1;

                        wx = x - x0;
                        wy = y - y0;

                        % Bilinear weights
                        w00 = (1 - wx) * (1 - wy);
                        w01 = wx * (1 - wy);
                        w10 = (1 - wx) * wy;
                        w11 = wx * wy;

                        % Splat to the 4 nearest target locations
                        pixel_val = reshape(I_dbl(r, c, :), 1, 1, C);

                        % Splat (y0, x0)
                        if y0 >= 1 && y0 <= H && x0 >= 1 && x0 <= W
                            I_accum(y0, x0, :) = I_accum(y0, x0, :) + w00 * pixel_val;
                            W_accum(y0, x0) = W_accum(y0, x0) + w00;
                        end

                        % Splat (y0, x1)
                        if y0 >= 1 && y0 <= H && x1 >= 1 && x1 <= W
                            I_accum(y0, x1, :) = I_accum(y0, x1, :) + w01 * pixel_val;
                            W_accum(y0, x1) = W_accum(y0, x1) + w01;
                        end

                        % Splat (y1, x0)
                        if y1 >= 1 && y1 <= H && x0 >= 1 && x0 <= W
                            I_accum(y1, x0, :) = I_accum(y1, x0, :) + w10 * pixel_val;
                            W_accum(y1, x0) = W_accum(y1, x0) + w10;
                        end

                        % Splat (y1, x1)
                        if y1 >= 1 && y1 <= H && x1 >= 1 && x1 <= W
                            I_accum(y1, x1, :) = I_accum(y1, x1, :) + w11 * pixel_val;
                            W_accum(y1, x1) = W_accum(y1, x1) + w11;
                        end
                    end
                end
            end

            % 3. Normalization
            I_warped = I_accum ./ max(W_accum, 1e-12);

            % 4. Hole Filling (where weight is 0)
            holes = (W_accum < 1e-5);
            if any(holes(:))
                I_filled = I_warped;
                [hr, hc] = find(holes);
                for i = 1:length(hr)
                    r_idx = hr(i);
                    c_idx = hc(i);

                    % Query 3x3 neighborhood
                    r_min = max(1, r_idx - 1); r_max = min(H, r_idx + 1);
                    c_min = max(1, c_idx - 1); c_max = min(W, c_idx + 1);

                    neigh_weights = W_accum(r_min:r_max, c_min:c_max);
                    neigh_colors = I_warped(r_min:r_max, c_min:c_max, :);

                    valid = (neigh_weights >= 1e-5);
                    if any(valid(:))
                        neigh_colors_2d = reshape(neigh_colors, [], C);
                        I_filled(r_idx, c_idx, :) = mean(neigh_colors_2d(valid(:), :), 1);
                    end
                end
                I_warped = I_filled;
            end

            % Cast back to original type if uint8
            if isa(I_src, 'uint8')
                I_warped = uint8(max(0, min(255, round(I_warped))));
            end
        end
    end
end
