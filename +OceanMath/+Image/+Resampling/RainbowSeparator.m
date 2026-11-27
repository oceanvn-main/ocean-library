classdef RainbowSeparator
    % RAINBOWSEPARATOR  Computational engine for circle fitting and additive separation.
    %
    %   Performs linear least-squares circle fitting, unrolls radial image 
    %   profiles to polar coordinates, separates additive signals from smooth
    %   background polynomial trends, and reconstructs 2D spectral fields.
    %
    %   Author: Antigravity Code Assistant
    %   Date: 2026-07-16
    
    methods (Static)
        function [xc, yc, R] = fitCircle(pts)
            % FITCIRCLE Fits a circle (x-xc)^2 + (y-yc)^2 = R^2 to N points.
            % pts is a [2 x N] or [N x 2] matrix of coordinates.
            if size(pts, 1) ~= 2
                pts = pts';
            end
            N = size(pts, 2);
            if N < 3
                error('RainbowSeparator:InsufficientPoints', 'At least 3 points are required to fit a circle.');
            end
            
            % Solve linear least-squares: 2*x*xc + 2*y*yc + (R^2 - xc^2 - yc^2) = x^2 + y^2
            A = [pts(1,:)', pts(2,:)', ones(N, 1)];
            f = (pts(1,:).^2 + pts(2,:).^2)';
            w = A \ f;
            
            xc = w(1) / 2;
            yc = w(2) / 2;
            R = sqrt(w(3) + xc^2 + yc^2);
        end
        
        function I_polar = unrollPolar(I_lin, xc, yc, r_range, theta_range, num_r, num_theta)
            % UNROLLPOLAR Unrolls Cartesian image into a polar coordinate grid.
            % I_lin is a [H x W x C] double image in linear intensity space.
            [H, W, C] = size(I_lin);
            
            r_grid = linspace(r_range(1), r_range(2), num_r);
            theta_grid = linspace(theta_range(1), theta_range(2), num_theta);
            
            [R_mat, T_mat] = ndgrid(r_grid, theta_grid);
            
            % Map polar coordinates to Cartesian
            X_grid = xc + R_mat .* cos(T_mat);
            Y_grid = yc + R_mat .* sin(T_mat);
            
            % Perform vectorized bilinear interpolation
            x1 = floor(X_grid); x2 = x1 + 1;
            y1 = floor(Y_grid); y2 = y1 + 1;
            
            % Clamp to avoid out of bounds
            x1 = max(1, min(W, x1)); x2 = max(1, min(W, x2));
            y1 = max(1, min(H, y1)); y2 = max(1, min(H, y2));
            
            wa = (x2 - X_grid) .* (y2 - Y_grid);
            wb = (X_grid - x1) .* (y2 - Y_grid);
            wc = (x2 - X_grid) .* (Y_grid - y1);
            wd = (X_grid - x1) .* (Y_grid - y1);
            
            I_polar = zeros(num_r, num_theta, C);
            for ch = 1:C
                I_ch = I_lin(:, :, ch);
                idx_a = y1 + (x1 - 1) * H;
                idx_b = y1 + (x2 - 1) * H;
                idx_c = y2 + (x1 - 1) * H;
                idx_d = y2 + (x2 - 1) * H;
                
                I_polar(:, :, ch) = wa .* I_ch(idx_a) + wb .* I_ch(idx_b) + ...
                                    wc .* I_ch(idx_c) + wd .* I_ch(idx_d);
            end
        end
        
        function [R_profile, B_profile] = separateProfile(I_polar, r_grid, r_rainbow_range, poly_order)
            % SEPARATEPROFILE Teases apart rainbow signal from smooth background trend.
            % I_polar is [num_r x num_theta x C] polar unrolled image.
            % r_grid is [1 x num_r] radial grid.
            % r_rainbow_range is [r_start, r_end] radial span of the rainbow.
            [num_r, ~, C] = size(I_polar);
            
            % Compute average color profile along angular dimension
            I_avg = squeeze(mean(I_polar, 2));
            if C == 1
                I_avg = I_avg(:); % Keep as column vector if grayscale
            end
            
            % Define mask for background (outside rainbow zone)
            bg_mask = (r_grid < r_rainbow_range(1)) | (r_grid > r_rainbow_range(2));
            
            R_profile = zeros(num_r, C);
            B_profile = zeros(num_r, C);
            
            for ch = 1:C
                % Fit polynomial to smooth background trend outside the rainbow
                p = polyfit(r_grid(bg_mask)', I_avg(bg_mask, ch), poly_order);
                bg_val = polyval(p, r_grid');
                
                B_profile(:, ch) = bg_val;
                % Subtract background to isolate the additive rainbow signal
                R_profile(:, ch) = max(0, I_avg(:, ch) - bg_val);
            end
        end
        
        function R_2D = reconstruct2D(R_profile, r_grid, H, W, xc, yc, theta_range)
            % RECONSTRUCT2D Re-maps 1D radial profile to 2D image coordinates.
            [~, C] = size(R_profile);
            
            [X, Y] = meshgrid(1:W, 1:H);
            Dist = sqrt((X - xc).^2 + (Y - yc).^2);
            Theta = atan2(Y - yc, X - xc);
            
            % Wrap angles to be centered around the range mid-point
            tc = mean(theta_range);
            Theta = Theta - 2*pi * round((Theta - tc) / (2*pi));
            
            % Create angular mask
            angle_mask = (Theta >= theta_range(1)) & (Theta <= theta_range(2));
            
            % Remap profile
            R_2D = zeros(H, W, C);
            for ch = 1:C
                R_2D(:, :, ch) = interp1(r_grid, R_profile(:, ch), Dist, 'linear', 0.0) .* angle_mask;
            end
        end
    end
end
