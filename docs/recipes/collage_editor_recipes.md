# 2D Image Collage Editor Recipes

## Programmatic Collage Composition Loop
This recipe demonstrates how to manually initialize a canvas, define image layers, compute their homographies, and composite them onto a single output image without opening the GUI.

```matlab
% Set up paths
cd('c:\Ocean Library\Matlab code practice');
addpath(genpath(pwd));

% Initialize blank 800 x 600 canvas
I_canvas = repmat(reshape([0.10, 0.12, 0.16], 1, 1, 3), 600, 800);

% Load a source image
I_src = im2double(imread('Data/images/ocean_img.jpg'));

% Crop a region (200x200)
cropped = I_src(50:249, 50:249, :);
[h, w, ~] = size(cropped);

% Define base corners of crop
base_pts = [0, 0; w, 0; w, h; 0, h];

% Define destination coordinates on the canvas (Perspective Warp effect)
dst_pts = [150, 100;  350, 150;  300, 400;  100, 350];

% Solve homography H using Direct Linear Transform (DLT)
A = zeros(8, 8);
B = zeros(8, 1);
for i = 1:4
    x = base_pts(i,1); y = base_pts(i,2);
    u = dst_pts(i,1);  v = dst_pts(i,2);
    A(2*i-1, :) = [x, y, 1, 0, 0, 0, -u*x, -u*y];
    A(2*i, :)   = [0, 0, 0, x, y, 1, -v*x, -v*y];
    B(2*i-1) = u;
    B(2*i) = v;
end
h_sol = A \ B;
H = eye(3);
H(1, 1) = h_sol(1); H(1, 2) = h_sol(2); H(1, 3) = h_sol(3);
H(2, 1) = h_sol(4); H(2, 2) = h_sol(5); H(2, 3) = h_sol(6);
H(3, 1) = h_sol(7); H(3, 2) = h_sol(8); H(3, 3) = 1.0;

% Warp and overlay onto canvas
layer.img = cropped;
layer.base_corners = base_pts;
layer.H = H;

% Bounding box limits in destination canvas
pts_hom = [base_pts, ones(4, 1)]';
warped = H * pts_hom;
w_val = warped(3, :);
w_val(abs(w_val) < 1e-12) = 1e-12;
cx = warped(1, :) ./ w_val;
cy = warped(2, :) ./ w_val;

min_x = max(1, floor(min(cx)));
max_x = min(800, ceil(max(cx)));
min_y = max(1, floor(min(cy)));
max_y = min(600, ceil(max(cy)));

[X_grid, Y_grid] = meshgrid(min_x:max_x, min_y:max_y);
N_pts = numel(X_grid);
pts_dst = [X_grid(:), Y_grid(:), ones(N_pts, 1)]';

% Backward mapping
H_inv = inv(H);
pts_src = H_inv * pts_dst;
ws = pts_src(3, :);
ws(abs(ws) < 1e-12) = 1e-12;
src_x = pts_src(1, :) ./ ws;
src_y = pts_src(2, :) ./ ws;

inside = (src_x >= 0 & src_x <= w-1 & src_y >= 0 & src_y <= h-1);
if any(inside)
    src_x_1 = src_x(inside) + 1;
    src_y_1 = src_y(inside) + 1;
    x_in = X_grid(inside);
    y_in = Y_grid(inside);
    
    % Bilinear sampling weights
    x0 = floor(src_x_1);
    y0 = floor(src_y_1);
    x1 = x0 + 1;
    y1 = y0 + 1;
    
    x0_c = max(1, min(w, x0));
    x1_c = max(1, min(w, x1));
    y0_c = max(1, min(h, y0));
    y1_c = max(1, min(h, y1));
    
    dx = src_x_1 - x0;
    dy = src_y_1 - y0;
    
    w00 = (1 - dx) .* (1 - dy);
    w01 = dx .* (1 - dy);
    w10 = (1 - dx) .* dy;
    w11 = dx .* dy;
    
    idx00 = y0_c + (x0_c - 1) * h;
    idx01 = y0_c + (x1_c - 1) * h;
    idx10 = y1_c + (x0_c - 1) * h;
    idx11 = y1_c + (x1_c - 1) * h;
    
    canvas_idx = y_in + (x_in - 1) * 600;
    for c = 1:3
        Ic = cropped(:, :, c);
        val = w00 .* Ic(idx00) + w01 .* Ic(idx01) + w10 .* Ic(idx10) + w11 .* Ic(idx11);
        I_canvas(canvas_idx + (c - 1) * 600 * 800) = val;
    end
end

% Plot result
figure;
imshow(I_canvas);
title('Programmatic Composite');
```
