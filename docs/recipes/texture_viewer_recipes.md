# 3D Texture-Mapped Software Rasterizer Recipes

## Programmatic 3D Perspective-Correct Triangle Rasterization
This recipe projects a 3D unit cube face and rasterizes a texture onto it. It triangulates the quadrilateral and interpolates the inverse depth ($1/Z$) and projective coordinates ($u/Z, v/Z$) per pixel, performing the perspective divide at each fragment to eliminate affine warping creases.

```matlab
% Set up paths
cd('c:\Ocean Library\Matlab code practice');
addpath(genpath(pwd));

% Generate/load texture image
I_tex = im2double(imread('Data/images/ocean_img.jpg'));
[H_tex, W_tex, ~] = size(I_tex);

% Define 3D front face vertices [x, y, z, 1] in world space
face_3d = [
    -1, -1, -1, 1;
     1, -1, -1, 1;
     1,  1, -1, 1;
    -1,  1, -1, 1
]';

% Define camera look-at position
camera_pos = [2.0; 1.5; -5.0];
target_pos = [0.0; 0.0; 0.0];
up_vector  = [0.0; 1.0; 0.0];

% Derivation of look-at view matrix (+Z as camera forward)
z_axis = target_pos - camera_pos;
z_axis = z_axis / norm(z_axis);
x_axis = cross(up_vector, z_axis);
x_axis = x_axis / norm(x_axis);
y_axis = cross(z_axis, x_axis);
R_cam = [x_axis, y_axis, z_axis];

M_view = eye(4);
M_view(1:3, 1:3) = R_cam';
M_view(1:3, 4)   = -R_cam' * camera_pos;

% Transform vertices to camera space
pts_cam = M_view * face_3d;

% Project vertices to screen coordinates (800x600 canvas)
f = 500.0;
cx = 400; cy = 300;
pts_screen = [f * pts_cam(1,:) ./ pts_cam(3,:) + cx;
             -f * pts_cam(2,:) ./ pts_cam(3,:) + cy]';

% Texture coordinates for the quad vertices
u_tex = [1, W_tex, W_tex, 1];
v_tex = [H_tex, H_tex, 1, 1];

% Initialize blank 800 x 600 canvas
I_canvas = repmat(reshape([0.08, 0.09, 0.11], 1, 1, 3), 600, 800);

% Triangulate the 4-vertex quad into two triangles using a fan
for t = 1:2
    idxA = 1;
    idxB = t + 1;
    idxC = t + 2;
    
    p1 = pts_screen(idxA, :);
    p2 = pts_screen(idxB, :);
    p3 = pts_screen(idxC, :);
    
    Z1 = max(pts_cam(3, idxA), 1e-6);
    Z2 = max(pts_cam(3, idxB), 1e-6);
    Z3 = max(pts_cam(3, idxC), 1e-6);
    
    ut1 = u_tex(idxA); vt1 = v_tex(idxA);
    ut2 = u_tex(idxB); vt2 = v_tex(idxB);
    ut3 = u_tex(idxC); vt3 = v_tex(idxC);
    
    % Linear projective values for interpolation
    w1 = 1 / Z1;  u1_z = ut1 / Z1;  v1_z = vt1 / Z1;
    w2 = 1 / Z2;  u2_z = ut2 / Z2;  v2_z = vt2 / Z2;
    w3 = 1 / Z3;  u3_z = ut3 / Z3;  v3_z = vt3 / Z3;
    
    % Bounding box limits in destination screen
    min_x = max(1,   floor(min([p1(1), p2(1), p3(1)])));
    max_x = min(800, ceil( max([p1(1), p2(1), p3(1)])));
    min_y = max(1,   floor(min([p1(2), p2(2), p3(2)])));
    max_y = min(600, ceil( max([p1(2), p2(2), p3(2)])));
    
    if min_x > max_x || min_y > max_y, continue; end
    
    [X_grid, Y_grid] = meshgrid(min_x:max_x, min_y:max_y);
    
    x1 = p1(1); y1 = p1(2);
    x2 = p2(1); y2 = p2(2);
    x3 = p3(1); y3 = p3(2);
    
    D = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1);
    if abs(D) < 1e-6, continue; end
    
    beta  = ((X_grid - x1)*(y3 - y1) - (x3 - x1)*(Y_grid - y1)) / D;
    gamma = ((x2 - x1)*(Y_grid - y1) - (X_grid - x1)*(y2 - y1)) / D;
    alpha = 1 - beta - gamma;
    
    inside = (alpha >= 0) & (beta >= 0) & (gamma >= 0);
    if ~any(inside(:)), continue; end
    
    x_canvas = X_grid(inside);
    y_canvas = Y_grid(inside);
    
    a_v = alpha(inside);
    b_v = beta(inside);
    g_v = gamma(inside);
    
    % Interpolate w = 1/Z and projective attributes
    w_interp   = a_v * w1 + b_v * w2 + g_v * w3;
    u_z_interp = a_v * u1_z + b_v * u2_z + g_v * u3_z;
    v_z_interp = a_v * v1_z + b_v * v2_z + g_v * v3_z;
    
    % Final perspective divide
    w_interp(abs(w_interp) < 1e-12) = 1e-12;
    src_x = u_z_interp ./ w_interp;
    src_y = v_z_interp ./ w_interp;
    
    % Bilinear sampling
    valid_tex = (src_x >= 1 & src_x <= W_tex & src_y >= 1 & src_y <= H_tex);
    if any(valid_tex)
        src_x_v = src_x(valid_tex);
        src_y_v = src_y(valid_tex);
        xc = x_canvas(valid_tex);
        yc = y_canvas(valid_tex);
        
        tx_f = floor(src_x_v); ty_f = floor(src_y_v);
        tx_c = tx_f + 1;       ty_c = ty_f + 1;
        
        tx_f_c = max(1, min(W_tex, tx_f));
        tx_c_c = max(1, min(W_tex, tx_c));
        ty_f_c = max(1, min(H_tex, ty_f));
        ty_c_c = max(1, min(H_tex, ty_c));
        
        dx = src_x_v - tx_f;
        dy = src_y_v - ty_f;
        
        w00 = (1 - dx) .* (1 - dy);
        w01 = dx .* (1 - dy);
        w10 = (1 - dx) .* dy;
        w11 = dx .* dy;
        
        idx00 = ty_f_c + (tx_f_c - 1) * H_tex;
        idx01 = ty_f_c + (tx_c_c - 1) * H_tex;
        idx10 = ty_c_c + (tx_f_c - 1) * H_tex;
        idx11 = ty_c_c + (tx_c_c - 1) * H_tex;
        
        canvas_idx = yc + (xc - 1) * 600;
        for c = 1:3
            Ic = I_tex(:, :, c);
            val = w00 .* Ic(idx00) + w01 .* Ic(idx01) + w10 .* Ic(idx10) + w11 .* Ic(idx11);
            I_canvas(canvas_idx + (c - 1) * 600 * 800) = val;
        end
    end
end

% Display
figure;
imshow(I_canvas);
title('Perspective-Correct Rasterized Face');
```
