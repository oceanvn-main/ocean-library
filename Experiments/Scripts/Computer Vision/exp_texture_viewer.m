% exp_texture_viewer.m
% =========================================================================
% Experiment: 3D Perspective-Correct Software Rasterizer (Szeliski Ex 3.27)
% =========================================================================
%
% PURPOSE:
%   Demonstrates that we ARE the GPU. The code manually implements every
%   step of the rendering pipeline that a 3D API (e.g., OpenGL) would
%   normally perform internally:
%
%     1. Define a 3D cube model in world space.
%     2. Transform vertices to camera space (view matrix).
%     3. Project 3D vertices onto a 2D pixel grid via perspective divide.
%     4. For each 2D pixel INSIDE a projected triangle, compute barycentric
%        coordinates and perform the perspective-correct 1/Z divide to
%        recover true (u, v) texture coordinates.
%     5. Fetch the color from a SYNTHETIC texture using bilinear sampling.
%     6. Write that color into a flat 2D canvas matrix (zero 3D plotting).
%     7. Display the flat 2D canvas with imshow().
%
% SYNTHETIC TEXTURE:
%   Uses a programmatically generated checkerboard + UV gradient grid.
%   This pattern is the gold standard for verifying perspective-correct
%   interpolation: straight grid lines that curve on back-facing or oblique
%   surfaces reveal affine (broken) interpolation; straight lines across
%   the diagonal seam prove perspective correctness.
%
% ASSUMPTIONS:
%   - Depth-sorting (Painter's algorithm) resolves occlusion.
%   - Bounding-box pixel iteration + barycentric test replaces scanline.
%   - No built-in 3D plotting commands are used anywhere.
%
% REPRODUCIBILITY: All inputs are deterministic. No random seeds needed.
%
% Author: Antigravity Code Assistant
% Date: 2026-07-16
% =========================================================================

clear; close all; clc;
addpath(genpath('c:\Ocean Library\Matlab code practice'));
cd('c:\Ocean Library\Matlab code practice');

fprintf('=== Ex 3.27: Pure Software 3D Perspective-Correct Rasterizer ===\n\n');

%% =========================================================================
% STEP 0: GENERATE SYNTHETIC TEXTURE
% =========================================================================
% Checkerboard pattern with UV gradient overlay.
% - The high-frequency grid makes perspective distortion immediately visible.
% - Correct implementation: straight lines on each face, NO diagonal kink.
TEX_SIZE = 512;
fprintf('[Step 0] Generating synthetic %dx%d checkerboard texture...\n', TEX_SIZE, TEX_SIZE);

% Normalized UV coordinates [0,1] for each pixel
[U_grid, V_grid] = meshgrid(linspace(0, 1, TEX_SIZE), linspace(0, 1, TEX_SIZE));

% Checkerboard: 8x8 tiles
n_tiles = 8;
checker = mod(floor(U_grid * n_tiles) + floor(V_grid * n_tiles), 2);

% Color channels:
%   R = U gradient (left=dark, right=bright)
%   G = V gradient (top=dark, bottom=bright)
%   B = 0.15 base (dark blue tint)
I_tex = zeros(TEX_SIZE, TEX_SIZE, 3);
I_tex(:, :, 1) = 0.05 + 0.75 * (checker .* U_grid);   % warm red in checker
I_tex(:, :, 2) = 0.05 + 0.60 * (checker .* V_grid);   % cool green gradient
I_tex(:, :, 3) = 0.12 + 0.45 * (1 - checker);          % blue in off-checker

% Add thin white grid lines every 1/8th of texture space for extra precision
grid_line_width = 0.012; % fraction of texture
mask_u = abs(mod(U_grid, 1/n_tiles) - 1/(2*n_tiles)) < grid_line_width/2;
mask_v = abs(mod(V_grid, 1/n_tiles) - 1/(2*n_tiles)) < grid_line_width/2;
grid_mask = mask_u | mask_v;
I_tex(:, :, 1) = I_tex(:, :, 1) .* (1 - grid_mask) + 0.95 * grid_mask;
I_tex(:, :, 2) = I_tex(:, :, 2) .* (1 - grid_mask) + 0.95 * grid_mask;
I_tex(:, :, 3) = I_tex(:, :, 3) .* (1 - grid_mask) + 0.95 * grid_mask;
I_tex = min(1, max(0, I_tex)); % clamp

[H_tex, W_tex, ~] = size(I_tex);
fprintf('   Texture size: %d x %d\n', H_tex, W_tex);

%% =========================================================================
% STEP 1: 3D CUBE MODEL DEFINITION
% =========================================================================
fprintf('[Step 1] Defining 3D cube model (8 vertices, 6 faces)...\n');

% 8 vertices of a unit cube in world space
% Vertex layout (0-indexed labels in comments, 1-indexed in MATLAB arrays):
%   0=(-1,-1,-1), 1=(+1,-1,-1), 2=(+1,+1,-1), 3=(-1,+1,-1)  <- front face (Z=-1)
%   4=(-1,-1,+1), 5=(+1,-1,+1), 6=(+1,+1,+1), 7=(-1,+1,+1)  <- back face  (Z=+1)
vertices = [
    -1.0, -1.0, -1.0; ...  % vertex 1
     1.0, -1.0, -1.0; ...  % vertex 2
     1.0,  1.0, -1.0; ...  % vertex 3
    -1.0,  1.0, -1.0; ...  % vertex 4
    -1.0, -1.0,  1.0; ...  % vertex 5
     1.0, -1.0,  1.0; ...  % vertex 6
     1.0,  1.0,  1.0; ...  % vertex 7
    -1.0,  1.0,  1.0       % vertex 8
];
pts_3d = [vertices, ones(8, 1)]';  % homogeneous 4D: [4 x 8]

% 6 quad faces (1-based vertex indices)
faces = {
    [1, 2, 3, 4], ...  % face 1: Z=-1 (front)
    [5, 6, 7, 8], ...  % face 2: Z=+1 (back)
    [1, 2, 6, 5], ...  % face 3: Y=-1 (bottom)
    [2, 3, 7, 6], ...  % face 4: X=+1 (right)
    [3, 4, 8, 7], ...  % face 5: Y=+1 (top)
    [4, 1, 5, 8]       % face 6: X=-1 (left)
};
n_faces = numel(faces);

% Per-face texture coordinates in normalized [u, v, w=1] space.
% Mapping: vertex 1 of face -> tex (0,0), vertex 2 -> (1,0), etc.
% The v-axis is flipped (v=1 at row 0, v=0 at row H) to match image space.
tex_coords = [
    0.0, 1.0, 1.0;   % vertex A of polygon -> texture top-left
    1.0, 1.0, 1.0;   % vertex B            -> texture top-right
    1.0, 0.0, 1.0;   % vertex C            -> texture bottom-right
    0.0, 0.0, 1.0    % vertex D            -> texture bottom-left
];

%% =========================================================================
% STEP 2: CAMERA SETUP & VIEW MATRIX
% =========================================================================
fprintf('[Step 2] Computing camera view matrix...\n');

% Camera positioning and target
camera_pos = [2.5; 2.5; -6.0];
target_pos = [0.0; 0.0; 0.0];
up_vector  = [0.0; 1.0; 0.0];

% Derivation of look-at axes matching $+Z$ as forward
z_axis = target_pos - camera_pos;
z_axis = z_axis / norm(z_axis);
x_axis = cross(up_vector, z_axis);
x_axis = x_axis / norm(x_axis);
y_axis = cross(z_axis, x_axis);

R_cam = [x_axis, y_axis, z_axis];

% View matrix: transforms world space to camera space
M_view = eye(4);
M_view(1:3, 1:3) = R_cam';
M_view(1:3, 4)   = -R_cam' * camera_pos;

% Transform all 8 vertices to camera space
pts_cam = M_view * pts_3d;  % [4 x 8]
fprintf('   Camera at world pos: [%.1f, %.1f, %.1f]\n', camera_pos(1), camera_pos(2), camera_pos(3));

%% =========================================================================
% STEP 3: PERSPECTIVE PROJECTION (3D -> 2D pixel grid)
% =========================================================================
fprintf('[Step 3] Projecting 3D vertices onto 2D pixel grid...\n');

canvas_w = 800;
canvas_h = 600;
cx = canvas_w / 2;   % principal point X
cy = canvas_h / 2;   % principal point Y
f  = 500.0;          % focal length (pixels)

pts_screen = zeros(2, 8);
in_front   = false(1, 8);
for k = 1:8
    p = pts_cam(:, k);
    if p(3) > 0.1    % only project vertices in front of the camera
        pts_screen(1, k) = f * p(1) / p(3) + cx;         % screen X
        pts_screen(2, k) = -f * p(2) / p(3) + cy;        % screen Y (Y flipped)
        in_front(k) = true;
    end
end

% Log projected screen positions
for k = 1:8
    fprintf('   Vertex %d: screen=(%.1f, %.1f), depth=%.3f\n', ...
        k, pts_screen(1,k), pts_screen(2,k), pts_cam(3,k));
end

%% =========================================================================
% STEP 4: PAINTER'S ALGORITHM DEPTH SORT
% =========================================================================
fprintf('[Step 4] Depth-sorting faces (Painter''s algorithm)...\n');

face_depths = zeros(n_faces, 1);
valid_faces = false(n_faces, 1);
for k = 1:n_faces
    v_idx = faces{k};
    if all(in_front(v_idx))
        valid_faces(k)  = true;
        face_depths(k)  = mean(pts_cam(3, v_idx));
    end
end

valid_idx        = find(valid_faces);
[~, sort_idx]    = sort(face_depths(valid_idx), 'descend');
sorted_face_idx  = valid_idx(sort_idx);
fprintf('   Drawing %d/%d visible faces (back-to-front).\n', ...
    numel(sorted_face_idx), n_faces);

%% =========================================================================
% STEP 5: SOFTWARE RASTERIZATION (The Core of Ex 3.27)
% =========================================================================
% This is the step where we ARE the GPU.
%
% For each visible face, triangulate it into a fan, then for every 2D pixel
% inside the projected triangle compute:
%   - Barycentric weights (alpha, beta, gamma) in screen space
%   - Interpolate (w=1/Z) and (u/Z, v/Z) — the projective attributes
%   - Final perspective divide: u_true = (u/Z)_interp / (1/Z)_interp
%   - Bilinear sample the synthetic texture at (u_true, v_true)
%   - Write RGB into I_canvas[y, x, :]
%
% NO 3D plotting. I_canvas is a flat 2D matrix. imshow() is the only display.
fprintf('[Step 5] Rasterizing %d faces into 2D canvas (%dx%d)...\n', ...
    numel(sorted_face_idx), canvas_w, canvas_h);

% Allocate flat 2D canvas (dark background)
I_canvas = repmat(reshape([0.08, 0.09, 0.11], 1, 1, 3), canvas_h, canvas_w);

% Precompute texture pixel-space coordinates for the 4 face corners
u_norm = tex_coords(:, 1) ./ tex_coords(:, 3);
v_norm = tex_coords(:, 2) ./ tex_coords(:, 3);
u_tex  = u_norm * (W_tex - 1) + 1;  % [1, W_tex]
v_tex  = v_norm * (H_tex - 1) + 1;  % [1, H_tex]

total_pixels = 0;
for f_idx = 1:numel(sorted_face_idx)
    face_idx       = sorted_face_idx(f_idx);
    v_idx          = faces{face_idx};
    pts_poly_cam   = pts_cam(:, v_idx)';    % [N x 4] camera-space homogeneous
    pts_poly_screen = pts_screen(:, v_idx)'; % [N x 2] pixel-space

    nV = numel(v_idx);

    % Triangulate polygon with a triangle fan (vertex 1 is pivot)
    for t = 1:(nV - 2)
        idxA = 1;
        idxB = t + 1;
        idxC = t + 2;

        % ---- Screen-space triangle vertices ----
        p1 = pts_poly_screen(idxA, :);  % [x, y]
        p2 = pts_poly_screen(idxB, :);
        p3 = pts_poly_screen(idxC, :);

        % ---- Camera-space depths (Z coordinate) ----
        Z1 = pts_poly_cam(idxA, 3);
        Z2 = pts_poly_cam(idxB, 3);
        Z3 = pts_poly_cam(idxC, 3);
        Z1 = max(Z1, 1e-6);
        Z2 = max(Z2, 1e-6);
        Z3 = max(Z3, 1e-6);

        % ---- Texture pixel coords at each triangle vertex ----
        ut1 = u_tex(idxA);  vt1 = v_tex(idxA);
        ut2 = u_tex(idxB);  vt2 = v_tex(idxB);
        ut3 = u_tex(idxC);  vt3 = v_tex(idxC);

        % ---- Perspective-correct projective attributes ----
        % Interpolate (1/Z), (u/Z), (v/Z) linearly in screen space.
        % Then recover true u, v by the final divide:
        %   u_true = (u/Z)_interp / (1/Z)_interp
        w1 = 1/Z1;  u1_z = ut1/Z1;  v1_z = vt1/Z1;
        w2 = 1/Z2;  u2_z = ut2/Z2;  v2_z = vt2/Z2;
        w3 = 1/Z3;  u3_z = ut3/Z3;  v3_z = vt3/Z3;

        % ---- Bounding box (clamped to canvas) ----
        min_x = max(1,        floor(min([p1(1), p2(1), p3(1)])));
        max_x = min(canvas_w, ceil( max([p1(1), p2(1), p3(1)])));
        min_y = max(1,        floor(min([p1(2), p2(2), p3(2)])));
        max_y = min(canvas_h, ceil( max([p1(2), p2(2), p3(2)])));

        if min_x > max_x || min_y > max_y, continue; end

        % ---- Generate pixel grid inside bounding box ----
        [X_grid, Y_grid] = meshgrid(min_x:max_x, min_y:max_y);

        x1 = p1(1); y1 = p1(2);
        x2 = p2(1); y2 = p2(2);
        x3 = p3(1); y3 = p3(2);

        % ---- Barycentric coordinates via 2x2 linear solve ----
        % For point P inside triangle (P1,P2,P3):
        %   D    = (x2-x1)(y3-y1) - (x3-x1)(y2-y1)
        %   beta = [(P-P1) x (P3-P1)] / D
        %   gamma= [(P2-P1) x (P-P1)] / D
        %   alpha= 1 - beta - gamma
        D = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1);
        if abs(D) < 1e-6, continue; end

        beta  = ((X_grid - x1)*(y3 - y1) - (x3 - x1)*(Y_grid - y1)) / D;
        gamma = ((x2 - x1)*(Y_grid - y1) - (X_grid - x1)*(y2 - y1)) / D;
        alpha = 1 - beta - gamma;

        % ---- Pixel-in-triangle test ----
        inside = (alpha >= 0) & (beta >= 0) & (gamma >= 0);
        if ~any(inside(:)), continue; end

        x_canvas = X_grid(inside);
        y_canvas = Y_grid(inside);
        a_v = alpha(inside);
        b_v = beta(inside);
        g_v = gamma(inside);

        % ---- Perspective-correct interpolation ----
        % Step 1: interpolate projective attributes linearly in screen space
        w_interp   = a_v*w1   + b_v*w2   + g_v*w3;
        u_z_interp = a_v*u1_z + b_v*u2_z + g_v*u3_z;
        v_z_interp = a_v*v1_z + b_v*v2_z + g_v*v3_z;

        % Step 2: final perspective divide to recover true texture coords
        w_interp(abs(w_interp) < 1e-12) = 1e-12;
        src_x = u_z_interp ./ w_interp;   % true u in pixel space
        src_y = v_z_interp ./ w_interp;   % true v in pixel space

        % ---- Discard out-of-bounds texture samples ----
        valid_tex = (src_x >= 1) & (src_x <= W_tex) & ...
                    (src_y >= 1) & (src_y <= H_tex);
        if ~any(valid_tex), continue; end

        src_x_v = src_x(valid_tex);
        src_y_v = src_y(valid_tex);
        xc      = x_canvas(valid_tex);
        yc      = y_canvas(valid_tex);

        % ---- Bilinear texture sampling ----
        tx_f = floor(src_x_v);  ty_f = floor(src_y_v);
        tx_c = tx_f + 1;         ty_c = ty_f + 1;

        tx_f = max(1, min(W_tex, tx_f));
        tx_c = max(1, min(W_tex, tx_c));
        ty_f = max(1, min(H_tex, ty_f));
        ty_c = max(1, min(H_tex, ty_c));

        dx = src_x_v - floor(src_x_v);
        dy = src_y_v - floor(src_y_v);

        w00 = (1-dx).*(1-dy);
        w01 =    dx .*(1-dy);
        w10 = (1-dx).*   dy;
        w11 =    dx .*   dy;

        % Linear index into texture (column-major: row + (col-1)*H)
        idx00 = ty_f + (tx_f - 1)*H_tex;
        idx01 = ty_f + (tx_c - 1)*H_tex;
        idx10 = ty_c + (tx_f - 1)*H_tex;
        idx11 = ty_c + (tx_c - 1)*H_tex;

        % Linear index into canvas (column-major)
        canvas_idx = yc + (xc - 1)*canvas_h;

        % Write RGB into the flat 2D canvas matrix
        for ch = 1:3
            Ic  = I_tex(:, :, ch);
            val = w00.*Ic(idx00) + w01.*Ic(idx01) + ...
                  w10.*Ic(idx10) + w11.*Ic(idx11);
            I_canvas(canvas_idx + (ch-1)*canvas_h*canvas_w) = val;
        end
        total_pixels = total_pixels + numel(xc);
    end
end
fprintf('   Total pixels rasterized: %d\n', total_pixels);

%% =========================================================================
% STEP 6: 2D DISPLAY — imshow() on the flat canvas matrix
% =========================================================================
% The canvas I_canvas is a plain [600 x 800 x 3] double matrix.
% We call imshow() — a PURE 2D image viewer.
% There is NO 3D axes, NO surf/patch/mesh, NO OpenGL 3D scene.
% The 3D illusion exists entirely in the pixel values we computed above.
fprintf('[Step 6] Displaying 2D canvas with imshow()...\n');

viz = OceanViz.Core('Theme', 'Dark');
theme = viz.Colors;

fig = figure( ...
    'Name',     'Ex 3.27 — Pure Software Rasterizer', ...
    'Position', [80, 60, 960, 760], ...
    'Color',    theme.Background);

% --- Main render panel ---
ax_render = axes('Parent', fig, ...
    'Position', [0.02, 0.12, 0.96, 0.82], ...
    'Color',    theme.Background);

imshow(I_canvas, 'Parent', ax_render);
axis(ax_render, 'off');
title(ax_render, ...
    'Ex 3.27 — Perspective-Correct Software Rasterizer  |  Synthetic Checkerboard Texture', ...
    'Color', theme.Text, 'FontSize', 11, 'FontWeight', 'bold', ...
    'FontName', 'Consolas');

% --- Caption annotation (pure 2D text, not a 3D axis label) ---
annotation(fig, 'textbox', [0.02, 0.01, 0.96, 0.08], ...
    'String', {
        sprintf('Canvas: %d x %d px  |  Texture: %d x %d px  |  Pixels rasterized: %d', ...
            canvas_w, canvas_h, W_tex, H_tex, total_pixels), ...
        'Projection: perspective divide (f/Z)  |  Interpolation: barycentric (1/Z weighted)  |  Sampling: bilinear'
    }, ...
    'FitBoxToText', 'off', ...
    'EdgeColor', 'none', ...
    'BackgroundColor', 'none', ...
    'Color', theme.Grid, ...
    'FontSize', 8.5, ...
    'FontName', 'Consolas', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle');

%% =========================================================================
% STEP 7: SAVE OUTPUT
% =========================================================================
script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
results_dir  = fullfile(project_root, 'Results', 'Computer Vision');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

out_path = fullfile(results_dir, 'ex327_texture_demo.png');
exportgraphics(fig, out_path, 'Resolution', 150);
fprintf('[Step 7] Output saved: %s\n', out_path);

fprintf('\n=== Experiment complete. ===\n');
