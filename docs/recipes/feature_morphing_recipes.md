# Feature-Based Image Morphing Recipes

## Quick Morph at Halfway Point (t = 0.5)
```matlab
% Set up paths
cd('c:\Ocean Library\Matlab code practice');
addpath(genpath(pwd));

% Load and resize images to match
I_A = im2double(imread('Data/images/ocean_img.jpg'));
I_B = imresize(im2double(imread('Data/images/elon_musk.jpg')), [256, 256]);
I_A = imresize(I_A, [256, 256]);

% Initialize 4x4 control point grid
[pts_A, pts_B] = OceanMath.Image.Resampling.ImageMorpher.uniformGridPoints(size(I_A, 1:2), 4);

% Offsets to displace landmarks in Image B
pts_B(6, :)  = pts_B(6, :)  + [15, 20];
pts_B(11, :) = pts_B(11, :) + [-15, -20];

% Perform morph
I_morph = OceanMath.Image.Resampling.ImageMorpher.morphFrame(...
    I_A, I_B, pts_A, pts_B, 0.5, 'WarpMode', 'PiecewiseAffine');

% Plot
figure;
subplot(1,3,1); imshow(I_A); title('Image A');
subplot(1,3,2); imshow(I_morph); title('Morphed t=0.5');
subplot(1,3,3); imshow(I_B); title('Image B');
```

---

## Simulating Rotation and Measuring Chord-vs-Arc Deviation
```matlab
% Set up image size and rotation parameters
img_sz = [256, 256];
theta_deg = 60;
grid_res = 8;

% Get rotation control points
[pts_A, pts_B] = OceanMath.Image.Resampling.ImageMorpher.rotationControlPoints(...
    img_sz, theta_deg, grid_res);

% Load a grid/checkerboard to clearly show structural warping
I_grid = im2double(imread('Data/images/ocean_img.jpg')); % or any test image
I_grid = imresize(I_grid, img_sz);

% Compute morph at halfway (t = 0.5)
I_morph = OceanMath.Image.Resampling.ImageMorpher.morphFrame(...
    I_grid, I_grid, pts_A, pts_B, 0.5, 'WarpMode', 'PiecewiseAffine');

% Analyze radial contraction (chord midpoint r_mid = r * cos(theta/2))
cx = (img_sz(2) + 1) / 2;
cy = (img_sz(1) + 1) / 2;

% Radius of original coordinates
dx = pts_A(:, 1) - cx;
dy = pts_A(:, 2) - cy;
r_true = sqrt(dx.^2 + dy.^2);

% Radius of morphed coordinates at t=0.5
pts_mid = (pts_A + pts_B) / 2;
r_chord = sqrt((pts_mid(:,1) - cx).^2 + (pts_mid(:,2) - cy).^2);

% Compute average contraction ratio
valid_idx = r_true > 10; % avoid center division by zero
mean_ratio = mean(r_chord(valid_idx) ./ r_true(valid_idx));
theoretical_ratio = cos(deg2rad(theta_deg)/2);

fprintf('Mean measured radius ratio: %.4f\n', mean_ratio);
fprintf('Theoretical ratio (cos(theta/2)): %.4f\n', theoretical_ratio);
fprintf('Average radial contraction: %.2f pixels\n', mean(r_true(valid_idx) - r_chord(valid_idx)));
```
