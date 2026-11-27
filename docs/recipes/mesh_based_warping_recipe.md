# Recipes: Advanced Warping & Mesh Customization

This recipe file contains copy-pasteable MATLAB code snippets for advanced geometric warping and mesh customisation.

---

## 1. Swappable RBF Kernels (Thin Plate, Gaussian, Multiquadric, IDW)

```matlab
% Define landmarks on a 256x256 grid
N = 256;
pts_src = [1, 1; N, 1; N, N; 1, N; N/2, N/2];
pts_tgt = pts_src;
pts_tgt(5, :) = pts_src(5, :) + [30, -20]; % perturb center

I_src = imread('peppers.png');
I_src = imresize(I_src, [N, N]);

% Option A: Gaussian RBF Kernel
kernel_gauss = OceanMath.Image.Resampling.GaussianRBFKernel('Beta', 1e-4);
tps_gauss = OceanMath.Image.Resampling.ThinPlateSpline(pts_tgt, pts_src, kernel_gauss);
I_warped_gauss = tps_gauss.warpImage(I_src);

% Option B: Multiquadric RBF Kernel
kernel_mq = OceanMath.Image.Resampling.MultiquadricKernel('C', 50);
tps_mq = OceanMath.Image.Resampling.ThinPlateSpline(pts_tgt, pts_src, kernel_mq);
I_warped_mq = tps_mq.warpImage(I_src);
```

---

## 2. Forward Displacement Warping (Bilinear Splatting)

```matlab
I_src = imread('peppers.png');
[H, W, ~] = size(I_src);

% Create a horizontal wave displacement field
[X, Y] = meshgrid(1:W, 1:H);
Dx = 15 * sin(2 * pi * Y / 64);
Dy = zeros(H, W);

% Warp forward
I_warped = OceanMath.Image.Resampling.DisplacementFieldWarp.warp(I_src, Dx, Dy);
```

---

## 3. Beier-Neely Line-Based Morphing

```matlab
I_src = imread('peppers.png');
[H, W, ~] = size(I_src);

% Target line segment (where features are in output)
pts_tgt_P = [W/2, H/4];
pts_tgt_Q = [W/2, 3*H/4];

% Source line segment (corresponding features in input)
pts_src_P = [W/2 - 20, H/4];
pts_src_Q = [W/2 + 20, 3*H/4];

% Warp
I_warped = OceanMath.Image.Resampling.BeierNeelyWarp.warp(I_src, ...
    pts_tgt_P, pts_tgt_Q, pts_src_P, pts_src_Q, 'a', 0.5, 'b', 1.0, 'p', 1.0);
```

---

## 4. Content-Adapted Mesh Generation

```matlab
I_src = imread('peppers.png');

% 1. Generate adapted mesh vertices and triangulation
[pts, tri] = OceanMath.Image.Resampling.ContentAdaptedMesh.generate(I_src, 80, 'MinDistance', 12);

% 2. Perturb landmarks
pts_perturbed = pts;
pts_perturbed(pts(:, 1) > size(I_src, 2)/2, 1) = pts(pts(:, 1) > size(I_src, 2)/2, 1) + 20;

% 3. Warp image using Piecewise Affine Warping
I_warped = OceanMath.Image.Resampling.PiecewiseAffineWarp.warp(I_src, pts, pts_perturbed, tri);

% 4. Plot mesh overlay
figure; imshow(I_warped); hold on;
triplot(tri, pts_perturbed(:, 1), pts_perturbed(:, 2), 'y-', 'LineWidth', 1.0);
plot(pts_perturbed(:, 1), pts_perturbed(:, 2), 'ro', 'MarkerFaceColor', 'r');

---

## 5. Bayer Color Filter Array (CFA) Demosaicing

```matlab
% 1. Create a synthetic Bayer RGGB pattern image [128 x 128]
% RGGB sequence starting with Red at (1,1)
CFA = zeros(128, 128);
% Fill Red channel (odd row, odd col)
CFA(1:2:end, 1:2:end) = 0.8;
% Fill Green channels (odd row/even col and even row/odd col)
CFA(1:2:end, 2:2:end) = 0.5;
CFA(2:2:end, 1:2:end) = 0.5;
% Fill Blue channel (even row, even col)
CFA(2:2:end, 2:2:end) = 0.3;

% 2. Instantiate the demosaicing engine
import OceanMath.Image.Resampling.DemosaicBayer
db = DemosaicBayer('rggb');

% 3. Apply Bilinear demosaicing
I_bilinear = db.bilinear(CFA);

% 4. Apply Malvar-He-Cutler (MHC) gradient-corrected demosaicing
I_mhc = db.malvarHeCutler(CFA);

disp('Demosaiced RGB image size:');
disp(size(I_mhc));
```

---

## 6. Blue/Green Screen Matting & Compositing

```matlab
% Load background image and composite image
B = imread('empty_background.jpg');
C = imread('foreground_object.jpg');

% 1. Instantiate the matting engine
import OceanMath.Image.Resampling.BlueScreenMatting
bsm = BlueScreenMatting(B);

% 2. Pull the alpha matte and despilled foreground
% AlphaMin filters background noise, AlphaMax restores full foreground opacity
[alpha, foreground] = bsm.pullMatte(C, 'AlphaMin', 0.05, 'AlphaMax', 0.85);

% 3. Composite onto a new scenic background
new_bg = imread('ocean_scenic.jpg');
composite = bsm.blendOn(foreground, alpha, new_bg);

% 4. Display the composite result
figure; imshow(composite);
```
