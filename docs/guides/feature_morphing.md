# Feature-Based Image Morphing

## Overview
Feature-based image morphing combines geometric warping and color interpolation (cross-dissolving) to create smooth, natural transitions between two images. Implementing textbook Section 3.6.3, this subsystem implements the complete 4-step morphing pipeline:
1. **Triangulation / Structural representation**: Delaunay triangulation is constructed from control points on the source image(s).
2. **Intermediate Skeleton (In-Between Shape)**: For parameter `t` in `[0, 1]`, control coordinates are linearly interpolated:
   `pts_mid = (1 - t) * pts_A + t * pts_B`
3. **Double Inverse Mapping**: Both Image A and Image B are warped towards the intermediate shape `pts_mid` using backward bilinear resampling:
   `I_A_warp = warp(I_A, pts_A -> pts_mid)`
   `I_B_warp = warp(I_B, pts_B -> pts_mid)`
4. **Cross-Dissolve Blend**: The final frame is a linear blend of the two warped images:
   `I_morph = (1 - t) * I_A_warp + t * I_B_warp`

Additionally, this subsystem allows analyzing the *chord-vs-arc* artifact that occurs when simulating circular rotations via linear coordinate interpolation.

---

## Usage

The core class is `OceanMath.Image.Resampling.ImageMorpher`. It is a static utility providing methods to generate individual frames, sequence sets, or rotation control points.

### Morphing a Single Frame
```matlab
% Define source A and B images
I_A = imread('ocean_img.jpg');
I_B = imread('elon_musk.jpg');

% Create 4x4 uniform control points (16 landmarks)
res = 4;
[pts_A, pts_B] = OceanMath.Image.Resampling.ImageMorpher.uniformGridPoints(size(I_A, 1:2), res);

% Apply a manual offset to some target control points
pts_B(6, :) = pts_B(6, :) + [20, -10];
pts_B(7, :) = pts_B(7, :) + [-20, 15];

% Morph at the halfway point (t = 0.5) using Piecewise Affine warping
I_morph = OceanMath.Image.Resampling.ImageMorpher.morphFrame(...
    I_A, I_B, pts_A, pts_B, 0.5, 'WarpMode', 'PiecewiseAffine');

imshow(I_morph);
```

### Supported Warp Backends
Three backends from Exercise 3.23 are supported:
* `'PiecewiseAffine'`: Triangulates the grid and maps each triangle via barycentric coordinates (C0-continuous).
* `'ThinPlateSpline'`: Globally smooth Thin Plate Spline RBF interpolation (C-infinity continuous).
* `'BSplineGrid'`: Free-form deformation using cubic B-spline grid interpolation (C2-continuous).

---

## Examples

### Generating an Animated Morph Sequence
This example generates 12 frames animating a transition between two images and saves the result as an animated GIF:
```matlab
I_A = im2double(imread('ocean_img.jpg'));
I_B = imresize(im2double(imread('elon_musk.jpg')), size(I_A, 1:2));

res = 5;
[pts_A, pts_B] = OceanMath.Image.Resampling.ImageMorpher.uniformGridPoints(size(I_A, 1:2), res);

% Displace center points
pts_B(13, :) = pts_B(13, :) + [15, 30];

% Generate 24-frame cell array
N = 24;
frames = OceanMath.Image.Resampling.ImageMorpher.morphSequence(...
    I_A, I_B, pts_A, pts_B, N, 'WarpMode', 'ThinPlateSpline');

% Write to animated GIF
gifPath = 'my_morph.gif';
for i = 1:N
    [rgb_idx, cmap] = rgb2ind(im2uint8(frames{i}), 256);
    if i == 1
        imwrite(rgb_idx, cmap, gifPath, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
    else
        imwrite(rgb_idx, cmap, gifPath, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
    end
end
```

---

## See Also
* [Local Image Warping](local_image_warping.md)
* [Mesh-Based Warping](mesh-based_warping_guide.md)
