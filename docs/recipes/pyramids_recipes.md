# Multi-resolution Pyramids Recipes

Here are copy-pasteable recipes for instantiating, building, and reconstructing image pyramids using the `OceanMath.Image.MultiResolution` package.

---

## 1. Building a Gaussian Pyramid
Use the `GaussianPyramid` class to generate a sequence of low-pass decimated images.

```matlab
% Import the package classes
import OceanMath.Image.MultiResolution.GaussianPyramid

% Load input image and convert to double
I = double(imread('Data/images/ocean_img_gray.jpg')) / 255.0;

% Instantiate and build a 5-level Gaussian Pyramid
gp = GaussianPyramid('Boundary', 'replicate');
gp.build(I, 5);

% Inspect level sizes
for level = 1:gp.NumLevels
    [h, w, ~] = size(gp.Levels{level});
    fprintf('Gaussian Level %d size: %d x %d\n', level, h, w);
end

% Display Level 3
figure;
imshow(gp.Levels{3});
title('Gaussian Pyramid Level 3');
```

---

## 2. Building a Laplacian Pyramid & Perfect Reconstruction
Use the `LaplacianPyramid` class to capture band-pass detail levels and verify mathematically perfect reconstruction.

```matlab
% Import the package classes
import OceanMath.Image.MultiResolution.LaplacianPyramid

% Load input image
I = double(imread('Data/images/ocean_img.jpg')) / 255.0;

% Instantiate and build a 5-level Laplacian Pyramid
lp = LaplacianPyramid('Boundary', 'replicate');
lp.build(I, 5);

% Reconstruct the original image from the pyramid
I_rec = lp.reconstruct();

% Compute reconstruction errors
max_err = max(abs(I(:) - I_rec(:)));
fprintf('Maximum Reconstruction Error: %e\n', max_err);

% Verify reconstruction matches down to double precision
if max_err < 1e-12
    disp('Perfect reconstruction verified!');
else
    disp('Reconstruction contains numerical errors.');
end
```

---

## 3. Visualizing Laplacian Detail Levels
Since detail levels represent positive and negative differences, they must be offset to a mid-gray level (`0.5`) to display correctly.

```matlab
% Import package classes
import OceanMath.Image.MultiResolution.LaplacianPyramid

I = double(imread('Data/images/ocean_img_gray.jpg')) / 255.0;
lp = LaplacianPyramid();
lp.build(I, 5);

figure;
% Plot a detail level (Level 2) with gray scaling
subplot(1, 2, 1);
imshow(max(0.0, min(1.0, 2.0 * lp.Levels{2} + 0.5)));
title('Laplacian Level 2 Detail');

% Plot the residual low-frequency level (Level 5) directly
subplot(1, 2, 2);
imshow(lp.Levels{5});
title('Laplacian Level 5 Residual');
```

---

## 4. Custom Filter Kernels at Runtime
You can pass custom kernel instances (subclasses of `IFilterKernel` such as `TriangleKernel`, `GaussianKernel`), built-in string identifiers (like `'box'`), or raw numeric arrays. The constructors automatically resolve them using `KernelFactory`.

```matlab
% Import package classes
import OceanMath.Image.MultiResolution.LaplacianPyramid
import OceanMath.Signal.Kernels.TriangleKernel

I = double(imread('Data/images/ocean_img_gray.jpg')) / 255.0;

% Option A: Pass an IFilterKernel instance (Triangle/Bilinear filter)
triangle_kernel = TriangleKernel(3);
lp = LaplacianPyramid('Kernel', triangle_kernel);
lp.build(I, 4);

% Option B: Pass a string name (Box average filter)
% lp = LaplacianPyramid('Kernel', 'box');

% Option C: Pass a raw numeric matrix (automatically wrapped in a CustomKernel)
% lp = LaplacianPyramid('Kernel', [1.0, 2.0, 1.0] / 4.0);

% Verify perfect reconstruction is still achieved
I_rec = lp.reconstruct();
max_err = max(abs(I(:) - I_rec(:)));
fprintf('Custom Kernel Max Reconstruction Error: %e\n', max_err);
```

---

## 5. Seamless Multiresolution Blending (Orapple Blend)
Use the `PyramidBlender` class to seamlessly merge two images using a spatial-frequency feathered mask.

```matlab
% Import classes
import OceanMath.Image.MultiResolution.PyramidBlender

% Load two input images (must be same size)
IA = double(imread('Data/images/ocean_img.jpg')) / 255.0;
IB = double(imread('Data/images/elon_musk.jpg')) / 255.0;

% Resize IB to match IA if necessary
[hA, wA, cA] = size(IA);
IB = imresize(IB, [hA, wA]);

% Create a soft horizontal feathered blend mask [0, 1]
[x, ~] = meshgrid(linspace(0, 1, wA), linspace(0, 1, hA));
maskA = 1 ./ (1 + exp(-30 * (0.5 - x)));  % Sigmoid transition
maskB = 1 - maskA;

% Instantiate PyramidBlender with 5 levels and Gaussian kernel
blender = PyramidBlender('NumLevels', 5);

% Perform multiresolution blend
I_blend = blender.blend({IA, IB}, {maskA, maskB});

% Display results
figure;
subplot(1, 3, 1); imshow(IA); title('Image A');
subplot(1, 3, 2); imshow(IB); title('Image B');
subplot(1, 3, 3); imshow(I_blend); title('Seamless Pyramid Blend');
```

---

## 6. Filter Bypassing to Demonstrate Aliasing
Set the `BypassFilter` name-value parameter to true in reduction, expansion, or pyramid creation to bypass low-pass filtering and demonstrate Moiré artifacts.

```matlab
% Import classes
import OceanMath.Image.MultiResolution.GaussianPyramid

% Generate a synthetic high-frequency chirp signal
gridSize = 512;
[x, y] = meshgrid(linspace(-1, 1, gridSize), linspace(-1, 1, gridSize));
r = sqrt(x.^2 + y.^2);
I_chirp = 0.5 + 0.5 * sin(120 * r.^2);

% Build standard anti-aliased Gaussian Pyramid
gp_clean = GaussianPyramid();
gp_clean.build(I_chirp, 4);

% Build aliased Gaussian Pyramid by bypassing the filter stage
gp_aliased = GaussianPyramid('BypassFilter', true);
gp_aliased.build(I_chirp, 4);

% Display comparison at Level 3 (reduced to 128 x 128)
figure;
subplot(1, 2, 1);
imshow(gp_clean.Levels{3});
title('Anti-Aliased Level 3 (filtered)');

subplot(1, 2, 2);
imshow(gp_aliased.Levels{3});
title('Aliased Level 3 (raw decimation)');
```

---

## 7. Local Laplacian Contrast Manipulation (Exercise 3.20)
Use the `LocalLaplacianFilter` class to perform edge-preserving detail enhancement, skin smoothing, or high dynamic range (HDR) tone mapping. The filter automatically converts color images to CIE L*a*b* to process the luminance (L*) channel exclusively.

```matlab
% Import the package classes
import OceanMath.Image.MultiResolution.LocalLaplacianFilter

% Load a color image [0, 1] double
I = im2double(imread('Data/images/ocean_img.jpg'));

% Configure Detail Enhancement (Alpha < 1, Beta = 1)
% This boosts fine textures without causing halos
llf_enhance = LocalLaplacianFilter('NumLevels', 4, 'NumBins', 8, ...
                                   'Sigma', 0.1, 'Alpha', 0.25, 'Beta', 1.0);
I_enhance = llf_enhance.filter(I);

% Configure Edge-Preserving Denoising / Smoothing (Alpha > 1, Beta = 1)
% This flattens detail/noise (skin smoothing) while maintaining sharp boundaries
llf_smooth = LocalLaplacianFilter('NumLevels', 4, 'NumBins', 8, ...
                                  'Sigma', 0.15, 'Alpha', 2.0, 'Beta', 1.0);
I_smooth = llf_smooth.filter(I);

% Configure Tone Mapping / Dynamic Range Compression (Alpha = 1, Beta < 1)
% This compresses strong boundaries to reveal details in shadows and highlights
llf_tone = LocalLaplacianFilter('NumLevels', 4, 'NumBins', 8, ...
                                'Sigma', 0.2, 'Alpha', 1.0, 'Beta', 0.25);
I_tone = llf_tone.filter(I);

% Plot results
figure;
subplot(2, 2, 1); imshow(I); title('Original');
subplot(2, 2, 2); imshow(I_enhance); title('Detail Enhancement (\alpha = 0.25)');
subplot(2, 2, 3); imshow(I_smooth); title('Smoothing (\alpha = 2.0)');
subplot(2, 2, 4); imshow(I_tone); title('Tone Mapping (\beta = 0.25)');
```
