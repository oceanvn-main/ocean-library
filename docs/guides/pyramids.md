# Multi-resolution Image Pyramids (Gaussian & Laplacian) Guide

## Overview
Multi-resolution representation allows an image to be processed and analyzed at different scales. This is a fundamental primitive in computer vision for scale-invariant feature extraction, template matching, image blending, and coarse-to-fine motion estimation.

The two main pyramid types implemented are:
1. **Gaussian Pyramid**: A sequence of low-pass filtered images, where each level is obtained by smoothing the previous level with a Gaussian binomial filter and downsampling by a factor of 2 (decimation).
2. **Laplacian Pyramid**: A sequence of band-pass (detail) images, where each level represents the difference between a Gaussian level and the expanded/interpolated version of the next lower resolution level. The Laplacian pyramid allows mathematically perfect reconstruction of the original image.

---

## Mathematical Foundations

### 1. Reduction (Direct Strided Convolution)
To compute the next lower-resolution level in a Gaussian pyramid, we apply a low-pass filter to the current level to prevent aliasing, and sub-sample by taking every second pixel.
* **Separable Filter Kernel**: We use a 1D binomial filter `w = [1, 4, 6, 4, 1] / 16` by default.
* **Polymorphic Kernels**: Any `IFilterKernel` (like `GaussianKernel`, `TriangleKernel`, `BoxKernel`) or raw numeric array can be resolved via `KernelFactory`.
* **Direct Strided filtering**: Implemented completely from scratch in pure MATLAB (no toolbox dependencies like `imfilter`). Decimation is performed directly inside the horizontal and vertical convolution loops (strided summation), avoiding large intermediate memory buffers.

### 2. Expansion (Polyphase Upsampling & Interpolation)
To reconstruct a lower-resolution level to match the next higher resolution, we upsample by a factor of 2 and interpolate.
* **Polyphase Decomposition**: Instead of zero-stuffing the image to full size and then filtering (which wastes ~50% of multiply operations on zero samples), we decompose the filter into even/odd polyphase sub-filters and convolve the *small* input directly:
  - For a 1D filter `h` of length K, split into `h_e[k] = h[2k]` (even) and `h_o[k] = h[2k+1]` (odd).
  - Compute `y[2n] = conv(x, h_e)` and `y[2n+1] = conv(x, h_o)` on the small input `x`, then interleave.
  - For 2D non-separable kernels, decompose into four sub-kernels (ee, eo, oe, oo) applied to the small input.
* **Separable Polyphase**: Two 1D polyphase passes (horizontal, then vertical), each operating on the small-dimension arrays.
* **Energy Preservation**: The upsampled image is filtered using the kernel scaled by 2.0 in each 1D pass (total 4.0 in 2D) to restore the correct intensity scale.

### 3. Laplacian Pyramid & Reconstruction
The Laplacian pyramid represents detail bands (high-frequency components) at each level:
* **Detail Band**: For level `i`:
  `L_i = G_i - expand(G_{i+1})`
* **Top Level**: The last level is the low-frequency residual:
  `L_N = G_N`
* **Perfect Reconstruction**: Starting from the smallest level `L_N`, we recursively expand and add the detail band:
  `I_rec = L_i + expand(I_rec)`

---

## Class Architecture

All pyramid classes live under the namespace `OceanMath.Image.MultiResolution`:

* **`IPyramid`**: Abstract base class defining the contract properties (`Levels`, `NumLevels`) and methods (`build`, `reconstruct`).
* **`PyramidOperator`**: Static class containing the optimized separable `reduce` and `expand` algorithms.
* **`GaussianPyramid`**: Concrete implementation representing Gaussian low-pass scale-space.
* **`LaplacianPyramid`**: Concrete implementation representing Laplacian band-pass detail levels.
* **`PyramidBlender`**: Concrete class implementing Laplacian pyramid image blending (Burt & Adelson 1983).
* **`LocalLaplacianFilter`**: Concrete class implementing local Laplacian filtering (Paris et al. 2011) with stored intermediate pyramids (`InputGaussianPyramid`, `InputLaplacianPyramid`, `OutputLaplacianPyramid`) and a visualization method `plotPyramids()`.

---

## Laplacian Pyramid Blending (Exercise 3.18)
When two images are merged along a seam, direct alpha blending causes a visible boundary (if the transition is sharp) or blur/ghosting (if the transition is feathered). Laplacian pyramid blending resolves this by blending each spatial frequency scale independently with a matching transition width:
1. **Pyramid Construction**: Build Laplacian pyramids for both images A and B, and a Gaussian pyramid for the blending mask W.
2. **Multi-scale Blend**: At each pyramid level l, blend the Laplacian detail levels using the Gaussian-filtered mask weights at that level:
   B_l = LP_A_l * GP_W_l + LP_B_l * (1 - GP_W_l)
3. **Pyramid Reconstruction**: Collapse/reconstruct the blended Laplacian pyramid to obtain the final seamless blended image.

---

## Filter Bypassing & Aliasing (Exercise 3.17)
To demonstrate aliasing and Moiré patterns caused by raw downsampling without pre-filtering:
* **`BypassFilter`**: A configuration property/flag supported by the reduction/expansion operators and pyramids.
* **Gaussian Reduction with Bypass**: If true, the low-pass filter stage is skipped, performing direct subsampling (every 2nd pixel is retained). This exhibits heavy aliasing in high-frequency regions (e.g., Moiré patterns in chirp signals).
* **Gaussian Expansion with Bypass**: If true, interpolation is skipped in favor of nearest-neighbor pixel replication.

---

## Local Laplacian Contrast Manipulation (Exercise 3.20)
Local Laplacian Filtering (Paris et al. 2011) is a fast, edge-preserving filter that performs contrast manipulation, detail enhancement, smoothing/denoising, and dynamic range compression (tone mapping) without generating halos or gradient reversal artifacts.

### 1. Algorithm Overview
Standard Laplacian pyramid reconstruction builds details and combines them. If we directly modify Laplacian coefficients, we introduce halo artifacts near strong edges. Local Laplacian filters resolve this by computing a spatially-varying remapping function at every pixel location.
To make this computationally efficient (O(N) instead of O(N^2) complexity), the algorithm uses discretized intensity bins:
1. **Intensity Discretization**: Define a set of M intensity reference bins (e.g. M = 8 or 16) spanning [0, 1].
2. **Remapped Pyramids**: For each bin g_k, create a remapped image I_k = r(I; g_k) using a threshold-based remapping function and construct its corresponding Laplacian pyramid L^(k).
3. **Local Interpolation**: For each pixel p at pyramid level l, lookup its local Gaussian value G_l[p] from the input image's Gaussian pyramid. Interpolate the final Laplacian coefficient L_out_l[p] from the matching spatial locations in the neighboring bin pyramids L^(k)_l[p] and L^(k+1)_l[p] such that g_k <= G_l[p] <= g_(k+1).
4. **Reconstruction**: Collapse/reconstruct the interpolated Laplacian pyramid L_out to retrieve the final output image.

### 2. Remapping Function
The remapping function r(x; g) operates on pixel intensity x relative to reference level g:
* If |x - g| <= sigma: The detail region is scaled by a power function to enhance or suppress local details:
  r(x; g) = g + sign(x - g) * sigma * (|x - g| / sigma)^alpha
  - alpha < 1: Detail enhancement (increases local micro-contrast).
  - alpha > 1: Detail smoothing (smooths out textures, preserving outlines).
* If |x - g| > sigma: The edge region is scaled linearly by beta to compress or boost high-contrast boundaries:
  r(x; g) = g + sign(x - g) * (beta * (|x - g| - sigma) + sigma)
  - beta < 1: Tone mapping / edge compression (compresses large range gradients).

---

## Visualization
Because Laplacian detail bands represent differences, they contain both negative and positive values centered around zero. To visualize them, we scale and offset them to a mid-gray baseline:
`I_viz = 2.0 * Laplacian_Level + 0.5`
The residual (last level) is a standard low-frequency Gaussian image and does not require offsetting.

---

## Interactive Sandbox Visualizer
The library includes an interactive GUI visualization sandbox inside the [Computer Vision Sandbox](file:///c:/Ocean%20Library/Matlab%20code%20practice/+OceanViz/+App/Computer%20Vision/ComputerVisionSandbox.m). 

### How to Run
Launch the application via:
```matlab
app = OceanViz.App.ComputerVisionSandbox();
```

### Features
* **5. Image Pyramids Tab**:
  - Toggle between Gaussian decimation levels and Laplacian band-pass detail levels.
  - Adjust the levels (2 to 6) and kernel size (3 to 9) interactively.
  - Swap smoothing kernels (`Gaussian`, `Triangle`, `Box`, `Windowed Sinc`).
  - Check **Bypass Low-Pass Filter** to view aliasing/decimation artifacts in real time.
* **7. Image Blending Tab (New)**:
  - Load custom or preset images A & B and select a blend mask (e.g. Feathered Split, Center Circle).
  - Compare standard linear alpha blending (exhibiting seam artifacts) with multiresolution Pyramid Blending side-by-side.
  - View real-time PSNR metrics comparing the two blending approaches.

---

## See Also
* [Spatial Filtering & Lifting](filtering_and_lifting.md)
* [Computer Vision Primitives](computer_vision_primitives.md)
* [Image Pyramids Recipes](../recipes/pyramids_recipes.md)
