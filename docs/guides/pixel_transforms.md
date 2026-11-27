# Pixel Transformations Guide

## Overview

Pixel transformations and spatial filtering are fundamental operations in image processing and computer vision. The `OceanMath.Image.Pixels` package implements a generic, high-performance, and modular architecture for pixel-level manipulations. 

By utilizing the **Strategy** and **Pipes and Filters** design patterns, the package decouples individual mathematical transformation algorithms from execution flows, enabling runtime composition, pipeline chaining, and clean code reuse.

---

## Design Patterns & Architecture

The package architecture consists of three primary components:

1. **`IPixelTransform` (Abstract Strategy)**: Defines the standard polymorphic interface for all transformation strategies. Every pixel operator implements the `apply(I_in)` method.
2. **Concrete Strategies**: Classes representing specific mathematical operations (e.g., contrast stretching, color conversion, convolution).
3. **`TransformPipeline` (Compositor / Pipeline)**: Chains a sequential list of `IPixelTransform` objects, piping the output of one strategy directly into the input of the next.

```
       [Input Image]
             │
             ▼
┌───────────────────────────┐
│     TransformPipeline     │
│  (Applies sequential list)│
└─────┬───────────────┬─────┘
      │               │
      ▼ (Strategy 1)  ▼ (Strategy 2)
┌───────────┐   ┌───────────┐
│ColorSpace │   │LinearScale│  ... [Output Image]
│Converter  │   │Operator   │
└───────────┘   └───────────┘
```

---

## Mathematical Models

### 1. Point Operators

#### Linear Operator
Adjusts contrast and brightness by scaling and shifting intensity levels:
`I_out = Scale * I_in + Bias`
Inputs are optionally clipped to the standard range [0, 1] for double precision or [0, 255] for uint8 precision.

#### Gamma Correction
Applies a power-law transformation to adjust luminance non-linearly:
`I_out = C * (I_in ^ Gamma)`
Intensity levels are normalized to [0, 1] before exponentiation, then scaled back.

### 2. Lookup Table (LUT) Mapping
Transforms intensity values using a pre-computed 256-element array `LUT`. For an 8-bit input intensity `r`:
`I_out(x, y) = LUT(r + 1)`
This maps intensities in O(1) time complexity. For multi-channel images (e.g. RGB), the same table is applied to all channels, or channel-specific tables are used.

### 3. Histogram Equalization & Adaptive Equalization (CLAHE)

#### Global Histogram Equalization
Enhances image contrast by mapping the cumulative distribution function (CDF) of luma intensities to a uniform distribution.

1. **Histogram Computation**:
   Compute pixel counts for each gray value `r` in the range [0, 255].
   `H(r) = count(I_in == r)`

2. **Cumulative Distribution Function (CDF)**:
   `CDF(r) = sum from i=0 to r of H(i)`

3. **Transfer Function**:
   `T(r) = (L - 1) * CDF(r) / N`
   Where `L = 256` (intensity levels) and `N` is the total number of pixels.

4. **Partial Equalization**:
   Blends the equalized transfer function with the identity mapping using a blending factor `Alpha` in the range [0, 1]:
   `T_partial(r) = (1 - Alpha) * r + Alpha * T(r)`

#### Contrast-Limited Adaptive Histogram Equalization (CLAHE)
Divides the image into a grid of non-overlapping tiles (contextual regions) and equalizes each tile individually:

1. **Histogram Clipping**: To limit contrast enhancement, the height of each tile's histogram is clipped at a user-defined threshold `ClipLimit`. The clipped pixels are redistributed uniformly across all histogram bins before computing the CDF.
2. **Bilinear Interpolation**: To eliminate boundary artifacts between tiles, mapping transfer functions are interpolated bilinearly. For any pixel at (x, y), its intensity is mapped using the transfer functions of the four closest tile centers, weighted by their linear distances. For pixels near image borders, mapping falls back to 1D linear interpolation (along edge tiles) or direct 1D mapping (corner tiles).

### 4. Dynamic Contrast Stretching

#### Min-Max Stretch
Stretches the image dynamic range to fill a target range `[T_min, T_max]` (defaulting to [0, 1] or [0, 255]):
`I_out = (I_in - I_min) / (I_max - I_min) * (T_max - T_min) + T_min`
Where `I_min` and `I_max` are the minimum and maximum intensities present in the input image.

#### Mean Match Stretch
Centers the image intensities around the luma mean, applies scaling, and shifts the distribution to align the mean with a target value `TargetMean`:
`I_out = Scale * (I_in - Mean(I_in)) + TargetMean`
Resulting intensities are clipped to the valid boundaries.

### 5. Color Space Conversion
Converts multi-channel image arrays between color spaces:
* **RGB to Gray**: Computed using Rec. 601 luma weights:
  `Y = 0.298936 * R + 0.587043 * G + 0.114021 * B`
* **Gray to RGB**: Replicates the single gray channel across three identical R, G, and B channels.
* **RGB to HSV / HSV to RGB**: Converts between additive color and cylindrical coordinate representations.
* **RGB to YCbCr / YCbCr to RGB**: Converts to luminance and chrominance representation.

### 6. Spatial Filtering (Neighborhood Operations)
Neighborhood filtering operations compute the output value at each pixel using a linear combination of its neighbors weighted by a kernel matrix.

#### Cross-Correlation (Direct Matching)
Applies the kernel exactly as written, multiplying overlapping coefficients directly:
`I_out(y, x) = sum_{i, j} I_in(y + i, x + j) * K(i, j)`

#### True Mathematical Convolution (Rotated Matching)
Requires rotating the kernel by 180 degrees (flipping it horizontally and vertically) before multiplying overlapping coefficients:
`I_out(y, x) = sum_{i, j} I_in(y + i, x + j) * K(-i, -j)`
Or equivalently, using the 180-degree flipped kernel:
`K_flipped = rot90(K, 2)`
`I_out(y, x) = sum_{i, j} I_in(y + i, x + j) * K_flipped(i, j)`

#### Separable Filtering Optimization
If the 2D convolution kernel `K` is **separable**, the 2D filtering operation can be decomposed into two successive 1D operations:
1. Convolve horizontally with a 1D horizontal kernel `h`.
2. Convolve vertically with a 1D vertical kernel `v`.

A kernel is separable if and only if its rank is 1. We determine separability using Singular Value Decomposition (SVD):
`K = sum_i (sigma_i * u_i * v_i^T)`
If `sigma_2 < 1e-12 * sigma_1`, only the first singular value `sigma_1` is non-zero, and the kernel can be decomposed:
`v = u_1 * sqrt(sigma_1)` (vertical 1D kernel, size kH x 1)
`h = v_1' * sqrt(sigma_1)` (horizontal 1D kernel, size 1 x kW)
Such that `K = v * h`.

Both `TrueConvolution` and `CrossCorrelation` dynamically detect separable kernels in their constructors. If separable, they perform the convolution horizontally first and then vertically using highly vectorized index arithmetic. This reduces the computational complexity from `O(K^2)` per pixel to `O(2*K)` per pixel, accelerating filtering for large kernels.

#### Decoupled Filter Kernels (`OceanMath.Signal.Kernels`)

For researcher flexibility and parameter inspection, standard filter kernels are implemented as decoupled classes under the `OceanMath.Signal.Kernels` package:
* **`IFilterKernel` (Abstract Interface)**: Defines the uniform contract for all discrete kernels. Exposes the 2D matrix representation, indicates whether it is separable, and returns pre-decomposed 1D horizontal/vertical vectors.
* **`BoxKernel`**: Average box blur filter of customizable size.
* **`TriangleKernel`**: Bilinear/triangle blur filter.
* **`GaussianKernel`**: Parameterized Gaussian approximation using binomial coefficients.
* **`SobelKernel`**: First-order directional derivative filter (horizontal/vertical).
* **`SecondDerivativeKernel`**: Second-order cross-derivative (corner) filter.

When given an `IFilterKernel` object, the convolution/correlation classes bypass the runtime SVD analysis and read the analytical pre-decomposed 1D vectors directly, avoiding numerical rounding issues and computational overhead while retaining full mathematical parameterization.

#### Neighborhood Denoising Filters (`OceanMath.Image.Filters`)

Non-linear and advanced linear filters are placed under the `OceanMath.Image.Filters` package, inheriting from the unified `INeighborhoodFilter` base class:

* **`GaussianFilter`**: Classic linear scale-space smoothing filter convolving the image with a symmetric Gaussian kernel of standard deviation `Sigma`:
  `G(x, y) = exp(-(x^2 + y^2) / (2 * Sigma^2)) / (2 * pi * Sigma^2)`
* **`MedianFilter`**: Highly effective non-linear filter for removing Salt-and-Pepper impulse noise while preserving sharp boundaries by replacing each pixel with the median value of its neighborhood.
* **`BilateralFilter`**: Edge-preserving filter combining spatial domain filtering with range intensity similarity filtering to smooth flat details while keeping sharp transitions.
* **`NonLocalMeansFilter`**: Advanced patch-based denoising filter that computes each pixel value as a weighted average of other pixels with similar patch neighborhoods within a search window. Uses a fast O(N) vectorized shift algorithm.

### 7. Boundary Padding (Border Effects)
Neighborhood filtering operations require evaluating pixels outside the original image boundaries. The `BoundaryPadder` class expands an input H x W x C image into an array of size (H + 2*P_r) x (W + 2*P_c) x C using one of the following mechanisms (Szeliski Figure 3.13):
- **Zero**: Fills boundaries with 0.
- **Constant**: Fills boundaries with an arbitrary scalar or channel-specific vector.
- **Clamp / Replicate**: Repeats the border pixels indefinitely.
- **Wrap / Circular**: Wraps coordinates periodically (cyclic toroidal tiling).
- **Mirror / Reflect**: Symmetrically reflects coordinates across the image boundary without repeating the edge pixel. (Also supports 'symmetric' reflection which repeats the edge pixel).
- **Extend**: Extrapolates coordinates by mirroring the signal and subtracting differences from the edge pixel value (`I_pad(N + i) = 2 * I_edge - I(N - i)`), maintaining first-derivative continuity.

### 8. Zero-Stuffing and Subband Synthesis

#### Zero-Stuffing (Upsampling)
Zero-stuffing (or expansion) by integer factor `L` inserts `L - 1` zeros between consecutive samples along both dimensions:
`I_stuffed(L*(r-1) + 1, L*(c-1) + 1) = I_in(r, c)`

In the frequency domain, zero-stuffing physically compresses the original spectrum by a factor of `L` and creates `L-1` perfectly mirrored high-frequency "ghost" copies of the spectrum. This is known as synthesis aliasing or imaging artifacts. Visually, the zero-stuffed spatial image is covered in a dense checkerboard grid of black dots.

#### Synthesis Filtering
To reconstruct approximation or detail components, the zero-stuffed image is convolved with a 2D synthesis filter kernel `K` (which is the tensor product of 1D low-pass `g` and high-pass `h` synthesis filters):
- **Low-Pass (LL)**: `K = g' * g` interpolates/smooths the zero grid, removing the high-frequency ghost copies.
- **Horizontal High-Pass (LH)**: `K = g' * h` extracts horizontal details.
- **Vertical High-Pass (HL)**: `K = h' * g` extracts vertical details.
- **Diagonal High-Pass (HH)**: `K = h' * h` extracts diagonal details.

### 9. Decimation (Downsampling) and Anti-Aliasing

#### Decimation (Downsampling)
Decimation by an integer factor `L` reduces the resolution of an image by keeping only every L-th pixel along both spatial dimensions:
`I_out(r, c) = I_in(L*(r-1) + 1, L*(c-1) + 1)`

#### Physics of Aliasing and the Anti-Aliasing Shield
According to the Nyquist-Shannon sampling theorem, when shrinking a grid by deleting pixels (decimation), the new smaller grid cannot physically represent frequency components above its Nyquist frequency (half the new sampling rate). 

If you blindly delete pixels (decimation without filtering), these high-frequency components (such as sharp edges, fine lines, or noise) do not disappear. Instead, they mathematically fold backward (alias) across the Nyquist limit, permanently scrambling and polluting the lower frequencies.

To prevent this scrambling, an anti-aliasing low-pass filter must be placed immediately before the downsampling operator. This filter acts as a shield, mathematically blurring and removing all high-frequency details that cannot be represented in the smaller grid. This ensures that the downsampled base image remains pure and free from aliasing artifacts (like moire patterns). 

In a wavelet analysis filterbank, the complementary high-pass filter sits right next to the low-pass filter, catching the high-frequency details before they are lost, ensuring that the decomposition is perfectly invertible and zero information is lost overall.

---

## Class Reference

### Pixel Transformations (`OceanMath.Image.Pixels`)
* **`IPixelTransform`**: Abstract base class.
* **`TransformPipeline`**: Cell-array based composite pipeline.
* **`LinearOperator`**: Point operator scaling/bias class.
* **`GammaCorrection`**: Power-law adjustment class.
* **`LookupTableTransform`**: Fast lookup table indexing class.
* **`HistogramEqualization`**: Class for full and partial histogram equalization. Exposes computed `Histogram`, `CDF`, and `TransferFunction` as read-only properties.
* **`AdaptiveHistogramEqualization`**: Class for Contrast-Limited Adaptive Histogram Equalization (CLAHE) from scratch. Exposes `ClipLimit` and `NumTiles` properties.
* **`MinMaxStretch`**: Dynamic contrast range stretch class.
* **`MeanMatchStretch`**: Mean-centered luma alignment class.
* **`ColorSpaceConverter`**: Color space conversion strategy.
* **`BoundaryPadder`**: Unified boundary padding (border effects) strategy.
* **`SpatialFilter`**: General convolution filtering class using MATLAB defaults.
* **`CrossCorrelation`**: Applies direct spatial cross-correlation neighborhood filtering.
* **`TrueConvolution`**: Applies mathematical convolution by rotating the kernel 180 degrees.
* **`ZeroStuffing`**: Upsamples an image by inserting zeros between original pixel values.
* **`SynthesisFilter`**: Reconstructs approximation (LL) or detail (LH, HL, HH) subbands using wavelet synthesis filters.
* **`Decimation`**: Downsamples an image with optional anti-aliasing low-pass filtering. Supports both blind and anti-aliased downsampling modes.

### Filter Kernels (`OceanMath.Signal.Kernels`)
* **`IFilterKernel`**: Abstract interface for discrete 1D and 2D kernels.
* **`BoxKernel`**: Average box blur kernel generator.
* **`TriangleKernel`**: Bilinear/triangle blur kernel generator.
* **`GaussianKernel`**: Gaussian binomial kernel generator.
* **`SobelKernel`**: First-order directional Sobel derivative kernel generator.
* **`SecondDerivativeKernel`**: Second-order cross-derivative (corner) kernel generator.

---

## See Also
* `OceanMath.Geometry.CameraProjection` — Camera matrices and viewport projection.
* `OceanMath.Geometry.LensDistortion` — Optical distortion modeling.
