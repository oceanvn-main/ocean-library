# Spatial Filtering & Computational Lifting Guide

## Overview

Neighborhood filtering and multi-resolution analysis are foundational building blocks in signal processing and computer vision. The `OceanMath` package separates these operations into dedicated namespaces:
* **Spatial Neighborhood Filtering**: Located under `OceanMath.Image.Filters`, utilizing the Template Method pattern to automate boundary padding and datatype management.
* **Dimensional Computational Lifting**: Located under `OceanMath.Image.Lifting`, providing the higher-dimensional splat-blur-slice pipeline (Bilateral Grid) for O(1) edge-preserving smoothing.
* **Algebraic Computational Lifting**: Located under `OceanMath.Signal.Lifting`, implementing Discrete Wavelet Transforms (DWT) via Split-Predict-Update steps for Haar and CDF 5/3 wavelets.

---

## Design Patterns & Core Engine

### The Template Method Pattern (`INeighborhoodFilter`)

Every spatial filter inherits from the base class `INeighborhoodFilter`. This base class orchestrates the filtering lifecycle:
1. Computes the necessary padding dimensions by invoking the subclass's `getPadSize()` hook.
2. Expands the input image boundaries using `BoundaryPadder.pad()`.
3. Invokes the subclass's abstract `applyFilter(I_padded, I_in)` hook to run the actual filter mathematics.
4. Clamps values and restores the output to the input image class type (e.g. `uint8`, `single`, `double`).

By abstracting these mechanical operations, individual filters are kept small, highly focused, and free of redundant padding or type-casting logic.

---

## Mathematical Formulations

### 1. Box Filter via Integral Images

A standard sliding-window average box filter of size K x K convolved with an H x W image takes O(H * W * K^2) operations. 
By utilizing an **Integral Image** (Summed-Area Table), we reduce the computational complexity to **O(H * W)**, which is completely independent of the window size K.

1. **Integral Image Computation**:
   Compute the cumulative sums along the rows and columns:
   `intImg(y, x) = sum_{j=1..y} sum_{i=1..x} I(j, i)`
   This is computed efficiently in O(H * W) using cumulative sums.

2. **O(1) Region Sum**:
   The sum of any arbitrary rectangular area defined by corners (y1, x1) and (y2, x2) in the padded image is evaluated in constant time using exactly 4 table lookups:
   `windowSum = intImg(y2, x2) - intImg(y1-1, x2) - intImg(y2, x1-1) + intImg(y1-1, x1-1)`

---

### 2. Median Filter

The median filter is a non-linear neighborhood operator where each pixel's intensity is replaced by the median value in its local window.
It is optimal for removing impulse (salt-and-pepper) noise because the median statistic is robust against extreme outliers, unlike the average (box) filter which smears outliers across the window.

---

### 3. Bilateral Filter

The bilateral filter is a non-linear, edge-preserving smoothing operator. The filtered intensity at pixel `p` is a normalized weighted average of its neighbors `q`:
`I_out(p) = (1 / W_p) * sum_{q in S} G_s(||p - q||) * G_r(I(p) - I(q)) * I(q)`
Where:
* `G_s` is the spatial Gaussian weight (closeness in coordinates).
* `G_r` is the range Gaussian weight (similarity in intensity).
* `W_p` is the normalization factor: `sum_{q in S} G_s(||p - q||) * G_r(I(p) - I(q))`.

Because `G_r` depends on the dynamic pixel values, the filter is non-separable and non-linear, requiring a direct spatial search of O(H * W * K^2).

---

### 4. Bilateral Grid (Dimensional Lifting)

To achieve fast, constant-time edge-aware filtering, we lift the 2D image coordinates (y, x) and intensities I(y, x) into a 3D grid space:
`GridCoord(y, x) = ( [y / s_s], [x / s_s], [I(y, x) / s_r] )`
Where `s_s` and `s_r` are the spatial and range sampling steps.

The Bilateral Grid algorithm operates in three steps:
1. **Splatting**: Accumulate pixel intensities and interpolation weights into a 3D grid, creating a data grid `G_D` and weight grid `G_W`. Trilinear interpolation weights are used to distribute values to the 8 surrounding grid nodes.
2. **Blurring**: Convolve `G_D` and `G_W` with a 3D Gaussian kernel. Since the range dependency is now linearized in the grid space, the operation is completely linear. We perform it via three successive 1D Gaussian blurs along the row, column, and range dimensions of the grid.
3. **Slicing**: Extract the smoothed values and weights back to the 2D image plane by performing trilinear interpolation at continuous coordinates `(y/s_s, x/s_s, I(y,x)/s_r)` for each pixel. Divide the interpolated data by the weights.

This reduces the complexity of bilateral filtering to O(H * W) plus a small grid convolution overhead, enabling real-time edge-preserving smoothing on high-resolution images.

---

### 5. Wavelet Lifting Scheme

Traditional Discrete Wavelet Transforms (DWT) convolve signals with long low-pass and high-pass filter banks.
The **Lifting Scheme** factors the wavelet filters into a series of elementary lifting steps: Split, Predict, and Update.

1. **Split**: Split the input signal `x` into even and odd samples:
   `even = x(1:2:end)`
   `odd = x(2:2:end)`
2. **Predict**: Predict the odd samples from the even samples, replacing the odd samples with the prediction error (details `d`):
   * Haar: `d[n] = odd[n] - even[n]`
   * CDF 5/3: `d[n] = odd[n] - floor((even[n] + even[n+1]) / 2)`
3. **Update**: Update the even samples using the details `d` to preserve the average/low-frequency properties of the original signal (approximation `s`):
   * Haar: `s[n] = even[n] + d[n] / 2`
   * CDF 5/3: `s[n] = even[n] + floor((d[n-1] + d[n] + 2) / 4)`

This allows:
* **In-place computation**: Computations are performed directly in-place without auxiliary arrays.
* **Lossless mapping**: CDF 5/3 maps integers to integers, enabling perfectly lossless reconstruction during the inverse transform.
