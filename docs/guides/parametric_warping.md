# Parametric Image Warping & Texture Filtering Guide

## Overview

Image warping maps pixels from a source coordinate system $(u, v)$ to a target screen coordinate system $(x', y')$. This guide covers the mathematical formulations and implementations for:
1. **Affine and Projective (Homography) Warps**: Inverse mapping using different interpolants (Nearest, Bilinear, Bicubic).
2. **Isotropic MIP-Mapping**: Pre-filtered Gaussian pyramids to prevent aliasing under minification, analyzing the trade-offs of adjacent level selection.
3. **Anisotropic Texture Filtering**: Using the Elliptical Weighted Average (EWA) filter to handle highly skewed footprint ratios.

---

## 1. Inverse Mapping & Interpolation

Warping an image using forward mapping (copying source pixels to calculated target destinations) leaves holes and overlap artifacts due to discretization. Therefore, we use **inverse mapping**:

$$\begin{bmatrix} u \cdot w \\ v \cdot w \\ w \end{bmatrix} = \mathbf{H}^{-1} \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix}$$

For each target coordinate $(x', y')$, we compute the source coordinate $(u, v)$ and interpolate the source image.

### Interpolants and Quality

* **Nearest Neighbor**: Fast, but introduces jagged edges (aliasing) and pixelation.
* **Bilinear**: Computes a weighted average of the 4 surrounding pixels. Smooths out high frequencies but blurs detail.
* **Bicubic (Keys' kernel)**: Uses a 16-pixel neighborhood with a cubic spline kernel. Preserves sharpness and edges, with minimal ringing.

---

## 2. MIP-Mapping and Level Selection

A **MIP-map** is a pre-calculated sequence of low-pass filtered and downsampled images (Gaussian pyramid). For a screen pixel, the footprint in texture space can be approximated by a circle of diameter $2^d$, where $d$ is the fractional MIP-map level computed from the Jacobian matrix $\mathbf{J}$ of the mapping:

$$d = \log_2(\sigma_{\max}(\mathbf{J}))$$

Where $\sigma_{\max}(\mathbf{J})$ is the largest singular value of the local Jacobian matrix $\mathbf{J} = \frac{\partial(u,v)}{\partial(x,y)}$.

### Coarser vs. Finer Level Selection

When the target scale lands on a fractional level $d$ (e.g., $d = 1.5$):
* **Finer Level Selection** ($d_{low} = \lfloor d \rfloor$): The pre-filter size is too small for the actual minification. High-frequency content above the Nyquist limit survives, causing **aliasing** (shimmering, jagged borders).
* **Coarser Level Selection** ($d_{high} = \lceil d \rceil$): The pre-filter is too large. It removes valid frequencies below the Nyquist limit, resulting in an **over-blurred** image.
* **Trilinear Blending**: A linear interpolation between the bilinear samples of the adjacent coarser and finer levels. This prevents level-jumping "popping" artifacts and balances blur against aliasing.

---

## 3. Anisotropic Filtering and the EWA Sampler

In perspective projections (e.g., a tilted ground plane), a circular screen pixel maps to a highly elongated **ellipse** in texture space (anisotropic).

### The Anisotropic Problem
Isotropic MIP-mapping assumes a circular footprint. To avoid aliasing, it must choose a level based on the *longest* axis of the ellipse (the highly minified direction towards the horizon). This severely **over-blurs** the short axis (the uncompressed direction), washing out detail along the horizon.

### The EWA Solution
The **Elliptical Weighted Average (EWA)** filter calculates the exact covariance matrix of the mapped ellipse using the local Jacobian:

$$\mathbf{C} = \mathbf{J} \mathbf{J}^T + \mathbf{I}$$

It then sums texels inside the elliptical footprint weighted by a lookup table of a radial Gaussian profile:

$$w(d) = e^{-\frac{1}{2} d^T \mathbf{C}^{-1} d}$$

This achieves high sharpness along the uncompressed axis while perfectly anti-aliasing the compressed axis.

---

## Usage Example

### Running the Experiment
To execute the visual benchmark and print the discussion:
```matlab
run('Experiments/Scripts/Computer Vision/exp_parametric_warping.m');
```

---

## See Also
* [Image Pyramids Guide](pyramids.md) — `GaussianPyramid` representation.
* [Computer Vision Primitives Guide](computer_vision_primitives.md) — Coordinate transformations and projective geometry.
* [Spatial Filtering Guide](filtering_and_lifting.md) — Bilateral filters and windowed sinc interpolation.
