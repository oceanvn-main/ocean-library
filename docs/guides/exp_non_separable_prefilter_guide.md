# Non-Separability of Rotated Anisotropic Prefilters Guide

## Overview

In image warping and scaling, anti-aliasing requires prefiltering the source image with a low-pass filter whose footprint matches the back-projected Nyquist limit of the output grid.
* For axis-aligned, uniform scaling (zooms), the prefilter footprint is a circle or an axis-aligned square/ellipse. These filters are **separable**, meaning they can be factored and applied as a fast 1D horizontal pass followed by a 1D vertical pass.
* For non-uniform rotated/sheared transformations (non-zooms), the prefilter footprint is a **rotated ellipse**. Because the ellipse's primary axes are not aligned with the image grid, the corresponding 2D spatial kernel is **non-separable** and must be computed as a full 2D convolution, which is computationally expensive.

This guide explains the mathematics of prefilter non-separability, how to analyze it via Singular Value Decomposition (SVD), and the complexity scaling differences.

---

## Mathematical Formulation

### 1. Prefilter Covariance Matrix
Let the forward warp be `x' = A * x`. If we model the target-domain pixel footprint as a isotropic Gaussian with covariance `Sigma_tgt = sigma_p^2 * I`, the back-projected prefilter covariance in the source domain is:
`Sigma_src = A^-1 * Sigma_tgt * A^-T = sigma_p^2 * (A^T * A)^-1`

If we apply a rotated anisotropic scale transform:
`A = R * diag(s1, s2) * R'`
where `R` is a rotation matrix of angle `theta`, `s1` is the scale along the rotated axis, and `s2` is the scale along the perpendicular axis, then:
`Sigma_src = R * diag((sigma_p/s1)^2, (sigma_p/s2)^2) * R'`

When `theta` is not a multiple of 90 degrees and `s1 != s2`, the rotation matrix introduces off-diagonal terms:
`Sigma_src = [var_x, cov_xy; cov_xy, var_y]` with `cov_xy != 0`

### 2. Non-Separability of the 2D Kernel
The 2D spatial Gaussian prefilter is:
`G(x, y) = exp(-0.5 * [x, y] * Sigma_src^-1 * [x; y])`
Substituting `Sigma_src^-1 = [a, b; b, c]` with `b != 0`:
`G(x, y) = exp(-0.5 * (a*x^2 + 2*b*x*y + c*y^2))`
`G(x, y) = exp(-0.5*a*x^2) * exp(-b*x*y) * exp(-0.5*c*y^2)`

Because of the cross-term `exp(-b*x*y)`, the kernel **cannot** be factored into the form `g_x(x) * g_y(y)`. Thus, it is mathematically non-separable.

---

## Singular Value Decomposition (SVD) Rank Analysis

We can quantitatively analyze the separability of any discrete 2D kernel matrix `G` by computing its Singular Value Decomposition (SVD):
`G = U * S * V'`
where `S` is a diagonal matrix containing the singular values `s(1) >= s(2) >= ... >= s(N)`.

* **Separable Case**: The kernel has rank 1. Only the first singular value `s(1)` is non-zero, and the remaining singular values are zero:
  `s(2)/s(1) < 1e-12`
  The kernel can be factored exactly as the outer product of two 1D vectors: `g_v = U(:,1)*sqrt(s(1))` and `g_h = V(:,1)'*sqrt(s(1))`.
* **Non-Separable Case**: The kernel has rank > 1. Multiple singular values are significant:
  `s(2)/s(1) > 0.05`
  Forcing a separable approximation by keeping only the rank-1 component (`u1 * v1' * s(1)`) discards the other singular components, which changes the orientation of the blur and leaves residual high-frequency aliasing along the minified diagonal.

---

## Computational Complexity Analysis

Filtering an `W x H` image with a kernel of size `K x K` has different computational requirements depending on separability:

| Filter Type | Operations per Pixel (flops) | Complexity Class | For K = 15 |
| ----------- | ---------------------------- | ---------------- | ---------- |
| **Full 2D** | `2 * K^2`                    | `O(K^2)`         | 450 flops  |
| **Separable**| `4 * K`                      | `O(K)`           | 60 flops   |

For a 15x15 kernel, the separable pass provides a `7.5x` theoretical reduction in operations. As `K` increases, this discrepancy grows quadratically, demonstrating why non-separable filters are "very slow to compute" for large kernels.

---

## See Also

* [Directional Domain Scaling Guide](exp_domain_scaling_direction_guide.md)
* [Signal Processing Guide](signal_processing.md)
