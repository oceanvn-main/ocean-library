# Directional Domain Scaling Law Guide

## Overview

The Domain Scaling Law (textbook Equation 3.76) governs how spatial coordinate transformations affect a signal's frequency content. Under an affine transformation:
`x' = A * x`
The frequency spectrum of the transformed signal transforms according to:
`f' = A^-T * f`
where `A^-T` is the transpose of the inverse of the affine matrix `A`.

In a non-uniform scaling scenario where we stretch one axis (magnification) and squash the other (minification), the frequency scaling behavior is directional:
1. **Magnification (sx > 1)**: Stretches the signal in space, which compresses the frequency content (shifts peaks toward lower frequencies). This axis has no risk of aliasing.
2. **Minification (sy < 1)**: Squashes the signal in space, which expands the frequency content (shifts peaks toward higher frequencies). If a peak shifts beyond the output grid's Nyquist limit (0.5 cycles/pixel), it aliases and folds back into the spectrum.

This guide details the mathematical principles and explains how `exp_domain_scaling_direction.m` verifies these behaviors directionally in a single experiment.

---

## Mathematical Formulation

Let the continuous input signal be a 2D separable combination of two cosines:
`f(x, y) = cos(2 * pi * fx * x) + cos(2 * pi * fy * y)`

We apply a diagonal scaling transform:
`A = [sx, 0; 0, sy]`
with `sx > 1` (magnification along X) and `sy < 1` (minification along Y).

### 1. Spatial Transformation
Applying the warp `x' = A * x` means that to evaluate the transformed signal `g(x')` at target coordinate `x'`, we map back to the source coordinate using:
`x = A^-1 * x'`
which yields:
`g(x', y') = f(x' / sx, y' / sy) = cos(2 * pi * (fx / sx) * x') + cos(2 * pi * (fy / sy) * y')`

### 2. Frequency Shift
The continuous frequency components along each axis are:
- **X-axis target frequency**: `fx_new = fx / sx`
- **Y-axis target frequency**: `fy_new = fy / sy`

### 3. Sampling and Aliasing on the Output Grid
The output grid is sampled at integer intervals (spacing = 1.0), setting the Nyquist limit at `f_nyq = 0.5` cycles/pixel.

- Since `sx > 1`, we have:
  `fx_new = fx / sx < fx <= 0.5`
  This guarantees that the X-axis frequency is shifted further below the Nyquist limit, resulting in zero aliasing.
- Since `sy < 1`, we have:
  `fy_new = fy / sy > fy`
  If `fy` and `sy` are chosen such that `fy_new > 0.5`, the frequency exceeds the Nyquist limit. In the absence of an anti-aliasing prefilter (lowpass smoothing), this component aliases.
  The empirical peak in the spectrum folds back to:
  `fy_alias = |fy_new - round(fy_new)|`
  For example, if `fy = 0.25` and `sy = 0.4`, then `fy_new = 0.625`. The aliased peak appears at:
  `fy_alias = |0.625 - 1.0| = 0.375` cycles/pixel.

---

## Verification via FFT

To isolate the directional frequencies from the 2D output grid:
1. **X-axis Spectrum**: Extract a 1D row along the middle of the image, subtract the mean (removing the DC component), apply a Hann window (minimizing spectral leakage from the boundaries), and compute a zero-padded FFT.
2. **Y-axis Spectrum**: Identify the active (non-zero) coordinate range along the Y-axis, extract a column segment, subtract the mean, apply a Hann window, and compute a zero-padded FFT.

By searching for peak magnitudes in the positive frequency half `[0, 0.5]`, we confirm that:
- The empirical peak along X matches `fx_theory = fx / sx` exactly.
- The empirical peak along Y matches `fy_theory_alias = |fy/sy - round(fy/sy)|` exactly, confirming the folding-back effect of aliasing on the minified axis.

---

## See Also

* [Computer Vision Primitives Guide](computer_vision_primitives.md)
* [Signal Processing Guide](signal_processing.md)
