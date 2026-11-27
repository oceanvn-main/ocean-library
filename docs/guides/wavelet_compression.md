# Wavelet Compression Guide

## Overview

Exercise 3.21 benchmarks **entropy-based image compression** across three multiresolution representations:

1. **Laplacian Pyramid** — overcomplete (redundant), 4/3 × original pixel count, used as the baseline
2. **Haar Wavelet** (DWT via lifting scheme) — critically sampled, floating-point coefficients
3. **CDF 5/3 Wavelet** (DWT via integer lifting) — critically sampled, integer arithmetic, the wavelet family used in JPEG 2000 lossless mode

The central question is: *for a given visual quality (PSNR), how many bits per pixel does each representation require?* The answer is the **rate-distortion (R-D) curve**.

---

## Key Concept: Entropy as a Proxy for Compressed File Size

After decomposing an image into subband coefficients, we apply **uniform scalar quantization** (step `q`) and then measure the **Shannon entropy** of the quantized symbols. This entropy is the theoretical minimum number of bits an entropy coder (Huffman, arithmetic) could achieve — it is the lower bound on file size.

```
bpp = (1 / N_orig) * SUM_over_all_bands [ N_band * H(round(band / q)) ]
```

Where `H(·)` is the discrete Shannon entropy in bits (base 2).

### Why the Laplacian Pyramid is Penalised

The Laplacian pyramid is **overcomplete**: with 4 levels, it stores approximately 4/3 × N_orig pixels. Even though per-band entropies may be low, bpp is measured relative to N_orig — so the redundancy overhead is counted. This is why the pyramid consistently uses more bits than the critically-sampled wavelets.

---

## The Integer Lifting Contract for CDF 5/3

> **Critical rule**: CDF 5/3 lifting uses `floor()` in its predict and update steps, which means it is an **integer-to-integer transform**. It must be applied to integer-valued (or integer-rounded) signals.

If you apply CDF 5/3 directly to a `[0, 1]` floating-point image and then quantize with a small `q`, the coefficients are all near-zero fractions. `round(X/q)` then maps them to a near-uniform distribution — **maximum entropy, garbage PSNR**.

**Correct approach:**
```matlab
I_luma_int = round(I_luma * 255);           % scale to [0, 255] integer space
cdf53 = OceanMath.Signal.Lifting.LiftingScheme('WaveletType', 'CDF53', 'Levels', 4);
coeffs = cdf53.forward(I_luma_int);          % forward in integer space
% Quantize with q in gray-level units (e.g., 0.25 for 1/4 gray level)
% ... quantize, compute entropy, dequantize ...
I_recon_int = cdf53.inverse(coeffs_q);      % inverse in integer space
I_recon = min(max(I_recon_int / 255.0, 0), 1);  % scale back to [0,1]
```

Haar lifting does **not** have this constraint (it uses exact division `d/2`), so it works correctly in floating-point.

---

## Usage

### Running the Experiment

```matlab
run('Experiments/Scripts/Computer Vision/exp_wavelet_compression.m');
```

Outputs:
- `Results/wavelet_compression_results.png` — rate-distortion dashboard + visual comparison
- Console: per-band entropy audit at q = 1/4 gray level

### Per-Band Entropy Audit

The script performs an audit at the exercise-specified quantization step (q = 1/4 gray level):

```matlab
pyramid = OceanMath.Image.MultiResolution.LaplacianPyramid();
pyramid.build(I_luma, 4);
pyramid.report();  % print overcompleteness audit
```

---

## Typical Results (elon_musk.jpg, 3363×4602)

At q = 1/4 gray level:

| Method | bpp |
|---|---|
| Laplacian Pyramid (overcomplete) | 7.59 |
| Haar Wavelet (critically sampled) | 5.32 |
| CDF 5/3 Wavelet (critically sampled, integer) | 3.00 |

At moderate compression (q = 16/255, PSNR ~34 dB):

| Method | PSNR | bpp |
|---|---|---|
| Laplacian Pyramid | 34.4 dB | 0.75 |
| CDF 5/3 Wavelet | 34.1 dB | 0.29 |
| Haar Wavelet | 32.4 dB | 0.41 |

CDF 5/3 achieves roughly the same PSNR as the pyramid at less than **half the bit rate**, confirming why it is the wavelet of choice for JPEG 2000.

---

## See Also

- [Image Pyramids Guide](pyramids.md) — `GaussianPyramid`, `LaplacianPyramid`
- [Spatial Filtering & Lifting Guide](filtering_and_lifting.md) — `LiftingScheme` (Haar, CDF 5/3)
- [Wavelet Compression Recipes](../recipes/wavelet_compression_recipes.md) — copy-paste code
- `Experiments/Scripts/Computer Vision/exp_wavelet_compression.m` — full experiment
