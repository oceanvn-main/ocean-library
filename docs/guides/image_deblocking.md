# Image Deblocking Guide

## Overview

This guide covers the `+OceanMath/+Image/+Compression/` and related packages that implement **Exercise 4.3: Image Deblocking via POCS and Guided Restoration** (Szeliski §4.3).

The pipeline addresses two core problems:

1. **JPEG Blocking Artifacts**: Visible 8x8 tile boundaries caused by independent DCT quantization per block.
2. **Chrominance Subsampling Degradation**: Loss of chroma resolution in YCbCr 4:2:0 encoding.

### Key Mathematical Insight

JPEG deblocking is a constrained recovery problem. Given quantized DCT index `q(i,j)` and step size `Q(i,j)`, the true DCT coefficient must lie in the convex interval:

```
c in [ Q(i,j) * (q(i,j) - 0.5),  Q(i,j) * (q(i,j) + 0.5) ]
```

POCS (Projection onto Convex Sets) iterates between a spatial smoother and this DCT projection, converging to a solution consistent with all constraints.

---

## Architecture

```
JpegSyntheticCompressor   — Software DCT encoder: compress() / decompress()
          |
          | q_all, Q_table
          v
PocsDeblockingEngine      — POCS loop: smooth → DCT project → repeat
          |
          | deblocked YCbCr image
          v
ChromaSubsampler          — subsample() → upsample() via GuidedFilter / JBF
          |
          v
ImageQualityMetrics       — psnr() / ssim() / printTable()
```

---

## Usage

### Step 1 — Generate or Load a Clean Image

```matlab
% Synthetic test image (no file needed)
gen = OceanData.Generators.SyntheticImageGenerator( ...
    'Height', 256, 'Width', 256, 'Mode', 'mixed', 'Seed', 42);
img_clean = gen.generate();  % (256 x 256 x 3) double [0,1]

% Or load a real image
img_clean = double(imread('path/to/image.jpg')) / 255.0;
```

### Step 2 — Compress with Software JPEG

```matlab
comp = OceanMath.Image.Compression.JpegSyntheticCompressor('QualityFactor', 10);
[img_compressed, q_all, Q_table] = comp.compress(img_clean);

% View quantization table
disp(Q_table)
```

### Step 3 — POCS Deblocking

```matlab
engine = OceanMath.Image.Compression.PocsDeblockingEngine( ...
    'MaxIter',   30, ...
    'Tolerance', 1e-4, ...
    'Verbose',   true);

% Operates on double [0, 1] image (RGB or YCbCr)
[img_deblocked, history] = engine.deblock(img_compressed, q_all, Q_table);

% Plot convergence
plot(history.RMSE);
xlabel('Iteration'); ylabel('Relative Change');
```

### Step 4 — Chrominance Subsampling + Guided Upsampling

```matlab
% Convert to YCbCr
YCbCr = rgb2ycbcr(img_deblocked);

% Subsample Cb and Cr (4:2:0)
sub = OceanMath.Image.Compression.ChromaSubsampler( ...
    'Scheme', '4:2:0', 'UpsamplingMethod', 'guided');
[Y, Cb_sub, Cr_sub] = sub.subsample(YCbCr);

% Reconstruct with Guided Filter (Y as guidance)
YCbCr_restored = sub.upsample(Y, Cb_sub, Cr_sub);
img_restored = ycbcr2rgb(YCbCr_restored);
```

### Step 5 — Quality Evaluation

```matlab
metrics(1).Label = 'JPEG Compressed';
metrics(1).PSNR  = OceanStats.Image.ImageQualityMetrics.psnr(img_compressed, img_clean);
metrics(1).SSIM  = OceanStats.Image.ImageQualityMetrics.ssim(img_compressed, img_clean);

metrics(2).Label = 'POCS Deblocked';
metrics(2).PSNR  = OceanStats.Image.ImageQualityMetrics.psnr(img_deblocked, img_clean);
metrics(2).SSIM  = OceanStats.Image.ImageQualityMetrics.ssim(img_deblocked, img_clean);

OceanStats.Image.ImageQualityMetrics.printTable(metrics);
```

---

## Classes Reference

### `JpegSyntheticCompressor`

| Property | Default | Description |
|---|---|---|
| `QualityFactor` | `50` | JPEG quality [1=worst, 100=best]. Scales Annex K luma table. |
| `QTable` | `[]` | Custom 8x8 quantization table override. |
| `BlockSize` | `8` | DCT block size. |

**Key methods**: `compress(img)`, `decompress(q_all, Q_table)`, `getQTable()`

---

### `PocsDeblockingEngine`

| Property | Default | Description |
|---|---|---|
| `MaxIter` | `30` | Maximum POCS iterations. |
| `Tolerance` | `1e-4` | Convergence threshold (relative image change). |
| `Smoother` | `BilateralFilter` | Pluggable `INeighborhoodFilter`. Set to `[]` to disable. |
| `SmoothEvery` | `1` | Apply smoother every N iterations. |
| `Verbose` | `true` | Print iteration trace. |

**Key method**: `deblock(img, q_all, Q_table)` → `{img_out, history}`

#### Custom Smoother Example

```matlab
% Use Gaussian blur instead of Bilateral
engine.Smoother = OceanMath.Image.Filters.GaussianFilter('Sigma', 0.5);

% Or disable smoothing for pure POCS projection
engine.Smoother = [];
```

---

### `GuidedFilter`

O(N) Guided Image Filter (He et al. 2013) using integral image box filters.

| Property | Default | Description |
|---|---|---|
| `Radius` | `8` | Half-width of local window (window = 2r+1). |
| `Epsilon` | `0.01` | Regularization preventing over-smoothing. |

```matlab
gf = OceanMath.Image.Filters.GuidedFilter('Radius', 8, 'Epsilon', 0.01);

% Self-guided smoothing (edge-preserving)
p_smooth = gf.apply(p_noisy);

% Cross-guided upsampling (e.g., sparse chroma guided by sharp luma)
chroma_up = gf.applyGuided(chroma_nn, luma_Y);
```

---

### `JointBilateralFilter`

Bilateral Filter where range weights come from a separate guidance image.

| Property | Default | Description |
|---|---|---|
| `SpatialSigma` | `5.0` | Spatial Gaussian sigma (pixels). |
| `RangeSigma` | `0.15` | Range Gaussian sigma (intensity [0,1]). |

```matlab
jbf = OceanMath.Image.Filters.JointBilateralFilter( ...
    'SpatialSigma', 5, 'RangeSigma', 0.15);
chroma_up = jbf.applyGuided(chroma_nn, luma_Y);
```

---

### `ChromaSubsampler`

| Property | Default | Options |
|---|---|---|
| `Scheme` | `'4:2:0'` | `'4:4:4'`, `'4:2:2'`, `'4:2:0'` |
| `UpsamplingMethod` | `'guided'` | `'guided'`, `'jbf'`, `'nearest'` |
| `GuidedRadius` | `8` | — |
| `GuidedEpsilon` | `0.01` | — |

---

### `ImageQualityMetrics`

All static — no instantiation needed.

```matlab
psnr_val = OceanStats.Image.ImageQualityMetrics.psnr(img, ref);   % dB
ssim_val = OceanStats.Image.ImageQualityMetrics.ssim(img, ref);   % [0,1]
OceanStats.Image.ImageQualityMetrics.printReport(img, ref, 'My Method');
OceanStats.Image.ImageQualityMetrics.printTable(metrics_struct_array);
```

---

## Examples

### Run the Full Exercise 4.3 Pipeline

```matlab
% Run the complete Ex 4.3 experiment (synthetic mode)
run('Experiments/Scripts/Computer Vision/exp_image_deblocking.m')

% With a real image: edit IMAGE_SOURCE inside the script first
% IMAGE_SOURCE = 'c:/Ocean Library/Matlab code practice/Data/images/consu.jpg';
```

### Guided Filter as General Edge-Preserving Smoother

```matlab
gf = OceanMath.Image.Filters.GuidedFilter('Radius', 4, 'Epsilon', 0.001);
img_smooth = gf.apply(img_noisy);  % Strong edge-preserving denoising
```

### Custom Quantization Table for DCT Experiments

```matlab
% Create an extremely coarse Q table (uniform step = 50)
comp = OceanMath.Image.Compression.JpegSyntheticCompressor( ...
    'QTable', 50 * ones(8,8));
[img_degraded, q_all, Q_table] = comp.compress(img_clean);
```

---

## See Also

- `exp_image_deblocking.m` — Full Exercise 4.3 experiment script
- `exp_color_jpeg_spaces.m` — JPEG quality comparison (color spaces)
- `BilateralFilter` — Edge-preserving bilateral filter (default POCS smoother)
- `OceanViz.Strategies.DeblockingStrategy` — 6-panel deblocking dashboard
- Szeliski §4.3, p. 234 — POCS deblocking theory
