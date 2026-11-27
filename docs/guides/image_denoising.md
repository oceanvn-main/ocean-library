# Image Denoising Benchmark Guide (Exercise 3.28)

Overview and evaluation of linear and non-linear image denoising filters.

## Overview
This guide covers the implementation and benchmarking of five key spatial filtering techniques for image denoising:
- **Box Filter**: A simple linear low-pass average filter, implemented in O(1) time using integral images.
- **Gaussian Filter**: A linear filter that weights pixels by distance from the center, modeling natural blur.
- **Median Filter**: A non-linear rank filter that excels at removing Salt & Pepper noise while preserving edges.
- **Bilateral Filter**: An edge-preserving non-linear filter that weights pixels by spatial distance and intensity similarity.
- **Non-Local Means (NLM) Filter**: An advanced patch-based filter that exploits spatial redundancy across the entire image.

## Usage
The filters are available in the `+OceanMath/+Image/+Filters` and `+OceanMath/+Pixels` packages:

```matlab
% Instantiate the filters
boxFilter = OceanMath.Image.Filters.BoxFilter('Radius', 2);
gaussianFilter = OceanMath.Image.Filters.GaussianFilter('Sigma', 1.5);
medianFilter = OceanMath.Pixels.MedianFilter('Radius', 2);
bilateralFilter = OceanMath.Pixels.BilateralFilter('SpatialSigma', 2.0, 'RangeSigma', 0.15);
nlmFilter = OceanMath.Image.Filters.NonLocalMeansFilter('SearchRadius', 6, 'PatchRadius', 2, 'H', 0.1);
```

To run a full sweep comparing these filters under Additive White Gaussian Noise (AWGN) and Salt & Pepper (S&P) noise, run:
```matlab
exp_image_denoising;
```

## Examples

### 1. Applying a Bilateral Filter
```matlab
% Read input image
I = im2double(imread('C:\Ocean Library\Matlab code practice\Data\ocean_clean.png'));
I_noisy = imnoise(I, 'gaussian', 0, 0.01);

% Apply edge-preserving bilateral filtering
bilat = OceanMath.Pixels.BilateralFilter('SpatialSigma', 2.0, 'RangeSigma', 0.1);
I_denoised = bilat.filter(I_noisy);
```

### 2. Applying Non-Local Means (NLM) Denoising
```matlab
% Set up patch-based search parameters
nlm = OceanMath.Image.Filters.NonLocalMeansFilter('SearchRadius', 5, 'PatchRadius', 2, 'H', 0.12);
I_denoised = nlm.filter(I_noisy);
```

## See Also
- [Pixel Transformations Guide](pixel_transforms.md)
- [Spatial Filtering & Lifting Guide](filtering_and_lifting.md)
- [Local Laplacian Filter Guide](local_laplacian.md)
