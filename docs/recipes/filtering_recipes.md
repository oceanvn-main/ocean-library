# Spatial Filtering & Computational Lifting Recipes

Recipes for instantiating and applying spatial filters, Bilateral Grids, and the Wavelet Lifting Scheme.

---

## 1. Fast Box Filtering via Integral Images

```matlab
% Import the BoxFilter strategy
import OceanMath.Image.Filters.BoxFilter

% Load input image
I = imread('Data/images/ocean_img.jpg');

% Instantiate a 7x7 Box Filter with replicate boundary padding
% (The computation runs in constant O(1) time per pixel via Integral Images)
box_filt = BoxFilter('Size', [7, 7], 'Boundary', 'replicate');

% Apply filter
I_smoothed = box_filt.apply(I);

% Display results
figure;
subplot(1, 2, 1); imshow(I); title('Original');
subplot(1, 2, 2); imshow(I_smoothed); title('7x7 Box Filtered');
```

---

## 2. Removing Salt-and-Pepper Noise via Median Filter

```matlab
% Import the MedianFilter strategy
import OceanMath.Image.Filters.MedianFilter

% Load and corrupt image with Salt-and-Pepper noise
I = imread('Data/images/ocean_img.jpg');
I_noisy = imnoise(I, 'salt & pepper', 0.05);

% Instantiate a 5x5 Median Filter
med_filt = MedianFilter('Size', 5, 'Boundary', 'symmetric');

% Apply filter
I_denoised = med_filt.apply(I_noisy);

% Display results
figure;
subplot(1, 3, 1); imshow(I); title('Original');
subplot(1, 3, 2); imshow(I_noisy); title('Noisy (5% S&P)');
subplot(1, 3, 3); imshow(I_denoised); title('5x5 Median Filtered');
```

---

## 3. Edge-Preserving Smoothing via Bilateral Filter

```matlab
% Import the BilateralFilter strategy
import OceanMath.Image.Filters.BilateralFilter

% Load image and normalize to double
I = double(imread('Data/images/ocean_img.jpg')) / 255.0;

% Instantiate a Bilateral Filter
% Method Options:
%   'direct' - Vectorized direct neighborhood search
%   'grid'   - Fast 3D Bilateral Grid (Dimensional Lifting)
bilat_filt = BilateralFilter('SpatialSigma', 5.0, ...
                             'RangeSigma', 0.15, ...
                             'Method', 'grid', ...
                             'SpatialSampling', 4.0, ...
                             'RangeSampling', 0.05, ...
                             'Boundary', 'replicate');

% Apply filter
I_edge_preserved = bilat_filt.apply(I);

% Display results
figure;
subplot(1, 2, 1); imshow(I); title('Original');
subplot(1, 2, 2); imshow(I_edge_preserved); title('Bilateral Grid Filtered');
```

---

## 4. Multi-Resolution Wavelet Denoising via Lifting Scheme

```matlab
% Import the Wavelet Lifting Scheme
import OceanMath.Signal.Lifting.LiftingScheme

% Load image and normalize
I = double(imread('Data/images/ocean_img.jpg')) / 255.0;
I_noisy = I + 0.05 * randn(size(I));

% Instantiate a 2-Level CDF 5/3 Lifting Scheme
wavelet = LiftingScheme('WaveletType', 'CDF53', 'Levels', 2);

% 1. Forward Transform (returns subbands struct)
coeffs = wavelet.forward(I_noisy);

% 2. Denoise by applying soft thresholding to the detail subbands
threshold = 0.06;
for lvl = 1:coeffs.Levels
    % Access LH, HL, HH subbands at each level
    coeffs.Levels{lvl}.LH = sign(coeffs.Levels{lvl}.LH) .* max(0, abs(coeffs.Levels{lvl}.LH) - threshold);
    coeffs.Levels{lvl}.HL = sign(coeffs.Levels{lvl}.HL) .* max(0, abs(coeffs.Levels{lvl}.HL) - threshold);
    coeffs.Levels{lvl}.HH = sign(coeffs.Levels{lvl}.HH) .* max(0, abs(coeffs.Levels{lvl}.HH) - threshold);
end

% 3. Inverse Transform (Reconstruction)
I_denoised = wavelet.inverse(coeffs);
I_denoised = max(0.0, min(1.0, I_denoised)); % clamp bounds

% Display results
figure;
subplot(1, 3, 1); imshow(I_noisy); title('Noisy Input');
subplot(1, 3, 2); imshow(coeffs.LL / max(coeffs.LL(:))); title('Final LL Subband');
subplot(1, 3, 3); imshow(I_denoised); title('Wavelet Denoised');
```
