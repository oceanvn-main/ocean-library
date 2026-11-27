# Image Denoising Benchmark Recipes

Copy-pasteable code snippets for evaluating Box, Median, Gaussian, Bilateral, and Non-Local Means (NLM) filters.

## 1. Fast Box Filtering via Integral Images
```matlab
% Read image
I = im2double(imread('C:\Ocean Library\Matlab code practice\Data\ocean_clean.png'));
I_noisy = imnoise(I, 'gaussian', 0, 0.02);

% Box Filter
boxFilter = OceanMath.Image.Filters.BoxFilter('Radius', 3);
I_box = boxFilter.filter(I_noisy);

% Display results
figure;
subplot(1,2,1); imshow(I_noisy); title('Noisy');
subplot(1,2,2); imshow(I_box); title('Box Filtered');
```

## 2. Non-Linear Median Filter for Salt & Pepper Noise
```matlab
% Read image and add salt & pepper noise
I = im2double(imread('C:\Ocean Library\Matlab code practice\Data\ocean_clean.png'));
I_noisy = imnoise(I, 'salt & pepper', 0.05);

% Median Filter
medianFilter = OceanMath.Pixels.MedianFilter('Radius', 2);
I_median = medianFilter.filter(I_noisy);

% Display results
figure;
subplot(1,2,1); imshow(I_noisy); title('S&P Noisy');
subplot(1,2,2); imshow(I_median); title('Median Filtered');
```

## 3. Bilateral Filter vs. Non-Local Means (NLM)
```matlab
% Read image and add Gaussian noise
I = im2double(imread('C:\Ocean Library\Matlab code practice\Data\ocean_clean.png'));
I_noisy = imnoise(I, 'gaussian', 0, 0.015);

% Bilateral Filter (edge-preserving spatial & range weights)
bilat = OceanMath.Pixels.BilateralFilter('SpatialSigma', 2.0, 'RangeSigma', 0.12);
I_bilat = bilat.filter(I_noisy);

% Non-Local Means (patch-based similarity weights)
nlm = OceanMath.Image.Filters.NonLocalMeansFilter('SearchRadius', 5, 'PatchRadius', 2, 'H', 0.1);
I_nlm = nlm.filter(I_noisy);

% Display comparison
figure;
subplot(1,3,1); imshow(I_noisy); title('Noisy');
subplot(1,3,2); imshow(I_bilat); title('Bilateral');
subplot(1,3,3); imshow(I_nlm); title('Non-Local Means');
```

## See Also
- [Image Denoising Benchmark Guide](../guides/image_denoising.md)
- [Bilateral Filter Class](../../+OceanMath/+Image/+Filters/BilateralFilter.m)
- [Non-Local Means Filter Class](../../+OceanMath/+Image/+Filters/NonLocalMeansFilter.m)
