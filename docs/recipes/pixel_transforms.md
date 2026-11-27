# Pixel Transformations Recipes

Copy-pasteable code blocks for implementing modular image transformations and spatial filtering using `OceanMath.Image.Pixels`.

---

## 1. Contrast, Brightness, and Gamma Adjustments
This recipe applies point operators (linear scaling and power-law gamma) to adjust the contrast, brightness, and luminance of an image.

```matlab
% 1. Import transformation strategies
import OceanMath.Image.Pixels.LinearOperator
import OceanMath.Image.Pixels.GammaCorrection

% 2. Create sample image (intensity values [0, 1])
I = [0.1, 0.2; 0.3, 0.4];

% 3. Apply scale (contrast) and bias (brightness)
% Formula: I_out = 1.5 * I - 0.05 (clipped to [0, 1])
linOp = LinearOperator('Scale', 1.5, 'Bias', -0.05, 'Clip', true);
I_linear = linOp.apply(I);

% 4. Apply non-linear Gamma Correction
% Formula: I_out = 1.0 * I^0.5
gammaOp = GammaCorrection('Gamma', 0.5, 'C', 1.0);
I_gamma = gammaOp.apply(I);

disp('Linear adjustment result:');
disp(I_linear);
disp('Gamma adjustment result:');
disp(I_gamma);
```

---

## 2. Color Balancing (Linear vs. sRGB Space)
This recipe performs channel-wise color balancing. It demonstrates how to apply gains directly in sRGB/encoded space, and how to decode to linear space using gamma correction first to obtain mathematically correct blending.

```matlab
% 1. Import strategy
import OceanMath.Image.Pixels.ColorBalance

% 2. Create a test color image [2 x 2 x 3]
I = cat(3, [0.5, 0.6; 0.7, 0.8], [0.4, 0.5; 0.6, 0.7], [0.3, 0.4; 0.5, 0.6]);

% 3. Apply color balance gains directly in sRGB/encoded space
balance_srgb = ColorBalance('RedGain', 1.2, 'GreenGain', 1.0, 'BlueGain', 0.8, 'ApplyGammaFirst', false);
I_srgb = balance_srgb.apply(I);

% 4. Apply color balance gains in linear space (Gamma decode -> scale -> Gamma encode)
balance_linear = ColorBalance('RedGain', 1.2, 'GreenGain', 1.0, 'BlueGain', 0.8, 'Gamma', 2.2, 'ApplyGammaFirst', true);
I_linear = balance_linear.apply(I);

disp('sRGB balancing result (Red channel):');
disp(I_srgb(:,:,1));
disp('Linear space balancing result (Red channel):');
disp(I_linear(:,:,1));
```

---

## 3. Dynamic Min-Max and Mean-Match Stretching
This recipe dynamically rescales image intensities based on the image's own min, max, and mean statistics.

```matlab
import OceanMath.Image.Pixels.MinMaxStretch
import OceanMath.Image.Pixels.MeanMatchStretch

% 1. Create a low-contrast test image
I = [0.4, 0.45; 0.5, 0.55];

% 2. Stretch the intensities to fill the full [0, 1] range
minMax = MinMaxStretch('TargetRange', [0, 1]);
I_stretched = minMax.apply(I);

% 3. Center the intensities around a target mean and adjust scale
meanMatch = MeanMatchStretch('TargetMean', 0.5, 'Scale', 2.0);
I_centered = meanMatch.apply(I);

disp('Min-Max Stretched image:');
disp(I_stretched);
disp('Mean-Match Stretched image:');
disp(I_centered);
```

---

## 4. Full and Partial Histogram Equalization
This recipe performs global histogram equalization on an 8-bit luma image, demonstrating how to extract the computed histogram, CDF, and mapping transfer functions, and how to blend the equalized mapping with the identity mapping (partial equalization).

```matlab
import OceanMath.Image.Pixels.HistogramEqualization

% 1. Create a low-contrast 8-bit image
I = uint8([50, 100; 100, 150]);

% 2. Apply full equalisation (Alpha = 1.0)
eqFull = HistogramEqualization('Alpha', 1.0);
I_equalized = eqFull.apply(I);

% 3. Extract internal properties
hist = eqFull.Histogram;           % 256x1 counts
cdf = eqFull.CDF;                 % 256x1 cumulative sum
transferFunc = eqFull.TransferFunction; % 256x1 map

% 4. Apply partial equalisation (Alpha = 0.5)
eqPartial = HistogramEqualization('Alpha', 0.5);
I_partial = eqPartial.apply(I);

disp('Equalized 8-bit image:');
disp(I_equalized);
disp('Partial equalized 8-bit image:');
disp(I_partial);
```

---

## 5. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
This recipe performs adaptive, localized histogram equalization on an 8-bit luma image using a grid of contextual tiles and bilinear interpolation. It also demonstrates how to enable bilinear accumulation of histogram counts and spatial low-pass filtering of CDFs (Szeliski Exercise 3.8).

```matlab
import OceanMath.Image.Pixels.AdaptiveHistogramEqualization

% 1. Create a 16x16 gradient test image
[X, Y] = meshgrid(1:16, 1:16);
I = uint8(X * 10 + Y * 5);

% 2. Instantiate CLAHE with standard options
clahe = AdaptiveHistogramEqualization('NumTiles', [8, 8], 'ClipLimit', 0.05);

% 3. Apply standard CLAHE
I_clahe = clahe.apply(I);

% 4. Instantiate with Bilinear Accumulation and Spatial CDF Smoothing (Exercise 3.8 options)
clahe_ex38 = AdaptiveHistogramEqualization(...
    'NumTiles', [8, 8], ...
    'ClipLimit', 0.05, ...
    'BilinearAccumulate', true, ...
    'FilterCDFSpatial', true, ...
    'FilterSigma', 1.2);

% 5. Apply the advanced CLAHE transformation
I_clahe_ex38 = clahe_ex38.apply(I);

disp('Standard CLAHE processed 8-bit image corner:');
disp(I_clahe(1:4, 1:4));
disp('Advanced CLAHE (Bilinear Accum + Filtered CDF) corner:');
disp(I_clahe_ex38(1:4, 1:4));
```

---

## 6. Pipeline Composition: Color Convert -> Filter -> Stretch
This recipe chains multiple strategies into a unified processing pipeline using `TransformPipeline` (Pipes and Filters pattern). It converts an RGB image to grayscale, applies a box blur, and stretches the output dynamic range.

```matlab
import OceanMath.Image.Pixels.ColorSpaceConverter
import OceanMath.Image.Pixels.SpatialFilter
import OceanMath.Image.Pixels.MinMaxStretch
import OceanMath.Image.Pixels.TransformPipeline

% 1. Define a random color image [100 x 100 x 3]
I_rgb = rand(100, 100, 3);

% 2. Define composite pipeline
pipeline = TransformPipeline({ ...
    ColorSpaceConverter('RGB2Gray'), ...
    SpatialFilter('Kernel', 1/9 * ones(3,3), 'Boundary', 'replicate'), ...
    MinMaxStretch('TargetRange', [0, 1]) ...
});

% 3. Execute all transformations sequentially
I_out = pipeline.apply(I_rgb);

fprintf('Output size   : %dx%d\n', size(I_out, 1), size(I_out, 2));
fprintf('Output limits : min = %.4f, max = %.4f\n', min(I_out(:)), max(I_out(:)));
```

---

## 7. Neighborhood Linear Filtering: Cross-Correlation vs. True Convolution
This recipe demonstrates the mathematical difference between direct spatial cross-correlation and rotated mathematical convolution using a directional shift kernel.

```matlab
% 1. Import strategies
import OceanMath.Image.Pixels.CrossCorrelation
import OceanMath.Image.Pixels.TrueConvolution

% 2. Define an asymmetric impulse image
I = [ 0,  0,  0,  0,  0;
      0,  0,  0,  0,  0;
      0,  0, 10,  0,  0;
      0,  0,  0,  0,  0;
      0,  0,  0,  0,  0 ];

% 3. Shift kernel (1.0 on the right side of the center)
% - Cross-Correlation shifts the pixel LEFT.
% - True Convolution rotates the kernel 180 degrees first, shifting the pixel RIGHT.
K = [ 0.0,  0.0,  0.0;
      0.0,  0.0,  1.0;
      0.0,  0.0,  0.0 ];

xcorr_op = CrossCorrelation('Kernel', K, 'Boundary', 0);
I_xcorr = xcorr_op.apply(I);

conv_op = TrueConvolution('Kernel', K, 'Boundary', 0);
I_conv = conv_op.apply(I);

disp('Original Image (Impulse at center):');
disp(I);
disp('Cross-Correlation output (Shifted LEFT):');
disp(I_xcorr);
disp('True Convolution output (Shifted RIGHT):');
disp(I_conv);
```

---

## 8. Boundary Padding (Border Effects)
This recipe expands image boundaries by 2 pixels in all directions using various padding modes.

```matlab
import OceanMath.Image.Pixels.BoundaryPadder

% 1. Define a 3x3 test matrix
I = [ 1, 2, 3;
      4, 5, 6;
      7, 8, 9 ];

% 2. Constant value padding
I_const = BoundaryPadder.pad(I, [2, 2], 'constant', 0);

% 3. Clamp (replicate) padding
I_clamp = BoundaryPadder.pad(I, [2, 2], 'clamp');

% 4. Wrap (circular) padding
I_wrap = BoundaryPadder.pad(I, [2, 2], 'wrap');

% 5. Mirror (reflect) padding
I_mirror = BoundaryPadder.pad(I, [2, 2], 'mirror');

% 6. Extend padding (linear extrapolation)
I_extend = BoundaryPadder.pad(I, [2, 2], 'extend');

disp('Extend (linear extrapolation) padded image:');
disp(I_extend);
```

---

## 9. Decoupled Filter Kernels & Separable Convolutions
This recipe shows how to instantiate standard parameterized filter kernels (Box, Gaussian, Sobel, Corner) and apply them using `TrueConvolution` or `CrossCorrelation`. The strategies bypass Singular Value Decomposition (SVD) and run sequential horizontal and vertical 1D filtering directly.

```matlab
% 1. Import strategies and kernels
import OceanMath.Image.Pixels.TrueConvolution
import OceanMath.Signal.Kernels.*

% 2. Generate a random input image
I = rand(64, 64);

% 3. Instantiate various decoupled kernels
k_box = BoxKernel(5);
k_gauss = GaussianKernel('Size', 5, 'Sigma', 1.0);
k_sobel = SobelKernel('Direction', 'horizontal', 'Size', 3);
k_corner = SecondDerivativeKernel(3);

% 4. Create separable convolving operators
% Note: passing IFilterKernel instances avoids runtime SVD rank-1 checks.
conv_box = TrueConvolution('Kernel', k_box, 'Boundary', 'replicate');
conv_gauss = TrueConvolution('Kernel', k_gauss, 'Boundary', 'replicate');
conv_sobel = TrueConvolution('Kernel', k_sobel, 'Boundary', 'replicate');

% 5. Apply filters
I_blur = conv_box.apply(I);
I_smooth = conv_gauss.apply(I);
I_deriv = conv_sobel.apply(I);

% Scaling/offsetting derivative outputs for rendering (since values can be negative)
I_deriv_viz = min(max(2.0 * I_deriv + 0.5, 0.0), 1.0);

fprintf('Blur output size: %dx%d\n', size(I_blur, 1), size(I_blur, 2));
fprintf('Derivative min/max: %.4f / %.4f\n', min(I_deriv(:)), max(I_deriv(:)));
```

---

## 10. Zero-Stuffing and Synthesis Filtering
This recipe demonstrates how to upsample an image by injecting zeros (creating a checkerboard alias grid) and then passing it through a high-pass or low-pass synthesis filter to reconstruct subband details.

```matlab
% 1. Import strategies
import OceanMath.Image.Pixels.ZeroStuffing
import OceanMath.Image.Pixels.SynthesisFilter

% 2. Create sample input image
I = [ 10, 20; 30, 40 ];

% 3. Apply Zero-Stuffing by a factor of 2
zs = ZeroStuffing('Factor', 2);
I_stuffed = zs.apply(I);

% 4. Apply Synthesis Filters
% Low-pass synthesis (approximation component)
sf_lp = SynthesisFilter('Preset', 'Haar', 'Mode', 'Low-Pass (LL)');
I_approx = sf_lp.apply(double(I_stuffed));

% High-pass synthesis (diagonal detail component)
sf_hp = SynthesisFilter('Preset', 'Haar', 'Mode', 'Diagonal High-Pass (HH)');
I_detail = sf_hp.apply(double(I_stuffed));

disp('Original image:');
disp(I);
disp('Zero-stuffed image:');
disp(I_stuffed);
disp('Approximation (LL):');
disp(I_approx);
disp('Diagonal Details (HH):');
disp(I_detail);
```

---

## 11. Decimation (Downsampling): Blind vs. Anti-Aliased
This recipe demonstrates downsampling by an integer factor of 2, comparing blind decimation (which causes aliasing/grid deletion) with anti-aliasing low-pass filtered decimation.

```matlab
% 1. Import decimation strategy
import OceanMath.Image.Pixels.Decimation

% 2. Create sample input image (4x4 matrix representation)
I = [ 10,  20,  30,  40;
      50,  60,  70,  80;
      90, 100, 110, 120;
     130, 140, 150, 160 ];

% 3. Apply Blind Decimation (Factor = 2, simply delete pixels)
decim_blind = Decimation('Factor', 2, 'AntiAlias', false);
I_blind = decim_blind.apply(I);

% 4. Apply Anti-Aliased Decimation (Factor = 2, average filter before decimation)
decim_aa = Decimation('Factor', 2, 'AntiAlias', true, 'FilterType', 'Average');
I_aa = decim_aa.apply(I);

disp('Original Image (4x4):');
disp(I);
disp('Blindly Decimated Image (2x2):');
disp(I_blind);
disp('Anti-Aliased Decimated Image (2x2, Average filter):');
disp(I_aa);
```

---

## 12. Non-linear Filtering: Median and Bilateral Denoising (Szeliski Ex 3.12)
This recipe demonstrates non-linear rank filtering (Median) for salt-and-pepper impulse noise removal and edge-preserving bilateral filtering for Gaussian noise removal.

```matlab
% 1. Import non-linear filtering strategies
import OceanMath.Image.Pixels.MedianFilter
import OceanMath.Image.Pixels.BilateralFilter

% 2. Create a test image [64 x 64 x 3] with an edge
[X, Y] = meshgrid(1:64, 1:64);
edge_img = repmat(double(X > 32), [1, 1, 3]);

% 3. Synthesize Salt-and-Pepper noise and apply Median Filter
sp_noisy = edge_img;
rng(1);
sp_mask = rand(64, 64);
sp_noisy(repmat(sp_mask < 0.05, [1, 1, 3])) = 0; % Add black noise pixels
med = MedianFilter('WindowSize', [3, 3], 'BoundaryMode', 'clamp');
I_median = med.apply(uint8(sp_noisy * 255));

% 4. Synthesize Gaussian noise and apply Bilateral Filter
gauss_noisy = edge_img + 0.1 * randn(64, 64, 3);
bilat = BilateralFilter('WindowSize', [5, 5], 'SigmaSpatial', 2.0, 'SigmaRange', 0.15);
I_bilat = bilat.apply(gauss_noisy);

disp('Median filter denoised edge (Red channel middle row slice):');
disp(double(I_median(32, 28:36, 1)) / 255.0);
disp('Bilateral filter edge-preserved output (Red channel middle row slice):');
disp(I_bilat(32, 28:36, 1));
```

---

## 13. Directional Filtering: Freeman and Adelson's Steerable Filters (Szeliski Ex 3.13)
This recipe demonstrates first-order steerable filters. It convolved the image with basis derivatives once and then steers responses to arbitrary angles in O(1) time, extracting local dominant orientations.

```matlab
% 1. Import steerable filtering strategy
import OceanMath.Image.Pixels.SteerableFilter

% 2. Create a test image [64 x 64] with a diagonal line
[X, Y] = meshgrid(1:64, 1:64);
diag_line = double(abs(X - Y) <= 1);

% 3. Instantiate and apply first-order SteerableFilter
sf = SteerableFilter('Sigma', 1.5, 'Order', 1, 'Size', [11, 11]);
sf.apply(diag_line);

% 4. Steer responses to arbitrary angles in O(1) time
R_45  = sf.steer(45);  % Maximum response along the diagonal (45 deg)
R_135 = sf.steer(135); % Minimum response (orthogonal to the diagonal)

% 5. Compute dominant local orientation (theta_max) and response magnitude
[theta_max, mag] = sf.computeDominantOrientation();

disp('Steered response at 45 degrees (middle pixel):');
disp(R_45(32, 32));
disp('Steered response at 135 degrees (middle pixel):');
disp(R_135(32, 32));
disp('Dominant orientation angle in degrees (middle pixel):');
disp(theta_max(32, 32));
```

---

## 14. High-Quality Image Resampling: Separable Resizing (Szeliski Ex 3.16)
This recipe demonstrates high-quality image resizing using separable Nearest Neighbor, Bilinear, Bicubic, and Lanczos interpolation filters.

```matlab
% 1. Import image resampling strategy
import OceanMath.Image.Resampling.ImageResampler

% 2. Create a 2D sample signal [64 x 64]
[X, Y] = meshgrid(1:64, 1:64);
I_orig = double((X - 32).^2 + (Y - 32).^2 <= 15^2);

% 3. Magnify image (2.0x zoom) using Lanczos-3 filter
resampler_zoom = ImageResampler('TargetSize', [128, 128], 'FilterType', 'Lanczos');
I_zoom = resampler_zoom.apply(I_orig);

% 4. Minify image (0.5x scale) using anti-aliased Bicubic filter
resampler_shrink = ImageResampler('TargetSize', [32, 32], 'FilterType', 'Bicubic');
I_shrink = resampler_shrink.apply(I_orig);

disp('Zoomed size:');
disp(size(I_zoom));
disp('Shrunk size:');
disp(size(I_shrink));
```



