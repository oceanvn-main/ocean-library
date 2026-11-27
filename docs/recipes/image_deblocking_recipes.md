# Image Deblocking — Recipes

## Recipe 1: Run the Full Ex 4.3 Experiment (Synthetic Image)

```matlab
run('Experiments/Scripts/Computer Vision/exp_image_deblocking.m')
% IMAGE_SOURCE = 'synthetic' by default — no files needed
```

---

## Recipe 2: Run Ex 4.3 With Your Own Image

Edit the top of `exp_image_deblocking.m`:
```matlab
IMAGE_SOURCE = fullfile('c:', 'Ocean Library', 'Matlab code practice', ...
                        'Data', 'images', 'your_image.jpg');
QUALITY_FACTOR = 8;   % Lower = more blocking
```

---

## Recipe 3: Software JPEG Compress + Decompress

```matlab
comp = OceanMath.Image.Compression.JpegSyntheticCompressor('QualityFactor', 10);
[img_compressed, q_all, Q_table] = comp.compress(img_clean);
img_reconstructed = comp.decompress(q_all, Q_table);

fprintf('PSNR = %.2f dB\n', OceanStats.Image.ImageQualityMetrics.psnr(img_compressed, img_clean));
```

---

## Recipe 4: Standalone POCS Deblocking

```matlab
% Assumes img_compressed, q_all, Q_table are already computed
engine = OceanMath.Image.Compression.PocsDeblockingEngine( ...
    'MaxIter', 50, 'Tolerance', 1e-5, 'SmoothEvery', 2);
[img_out, history] = engine.deblock(img_compressed, q_all, Q_table);

% Visualise convergence
figure; semilogy(history.RMSE, '-o');
xlabel('Iteration'); ylabel('Relative Change'); title('POCS Convergence');
```

---

## Recipe 5: Swap the POCS Smoother

```matlab
% Default: BilateralFilter (edge-preserving, recommended)
engine = OceanMath.Image.Compression.PocsDeblockingEngine();

% Alternative: Gaussian blur (simpler but blurs edges)
engine.Smoother = OceanMath.Image.Filters.GaussianFilter('Sigma', 0.8);

% Alternative: Disable smoothing (pure DCT projection only)
engine.Smoother = [];
```

---

## Recipe 6: Guided Filter Upsampling (Standalone)

```matlab
gf = OceanMath.Image.Filters.GuidedFilter('Radius', 8, 'Epsilon', 0.01);

% Upsample sparse signal p using high-res guidance image I
p_upsampled = gf.applyGuided(p_lowres_nn, guidance_I_highres);

% Or self-guided edge-preserving smoothing
p_smooth = gf.apply(p_noisy);
```

---

## Recipe 7: Joint Bilateral Upsampling (Alternative to Guided)

```matlab
jbf = OceanMath.Image.Filters.JointBilateralFilter( ...
    'SpatialSigma', 5, 'RangeSigma', 0.1);
p_upsampled = jbf.applyGuided(p_lowres_nn, guidance_I_highres);
```

---

## Recipe 8: Chrominance Subsampling + Guided Restore

```matlab
YCbCr = rgb2ycbcr(img_rgb);

sub = OceanMath.Image.Compression.ChromaSubsampler( ...
    'Scheme', '4:2:0', 'UpsamplingMethod', 'guided');

[Y, Cb_sub, Cr_sub] = sub.subsample(YCbCr);
YCbCr_out = sub.upsample(Y, Cb_sub, Cr_sub);
img_out = ycbcr2rgb(YCbCr_out);
```

---

## Recipe 9: PSNR and SSIM Comparison Table

```matlab
m(1).Label = 'Compressed'; m(1).PSNR = psnr(img_comp, img_ref); m(1).SSIM = ssim(img_comp, img_ref);
m(2).Label = 'POCS Out';   m(2).PSNR = psnr(img_pocs, img_ref); m(2).SSIM = ssim(img_pocs, img_ref);

OceanStats.Image.ImageQualityMetrics.printTable(m);
% or single line:
OceanStats.Image.ImageQualityMetrics.printReport(img_pocs, img_ref, 'POCS Output');
```

---

## Recipe 10: Generate Synthetic Test Image

```matlab
% Gradient background + geometric shapes + sinusoidal texture + step edge
gen = OceanData.Generators.SyntheticImageGenerator( ...
    'Height', 512, 'Width', 512, 'Mode', 'mixed', 'Seed', 42);
img = gen.generate();   % (512 x 512 x 3) double [0,1]
imshow(img);

% Checkerboard (worst-case for JPEG blocking)
gen2 = OceanData.Generators.SyntheticImageGenerator('Mode', 'checkerboard');
img_cb = gen2.generate();
```

---

---

## Recipe 12: Dual Degradation Robustness Experiment (Noise + Severe JPEG Q=5)

```matlab
% Run the noisy POCS robustness benchmark using Sony Camera RAW data
run('Experiments/Scripts/Computer Vision/exp_pocs_robustness_noisy.m')
```

### Manual Setup for Dual Degradation Testing

```matlab
% 1. Load clean image (e.g. Sony RAW _DSC5165.ARW)
raw_cfa = rawread('Data/images/raw/_DSC5165.ARW');
img_clean = double(demosaic(raw_cfa, 'rggb')) / 65535.0;

% 2. Additive Gaussian Sensor Noise (sigma = 0.05)
YCbCr_clean = rgb2ycbcr(img_clean);
YCbCr_noisy = max(0.0, min(1.0, YCbCr_clean + 0.05 * randn(size(YCbCr_clean))));

% 3. Severe JPEG Quantization (Q = 5)
comp = OceanMath.Image.Compression.JpegSyntheticCompressor('QualityFactor', 5);
[YCbCr_degraded, q_all, Q_table] = comp.compress(YCbCr_noisy);

% 4. POCS Deblocking with Edge-Preserving Bilateral Filter
engine = OceanMath.Image.Compression.PocsDeblockingEngine( ...
    'MaxIter', 40, ...
    'Tolerance', 1e-5, ...
    'Smoother', OceanMath.Image.Filters.BilateralFilter('SpatialSigma', 1.5, 'RangeSigma', 0.08));

[YCbCr_restored, history] = engine.deblock(YCbCr_degraded, q_all, Q_table);
img_restored = ycbcr2rgb(YCbCr_restored);
```

