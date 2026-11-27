# Wavelet Compression Recipes

## Quick Start: Run the Full Benchmark

```matlab
run('Experiments/Scripts/Computer Vision/exp_wavelet_compression.m');
```

---

## Recipe 1: Rate-Distortion Sweep (Laplacian Pyramid)

```matlab
I_rgb  = im2double(imread('Data/images/elon_musk.jpg'));
I_lab  = rgb2lab(I_rgb);
I_luma = I_lab(:,:,1) / 100.0;    % L* normalized to [0,1]
N_orig = numel(I_luma);
num_levels = 4;

q_values = [0.25, 1, 4, 16, 64] / 255.0;
computePSNR = @(a, b) 10*log10(1.0 / mean((a(:)-b(:)).^2 + eps));

for qi = 1:numel(q_values)
    q = q_values(qi);
    pyr = OceanMath.Image.MultiResolution.LaplacianPyramid();
    pyr.build(I_luma, num_levels);

    total_bits = 0;
    for lvl = 1:num_levels
        band = pyr.Levels{lvl};
        idx  = round(band / q);
        pyr.Levels{lvl} = idx * q;   % dequantize in place
        H = OceanMath.Information.Entropy(idx(:), 'TypeAsString', 'Discrete', 'Base', 2);
        total_bits = total_bits + numel(band) * H;
    end
    bpp   = total_bits / N_orig;
    I_rec = min(max(pyr.reconstruct(), 0), 1);
    psnr  = computePSNR(I_luma, I_rec);
    fprintf('q=%.4f  bpp=%.3f  PSNR=%.1f dB\n', q, bpp, psnr);
end
```

---

## Recipe 2: Per-Band Entropy Audit (Haar Wavelet)

```matlab
haar = OceanMath.Signal.Lifting.LiftingScheme('WaveletType', 'Haar', 'Levels', 4);
coeffs = haar.forward(I_luma);
q_audit = 1 / (4 * 255);    % 1/4 gray level

% LL approximation subband
H_LL = OceanMath.Information.Entropy(round(coeffs.LL(:)/q_audit), 'TypeAsString','Discrete','Base',2);
fprintf('LL: %.4f bits, %d pixels\n', H_LL, numel(coeffs.LL));

% Detail subbands
for lvl = 1:4
    for fn = {'LH','HL','HH'}
        band = coeffs.Levels{lvl}.(fn{1});
        H = OceanMath.Information.Entropy(round(band(:)/q_audit), 'TypeAsString','Discrete','Base',2);
        fprintf('L%d %s: %.4f bits, %d pixels\n', lvl, fn{1}, H, numel(band));
    end
end
```

---

## Recipe 3: CDF 5/3 Quantize and Reconstruct (Integer Space)

> **Always** apply CDF 5/3 lifting to an integer-scaled image.
> Using `[0,1]` float violates the `floor()` lifting contract → maximum-entropy garbage.

```matlab
I_luma_int = round(I_luma * 255);     % scale to [0, 255] integer space
q_int = 8;                            % 8 gray levels = moderate compression

cdf = OceanMath.Signal.Lifting.LiftingScheme('WaveletType', 'CDF53', 'Levels', 4);
c   = cdf.forward(I_luma_int);

% Quantize LL
c.LL = round(c.LL / q_int) * q_int;
% Quantize detail subbands
for lvl = 1:4
    for fn = {'LH','HL','HH'}
        c.Levels{lvl}.(fn{1}) = round(c.Levels{lvl}.(fn{1}) / q_int) * q_int;
    end
end

% Reconstruct and scale back
I_recon_int = cdf.inverse(c);
I_recon = min(max(I_recon_int / 255.0, 0), 1);

psnr = 10 * log10(1.0 / mean((I_luma(:) - I_recon(:)).^2 + eps));
fprintf('CDF 5/3  q=%d  PSNR=%.1f dB\n', q_int, psnr);
```

---

## Recipe 4: Side-by-Side Comparison Plot

```matlab
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Color', viz.Colors.Background, 'Position', [100, 100, 1200, 400]);

titles = {'Original', 'Pyramid (Lossy)', 'Haar (Lossy)', 'CDF 5/3 (Lossy)'};
images = {I_luma, I_pyr_recon, I_haar_recon, I_cdf_recon};

for k = 1:4
    ax = subplot(1, 4, k);
    imshow(images{k}, 'Parent', ax);
    title(ax, titles{k}, 'Color', viz.Colors.Text, 'FontSize', 10);
end
```
