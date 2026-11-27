# Directional Domain Scaling Law Recipes

Recipes for setting up non-uniform scaling transforms, applying inverse coordinate mappings, and performing directional spectral analyses on 2D separable signals.

---

## 1. Setup Signal and Non-Uniform Scale Transform

Generate a 2D separable sinusoidal signal and warp it using a non-uniform scaling matrix `A = diag(sx, sy)` with magnification along X and minification along Y.

```matlab
% Parameters
N_src = 512;
N_tgt = 512;
fx = 0.15; % X frequency
fy = 0.25; % Y frequency
sx = 2.0;  % Magnify X
sy = 0.4;  % Minify Y

% Generate source signal
[X_src, Y_src] = meshgrid(1:N_src, 1:N_src);
I_src = cos(2 * pi * fx * X_src) + cos(2 * pi * fy * Y_src);

% Target coordinate mapping (Center-aligned Gather)
xc_src = (N_src + 1) / 2;
yc_src = (N_src + 1) / 2;
xc_tgt = (N_tgt + 1) / 2;
yc_tgt = (N_tgt + 1) / 2;

I_dst = zeros(N_tgt, N_tgt);

for yd = 1:N_tgt
    for xd = 1:N_tgt
        % Inverse coordinate transformation
        xs = (xd - xc_tgt) / sx + xc_src;
        ys = (yd - yc_tgt) / sy + yc_src;
        
        % Bilinear Interpolation
        if xs >= 1 && xs <= N_src && ys >= 1 && ys <= N_src
            x0 = floor(xs); x1 = min(N_src, x0 + 1); dx = xs - x0;
            y0 = floor(ys); y1 = min(N_src, y0 + 1); dy = ys - y0;
            
            w00 = (1-dx)*(1-dy); w01 = dx*(1-dy);
            w10 = (1-dx)*dy;     w11 = dx*dy;
            
            I_dst(yd, xd) = w00 * I_src(y0, x0) + w01 * I_src(y0, x1) + ...
                            w10 * I_src(y1, x0) + w11 * I_src(y1, x1);
        end
    end
end
```

---

## 2. Extract Slices and Compute 1D FFTs

Extract row and column slices from the active region, apply Hann windowing to suppress boundary artifacts, and compute a high-resolution zero-padded FFT.

```matlab
% 1. Extract X signal (Full row is active since sx > 1)
x_sig = I_dst(round(N_tgt / 2), :);

% 2. Extract Y signal (Find active range dynamically since sy < 1)
active_y = find(sum(abs(I_dst), 2) > 1e-4);
y_start = min(active_y) + 5;
y_end = max(active_y) - 5;
y_sig = I_dst(y_start:y_end, round(N_tgt / 2))';

% 3. FFT configuration
N_fft = 8192;
freq_axis = (-N_fft/2 : N_fft/2 - 1) / N_fft;

% 4. FFT X-axis
x_sig_detrend = x_sig - mean(x_sig);
win_x = hann(length(x_sig_detrend))';
X_fft = fftshift(fft(x_sig_detrend .* win_x, N_fft));
X_mag = abs(X_fft) / sum(win_x);

% 5. FFT Y-axis
y_sig_detrend = y_sig - mean(y_sig);
win_y = hann(length(y_sig_detrend))';
Y_fft = fftshift(fft(y_sig_detrend .* win_y, N_fft));
Y_mag = abs(Y_fft) / sum(win_y);
```

---

## 3. Find Spectral Peak Frequencies

Identify peak locations in the positive frequency domain `[0, 0.5]` and compare them to the theoretical values.

```matlab
% Find positive half indices
pos_indices = find(freq_axis >= 0);
freq_pos = freq_axis(pos_indices);

% X peak
X_mag_pos = X_mag(pos_indices);
[~, max_idx_x] = max(X_mag_pos);
fx_empirical = freq_pos(max_idx_x);
fx_theory = fx / sx;

% Y peak
Y_mag_pos = Y_mag(pos_indices);
[~, max_idx_y] = max(Y_mag_pos);
fy_empirical = freq_pos(max_idx_y);
fy_theory_alias = abs(fy / sy - round(fy / sy));

% Display results
fprintf('X-Axis Freq - Theory: %.3f, Empirical: %.3f\n', fx_theory, fx_empirical);
fprintf('Y-Axis Freq - Theory: %.3f, Empirical: %.3f\n', fy_theory_alias, fy_empirical);
```
