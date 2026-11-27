# Non-Separable Prefilter Recipes

Recipes for constructing rotated elliptical Gaussian filters, decomposing them via Singular Value Decomposition (SVD) for rank-1 approximation, and benchmarking 2D vs 1D separable convolutions.

---

## 1. Construct Rotated Anisotropic Gaussian Kernel

Create a 2D Gaussian kernel rotated by `theta` degrees with different standard deviations `sigma_1` and `sigma_2` along its primary axes.

```matlab
% Kernel parameters
K_size = 15;
w = (K_size - 1) / 2;
[X_k, Y_k] = meshgrid(-w:w, -w:w);
coords = [X_k(:), Y_k(:)]';

% Rotated covariance matrix
theta = 45;
rad = deg2rad(theta);
R = [cos(rad), -sin(rad); sin(rad), cos(rad)];
sigma_1 = 0.4; % Magnified axis standard deviation
sigma_2 = 2.0; % Minified axis standard deviation
Sigma = R * [sigma_1^2, 0; 0, sigma_2^2] * R';

% Evaluate Gaussian density
Sigma_inv = inv(Sigma);
G_flat = exp(-0.5 * sum(coords .* (Sigma_inv * coords), 1));
G = reshape(G_flat, K_size, K_size);
G = G / sum(G(:)); % Normalize
```

---

## 2. Factorize via SVD (Separable Approximation)

Decompose the non-separable 2D kernel into its rank-1 separable components using SVD.

```matlab
% Singular Value Decomposition
[U, S, V] = svd(G);
s_vals = diag(S);

% Check separability ratio (S(2,2)/S(1,1))
separability_ratio = s_vals(2) / s_vals(1);
fprintf('Singular value ratio: %.4f%%\n', separability_ratio * 100);

% Extract horizontal and vertical 1D filters (Rank-1 Approximation)
g_v = U(:, 1) * sqrt(s_vals(1));
g_h = V(:, 1)' * sqrt(s_vals(1));
```

---

## 3. Benchmark Convolutions

Compare the execution times of full 2D convolution against the separable horizontal-then-vertical pass on an input image.

```matlab
I = double(imread('Data/images/ocean_img_gray.jpg')) / 255.0;
boundary = 'replicate';

% 1. Full 2D Convolution (Non-separable)
t_start = tic;
K_conv = rot90(G, 2);
[Kh, Kw] = size(K_conv);
pad_rows = (Kh - 1) / 2;
pad_cols = (Kw - 1) / 2;

I_pad = OceanMath.Image.Pixels.BoundaryPadder.pad(I, [pad_rows, pad_cols], boundary);
I_out_2d = zeros(size(I));
for r = 1:Kh
    for c = 1:Kw
        I_out_2d = I_out_2d + K_conv(r, c) * I_pad(r : r + size(I,1) - 1, c : c + size(I,2) - 1, :);
    end
end
time_2d = toc(t_start);

% 2. Separable Rank-1 Convolution
t_start = tic;
% Horizontal pass (g_h is 1 x Kh)
Kh = length(g_h);
pad_cols = (Kh - 1) / 2;
g_h_conv = fliplr(g_h);
I_pad_h = OceanMath.Image.Pixels.BoundaryPadder.pad(I, [0, pad_cols], boundary);
I_h = zeros(size(I));
for m = 1:Kh
    I_h = I_h + g_h_conv(m) * I_pad_h(:, m : m + size(I,2) - 1, :);
end

% Vertical pass (g_v is Kv x 1)
Kv = length(g_v);
pad_rows = (Kv - 1) / 2;
g_v_conv = flipud(g_v(:));
I_pad_v = OceanMath.Image.Pixels.BoundaryPadder.pad(I_h, [pad_rows, 0], boundary);
I_out_1d = zeros(size(I));
for m = 1:Kv
    I_out_1d = I_out_1d + g_v_conv(m) * I_pad_v(m : m + size(I,1) - 1, :, :);
end
time_1d = toc(t_start);

fprintf('2D Time: %.4f s, 1D Separable Time: %.4f s (Speedup: %.1fx)\n', ...
        time_2d, time_1d, time_2d/time_1d);
```
