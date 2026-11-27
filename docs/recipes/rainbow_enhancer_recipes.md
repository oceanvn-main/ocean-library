# Rainbow Separation and Enhancer Recipes

Copy-pasteable code snippets for image linearization, region filling, background subtraction, and sRGB re-encoding.

## 1. Linearization and Clamping (Phase 1)
```matlab
% Read input sRGB image
P_total = im2double(imread('C:\Ocean Library\Matlab code practice\Data\images\sea.jpg'));

% Linearize sRGB data using inverse gamma expansion (with max(0,...) clamping to prevent complex numbers)
L_total = max(0, P_total) .^ 2.2;
```

## 2. Background Synthesis via Laplacian Inpainting (Phase 3)
```matlab
% Create a logical mask representing the rainbow region (H x W logical matrix)
[H, W, ~] = size(L_total);
M = false(H, W);
% (Example: circular arc region)
[X, Y] = meshgrid(1:W, 1:H);
Dist = sqrt((X - 256).^2 + (Y - 512).^2);
M(Dist >= 240 & Dist <= 290) = true;

% Solve Laplace equation inside the mask channel-by-channel
L_bg = zeros(H, W, 3);
for c = 1:3
    L_bg(:,:,c) = regionfill(L_total(:,:,c), M);
end
```

## 3. Mathematical Separation & Re-encoding (Phases 4 & 5)
```matlab
% Subtract background and clamp negative values to zero
L_rainbow = max(L_total - L_bg, 0);

% Contrast boost and re-apply display gamma compression
k = 3.0; % Gain factor
P_isolated = max(0, k * L_rainbow) .^ (1 / 2.2);

% Display pure isolated spectrum
figure;
imshow(P_isolated);
title('Isolated Rainbow (sRGB)');
```

## See Also
- [Rainbow Separation and Enhancer Guide](../guides/rainbow_enhancer.md)
- [Bilateral Filter Recipe](../recipes/filtering_recipes.md)
