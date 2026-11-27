# Parametric Warping & Mipmapping Recipes

## Recipe 1: Affine Warp with Inverse Mapping and Bilinear Interpolation

This snippet performs a rotation and scale around the center of the image using inverse mapping.

```matlab
% Load image and convert to double
I = im2double(rgb2gray(imread('Data/images/elon_musk.jpg')));
[H, W] = size(I);

% Rotation and Scale transformation matrices
theta = 25; s = 0.85;
cx = W/2 + 0.5; cy = H/2 + 0.5;

T1 = [1, 0, -cx; 0, 1, -cy; 0, 0, 1];
R  = [cosd(theta), -sind(theta), 0; sind(theta), cosd(theta), 0; 0, 0, 1];
S  = [s, 0, 0; 0, s, 0; 0, 0, 1];
T2 = [1, 0, cx; 0, 1, cy; 0, 0, 1];

H_inv = inv(T2 * R * S * T1);

% Generate coordinates grid
[X_t, Y_t] = meshgrid(1:W, 1:H);
pts_t = [X_t(:)'; Y_t(:)'; ones(1, H*W)];

% Map target pixels back to source coordinate space
pts_s = H_inv * pts_t;
X_s = reshape(pts_s(1, :) ./ pts_s(3, :), H, W);
Y_s = reshape(pts_s(2, :) ./ pts_s(3, :), H, W);

% Interpolate using MATLAB's interp2
I_warp = interp2(I, X_s, Y_s, 'linear', 0); % 0 is background fill
```

---

## Recipe 2: MIP-Mapping with Trilinear Blending

Query and sample a fractional level from a pre-built MIP-map using `MipMap`.

```matlab
% Instantiate and build MipMap
mipmap = OceanMath.Image.MultiResolution.MipMap();
mipmap.build(I, 4);

% Suppose we are downsampling by 0.35x (ideal level is d = log2(1/0.35) = 1.515)
scale = 0.35;
J = [1/scale, 0; 0, 1/scale]; % Jacobian of uniform scale

W_out = 128; H_out = 128;
I_trilinear = zeros(H_out, W_out);

for r = 1:H_out
    for c = 1:W_out
        % Coordinates mapping screen coordinates to texture space
        u = c / scale;
        v = r / scale;
        
        % Sample using trilinear blending
        I_trilinear(r, c) = mipmap.sample(u, v, J, 'Trilinear');
    end
end
```

---

## Recipe 3: Perspective Ground Plane Mapping & EWA Anisotropic Filtering

Simulate a perspective floor rendering using analytical Jacobians and the `EWASampler`.

```matlab
% Set up a high-frequency grid texture
[U, V] = meshgrid(1:256, 1:256);
I_checker = double(mod(floor(U/16) + floor(V/16), 2) == 0);

% Perspective parameters
W_scr = 400; H_scr = 300;
xc = W_scr / 2; yh = 80; % Horizon line
su = 130; sv = 1100;
uc = 128; vc = 5;

% Output image
I_out = zeros(H_scr, W_scr);

% Instantiate EWA sampler
ewa = OceanMath.Image.Resampling.EWASampler('RadiusSquared', 3.0);

for y = (yh + 1):H_scr
    dy = y - yh;
    for x = 1:W_scr
        % Map coordinate (x,y) on screen to ground coordinate (u,v)
        u = uc + su * (x - xc) / dy;
        v = vc + sv / dy;
        
        % Compute local analytical Jacobian: d(u,v)/d(x,y)
        dudx = su / dy;
        dudy = -su * (x - xc) / (dy^2);
        dvdx = 0;
        dvdy = -sv / (dy^2);
        J = [dudx, dudy; dvdx, dvdy];
        
        % Sample the texture using EWA
        I_out(y, x) = ewa.sample(I_checker, u, v, J);
    end
end
```
