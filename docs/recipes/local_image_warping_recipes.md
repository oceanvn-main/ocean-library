# Recipe: Local Image Warping (Ex 3.23)

## Piecewise Affine Warp
```matlab
% Setup
res = 4;
[X, Y] = meshgrid(linspace(1,W,res), linspace(1,H,res));
pts_src = [X(:), Y(:)];
pts_tgt = pts_src;        % then modify pts_tgt to introduce deformation
pts_tgt(8,:) = [100, 80]; % e.g. move point 8

tri = delaunay(pts_src(:,1), pts_src(:,2));
I_warp = OceanMath.Image.Resampling.PiecewiseAffineWarp.warp(I_src, pts_src, pts_tgt, tri);
```

## Thin Plate Spline Warp
```matlab
tps = OceanMath.Image.Resampling.ThinPlateSpline(pts_tgt, pts_src, 'thinplate');
I_warp = tps.warpImage(I_src);
% Other kernels: 'gaussian', 'multiquadric', 'inversedistance'
```

## B-Spline FFD (Part 3)
```matlab
I_warp = OceanMath.Image.Resampling.BSplineWarp.warp(I_src, pts_src, pts_tgt, [res, res]);
```

## Beier-Neely Line Morphing (Part 2)
```matlab
% Define corresponding line pairs
P_src = [0.25*W, 0.45*H; 0.55*W, 0.25*H];  % start points
Q_src = [0.75*W, 0.45*H; 0.55*W, 0.75*H];  % end points
P_tgt = P_src + [10, 8; -12, 6];            % shift target endpoints
Q_tgt = Q_src + [5, -8; 10, -6];

I_warp = OceanMath.Image.Resampling.BeierNeelyWarp.warp(I_src, P_tgt, Q_tgt, P_src, Q_src);
```

## Paintbrush Dense Flow Warp (Part 4)
```matlab
[H, W, ~] = size(I_src);
[X, Y] = meshgrid(1:W, 1:H);

% Build a simple vortex displacement field
dx = X - W/2; dy = Y - H/2;
r  = sqrt(dx.^2 + dy.^2);
theta = atan2(dy, dx);
angle_shift = 0.5 * exp(-r.^2 / (2 * (W/6)^2));
Dx = (W/2 + r .* cos(theta + angle_shift)) - X;
Dy = (H/2 + r .* sin(theta + angle_shift)) - Y;

% Backward warp
u_src = X - Dx;  v_src = Y - Dy;
u0 = floor(u_src); v0 = floor(v_src);
% ... bilinear lookup (or use DisplacementFieldWarp)
I_warp = OceanMath.Image.Resampling.DisplacementFieldWarp.warp(I_src, Dx, Dy);
```

## Interactive GUI
```matlab
app = ComputerVisionSandbox();
app.show();
% Navigate to Tab 8: Local Warping
% Select mode from the "Warp Mode" dropdown
% Drag handles (grid/line endpoints) or paint directly on canvas
```
