# Computer Vision Recipes

Copy-pasteable code blocks for implementing standard computer vision math operations using `OceanMath.Vision`.

---

## 1. 2D Homogeneous Matrix Composition & Point Warping
This recipe builds a composite affine transformation matrix and applies it to homogeneous coordinates.

```matlab
% 1. Import mathematical package
import OceanMath.Geometry.Transform2D

% 2. Define transform parameters
tx = 1.5; ty = -2.0;       % Translation
theta_deg = 30;            % Rotation in degrees
sx = 1.2; sy = 0.8;        % Scaling
shx = 0.1; shy = -0.2;     % Shearing

% 3a. Build 2D rigid/Euclidean matrix (T * R) - 3 DOF
SE = Transform2D.rigid(tx, ty, theta_deg);

% 3b. Build 2D similarity transform matrix (T * R * S) - 4 DOF
Sim = Transform2D.similarity(tx, ty, theta_deg, sx); % uses uniform scaling sx

% 3c. Build 2D stretch/squash matrix (T * S) - 4 DOF (changes aspect ratio)
SS = Transform2D.stretchSquash(tx, ty, sx, sy);

% 3d. Build composite 3x3 affine matrix H = T * R * Sh * S - 6 DOF
H = Transform2D.affine(tx, ty, theta_deg, sx, sy, shx, shy);

% 4. Create coordinate grid of 2D points [2 x N]
pts = [0, 1, 1, 0; ...
       0, 0, 1, 1];

% 5. Warp coordinates using homogeneous normalization
ptsWarped = Transform2D.transformPoints(H, pts);

disp('Composite Matrix H:');
disp(H);
disp('Warped coordinates:');
disp(ptsWarped);
```

---

## 2. Spherical Linear Interpolation (SLERP) Between Rotations
This recipe performs geodesic rotation interpolation between two unit quaternions representing orientation states.

```matlab
import OceanMath.Geometry.Rotation3D

% 1. Define start rotation: 0 degrees orientation
q_start = [1, 0, 0, 0];

% 2. Define end rotation: 90 degrees around Z axis
R_end = Rotation3D.eulerToMatrix(0, 0, 90);
q_end = Rotation3D.matrixToQuat(R_end);

% 3. Interpolate orientation at 10 incremental steps along the geodesic path
t_vals = 0:0.1:1.0;
for t = t_vals
    q_t = Rotation3D.slerp(q_start, q_end, t);
    R_t = Rotation3D.quatToMatrix(q_t);
    
    % Display current rotation matrix
    fprintf('t = %.1f, R trace: %.3f\n', t, trace(R_t));
end
```

---

## 3. 3D Projective Camera Pipeline
This recipe models intrinsics and extrinsics to project 3D points onto a 2D camera viewport sensor.

```matlab
import OceanMath.Geometry.CameraProjection
import OceanMath.Geometry.Rotation3D

% 1. Create 3D world coordinate points
ptsWorld = [ -0.5, 0.5, 0.0; ...
              1.0, 2.0, 1.5; ...
              5.0, 6.0, 7.0 ]';

% 2. Setup Camera Extrinsics: Position in world at (0, 1, -10) tilted slightly down
camPos = [0; 1; -10];
R_c2w = Rotation3D.eulerToMatrix(0, 15, 0); % 15 degrees pitch tilt
T_w2c = [R_c2w', -R_c2w'*camPos; 0, 0, 0, 1];

% 3. Setup Camera Intrinsics (640x480 resolution calibration)
K = [600, 0, 320; ...
     0, 600, 240; ...
     0, 0, 1];

% 4. Instantiate Perspective Camera Model
camera = CameraProjection('Intrinsics', K, 'Extrinsics', T_w2c, 'ProjectionType', 'Perspective');

% 5. Project to 2D
[pts2D, inFront] = camera.project(ptsWorld);

disp('Projected image coordinates (pixels):');
disp(pts2D(:, inFront));
```

---

## 4. Brown-Conrady Radial and Tangential Lens Distortion
This recipe applies radial (barrel/pincushion) and decentering tangential lens distortion models to warp image-space pixels.

```matlab
import OceanMath.Geometry.LensDistortion

% 1. Define calibration intrinsics
K = [500, 0, 320; 0, 500, 240; 0, 0, 1];

% 2. Setup lens distortion coefficients (moderate barrel + tangential shear)
k1 = -0.15; k2 = 0.05; k3 = 0.0;  % Radial
p1 = 0.02;  p2 = -0.01;           % Tangential
distortion = LensDistortion('Radial', [k1, k2, k3], 'Tangential', [p1, p2]);

% 3. Define original ideal pixel coordinates
pixels = [ 200, 320, 480; ...
           150, 240, 330 ];

% 4. Apply aberration distortion to image-space pixels
pixelsWarped = distortion.distortImageCoordinates(pixels, K);

disp('Warped image pixels:');
disp(pixelsWarped);
```

---

## 5. Simard Tangent Distance and Subspace Projection
This recipe loads two handwritten digit samples from the package generator, computes their spatial gradients and rotational tangents (Lie Algebra generators), sweeps a rotation angle to construct their true transformation manifolds (Lie Group action), projects the manifolds into a 3D subspace using SVD, and solves for the invariant Tangent Distance using linear least-squares.

```matlab
% 1. Load data
gen = OceanData.Generators.ZipGenerator('Type', 'train');
[X_all, Y_all] = gen.generate();
idx3 = find(Y_all == 3);

% Flattened 16x16 images as 256-dimensional vectors
x1 = X_all(idx3(5), :)';
x2 = X_all(idx3(12), :)';

% 2. Smooth and calculate spatial gradients
smoothSigma = 0.6;
I1 = reshape(x1, [16, 16])';
I2 = reshape(x2, [16, 16])';
I1_smooth = imgaussfilt(I1, smoothSigma);
I2_smooth = imgaussfilt(I2, smoothSigma);
[dI1_dx, dI1_dy] = gradient(I1_smooth);
[dI2_dx, dI2_dy] = gradient(I2_smooth);

% 3. Define coordinate mesh grid centered at image center
cx = 8.5; cy = 8.5;
[Xc, Yc] = meshgrid((1:16) - cx, (1:16) - cy);

% 4. Compute horizontal, vertical, and rotational tangents (Lie algebra vectors)
T1_tx = -dI1_dx(:); T1_ty = -dI1_dy(:); T1_rot = -Yc(:).*dI1_dx(:) + Xc(:).*dI1_dy(:);
T2_tx = -dI2_dx(:); T2_ty = -dI2_dy(:); T2_rot = -Yc(:).*dI2_dx(:) + Xc(:).*dI2_dy(:);

% 5. Solve for optimal coefficients that minimize the distance between tangent hyperplanes
% Tangent spaces: S1(c1) = x1 + V1*c1, S2(c2) = x2 + V2*c2
V1 = [T1_tx, T1_ty, T1_rot]; % Size: 256 x 3
V2 = [T2_tx, T2_ty, T2_rot]; % Size: 256 x 3
M = [V1, -V2];
e = x1 - x2;
c_opt = - (M' * M) \ (M' * e);

% 6. Compute closest points and invariant Tangent Distance
p1 = x1 + V1 * c_opt(1:3);
p2 = x2 + V2 * c_opt(4:6);
d_tangent = norm(p1 - p2);
d_euclidean = norm(x1 - x2);

fprintf('Euclidean Distance: %.4f\n', d_euclidean);
fprintf('Tangent Distance  : %.4f\n', d_tangent);
```

---

## 6. Co-vector and Surface Normal Transformations (H^-T)
This recipe transforms 2D homogeneous lines and 3D surface mesh normals, demonstrating how the naïve standard point matrix multiplication breaks geometric alignment, and how the transposed inverse (`H^-T`) preserves intersection and perpendicularity.

```matlab
% === Part A: 2D Homogeneous Line Duality ===
% 1. Define point on line x - y = 0
x = [2; 2; 1];
l = [1; -1; 0]; % x - y = 0 line co-vector

% 2. Define a non-orthogonal shear matrix
H = [1.5, 1.0, 0.0; ...
     0.0, 0.8, 0.0; ...
     0.5, 0.0, 1.2];

% 3. Transform the point
x_warped = H * x;

% 4. Transform the line using Naive (H * l) vs Correct (H^-T * l)
l_naive = H * l;
l_correct = inv(H)' * l;

% 5. Verify intersection (must equal zero)
err_naive = l_naive' * x_warped;     % Generally non-zero!
err_correct = l_correct' * x_warped; % Strictly zero! (within eps)

fprintf('2D Naive Line Intersect Error  : %.4f\n', err_naive);
fprintf('2D Correct Line Intersect Error: %.4e\n', err_correct);


% === Part B: 3D Surface Normal Perpendicularity ===
% 1. Define a 3D point and analytic tangent vectors on paraboloid z = -(x^2+y^2) at (1,1)
pt = [1; 1; -2];
T_x = [1; 0; -2];  % dx tangent
T_y = [0; 1; -2];  % dy tangent
N = [2; 2; 1];     % normal = Tx x Ty
normals_unit = N / norm(N);

% 2. Transform geometry via 3D scale and shear H
H_3d = [1.8, 0.5, 0.0; ...
        0.0, 0.9, 0.0; ...
        0.2, 0.0, 1.5];

% 3. Transformed surface tangents warp directly with H (Chain Rule)
T_x_warped = H_3d * T_x;
T_y_warped = H_3d * T_y;

% 4. Transform surface normal using Naive (H * N) vs Correct (H^-T * N)
N_naive = H_3d * normals_unit;
N_correct = inv(H_3d)' * normals_unit;

% Normalize to unit vectors
n_naive = N_naive / norm(N_naive);
n_correct = N_correct / norm(N_correct);

% 5. Verify perpendicularity (dot products with tangents must equal zero)
err_naive_perp = abs(n_naive' * T_x_warped) + abs(n_naive' * T_y_warped);     % Non-zero!
err_correct_perp = abs(n_correct' * T_x_warped) + abs(n_correct' * T_y_warped); % Zero!

fprintf('3D Naive Normal Ortho Error  : %.4f\n', err_naive_perp);
fprintf('3D Correct Normal Ortho Error: %.4e\n', err_correct_perp);
```

---

## 7. 3D Homogeneous Matrix Composition & Coordinate Warping
This recipe builds rigid, similarity, affine, and projective 3D homogeneous transformation matrices and warps a grid of 3D points.

```matlab
% 1. Import mathematical package
import OceanMath.Geometry.Transform3D

% 2. Define coordinate points of a 3D wireframe cube
[X, Y, Z] = meshgrid([-1, 1], [-1, 1], [-1, 1]);
pts3D = [X(:)'; Y(:)'; Z(:)']; % Size [3 x 8]

% 3. Define transformation parameters
tx = 1.0; ty = -2.0; tz = 3.5;    % Translation vector
s = 1.5;                         % Uniform scaling factor
R_mat = [0, -1, 0; 1, 0, 0; 0, 0, 1]; % 90 deg rotation around Z-axis

% 4. Build homogeneous transformation matrices
SE = Transform3D.rigid(tx, ty, tz, R_mat);            % Rigid (6 DOF)
Sim = Transform3D.similarity(tx, ty, tz, R_mat, s);   % Similarity (7 DOF)

% Build 3D affine matrix by combining translation and linear transformations
A_mat = [1.2, 0.1, 0.0; -0.1, 0.9, 0.2; 0.0, 0.0, 1.0];
A = Transform3D.affine(A_mat, tx, ty, tz);            % Affine (12 DOF)

% 5. Warp coordinates using homogeneous normalization
warpedSE = Transform3D.transformPoints(SE, pts3D);
warpedSim = Transform3D.transformPoints(Sim, pts3D);
warpedA = Transform3D.transformPoints(A, pts3D);

disp('Warped with Rigid (SE3):');
disp(warpedSE(:, 1:4));
disp('Warped with Similarity:');
disp(warpedSim(:, 1:4));
disp('Warped with Affine:');
disp(warpedA(:, 1:4));
```

---

## 8. 3D Euler Rotations, Gimbal Lock, and Quaternion SLERP Interpolation
This recipe demonstrates how successive 3D Euler rotations lead to Gimbal Lock at Pitch = 90 degrees, proving algebraically that two distinct angular states produce the exact same physical rotation matrix, and demonstrates how to resolve trajectory singularities using unit quaternion Spherical Linear Interpolation (SLERP).

```matlab
% 1. Import mathematical package
import OceanMath.Geometry.Rotation3D

% --- Part A: Algebraic Gimbal Lock Proof ---
pitch_lock = 90;

% State A: Yaw = 45, Roll = 15 (Difference = 30)
R_A = Rotation3D.eulerToMatrix(15, pitch_lock, 45);

% State B: Yaw = 30, Roll = 0 (Difference = 30)
R_B = Rotation3D.eulerToMatrix(0, pitch_lock, 30);

% Compute Frobenius norm of difference
frob_diff = norm(R_A - R_B, 'fro');

fprintf('Gimbal Lock Proof:\n');
fprintf('  R_A (Yaw=45, Pitch=90, Roll=15) Frobenius diff from R_B (Yaw=30, Pitch=90, Roll=0): %.4e\n', frob_diff);


% --- Part B: Quaternion SLERP Geodesic Interpolation ---
% Define start/end orientations passing through Gimbal Lock boundary (Pitch = 90)
roll1 = -45; pitch1 = 45;  yaw1 = -60;
roll2 = 45;  pitch2 = 135; yaw2 = 60;

R_start = Rotation3D.eulerToMatrix(roll1, pitch1, yaw1);
R_end = Rotation3D.eulerToMatrix(roll2, pitch2, yaw2);

q_start = Rotation3D.matrixToQuat(R_start);
q_end = Rotation3D.matrixToQuat(R_end);

% Trace a point coordinate along the interpolated paths (t = 0.5)
q_mid = Rotation3D.slerp(q_start, q_end, 0.5);
R_mid_slerp = Rotation3D.quatToMatrix(q_mid);

% Linear interpolation of Euler angles
roll_mid = roll1 + 0.5 * (roll2 - roll1);
pitch_mid = pitch1 + 0.5 * (pitch2 - pitch1);
yaw_mid = yaw1 + 0.5 * (yaw2 - yaw1);
R_mid_euler = Rotation3D.eulerToMatrix(roll_mid, pitch_mid, yaw_mid);

% Rotate a reference point [0; 0; 1]
pt_ref = [0; 0; 1];
p_slerp = R_mid_slerp * pt_ref;
p_euler = R_mid_euler * pt_ref;

fprintf('\nInterpolated Midpoint Coordinates (t = 0.5):\n');
fprintf('  Euler linear interpolation: [%.4f; %.4f; %.4f]\n', p_euler(1), p_euler(2), p_euler(3));
fprintf('  Quaternion SLERP path      : [%.4f; %.4f; %.4f]\n', p_slerp(1), p_slerp(2), p_slerp(3));
```

---

## 9. 3D Room Light Simulation & Shading comparison
This recipe constructs a 3D simulated room containing a red sphere, configures a camera viewport, inserts an ambient light and point light source, and renders the scene under three different modalities (Vector Flat, Vector Gouraud, and Ray-Traced Raster with shadows) comparing their results.

```matlab
import OceanMath.Vision.Geometry.PlaneGeometry
import OceanMath.Vision.Geometry.SphereGeometry
import OceanMath.Graphics.Scene.Material
import OceanMath.Graphics.Scene.SceneObject
import OceanMath.Graphics.Scene.Scene
import OceanMath.Graphics.Lighting.AmbientLight
import OceanMath.Graphics.Lighting.PointLight
import OceanMath.Graphics.Shading.BlinnPhongStrategy
import OceanMath.Graphics.Renderer.VectorMeshRenderer
import OceanMath.Graphics.Renderer.RayTraceRenderer
import OceanMath.Geometry.CameraProjection
import OceanMath.Geometry.Transform3D

% 1. Setup Scene
scene = Scene();

% 2. Setup Ambient and Omni-directional Light sources
scene.addLight(AmbientLight('Intensity', [0.1; 0.1; 0.1]));
scene.addLight(PointLight('Position', [-3; 3; 1], 'Intensity', [1.0; 1.0; 1.0]));

% 3. Create Sphere with shiny material properties
matSphere = Material('Ambient', [0.1; 0.0; 0.0], 'Diffuse', [0.8; 0.0; 0.0], ...
                     'Specular', [0.6; 0.6; 0.6], 'Shininess', 32, 'Reflectivity', [0.2; 0.2; 0.2]);
sphere = SceneObject(SphereGeometry('Center', [0; 0; 0], 'Radius', 1.2), matSphere, ...
                     'Pose', Transform3D.translation(0, 0, 4));
scene.addObject(sphere);

% 4. Define camera sensor properties
K = [500, 0, 80; 0, 500, 60; 0, 0, 1]; % 160x120 camera view
T_w2c = eye(4);
camera = CameraProjection('Intrinsics', K, 'Extrinsics', T_w2c);

% 5. Shading model selection
strategy = BlinnPhongStrategy();

% 6. Render Mode A: Vector Flat shading
fig = figure;
ax1 = subplot(1, 3, 1);
flatRenderer = VectorMeshRenderer('RenderDimension', '2D', 'ShadingMode', 'Flat');
flatRenderer.render(scene, camera, 'Axes', ax1, 'LightingStrategy', strategy);
title(ax1, 'Flat Shading');

% 7. Render Mode B: Vector Gouraud shading
ax2 = subplot(1, 3, 2);
gouraudRenderer = VectorMeshRenderer('RenderDimension', '2D', 'ShadingMode', 'Gouraud');
gouraudRenderer.render(scene, camera, 'Axes', ax2, 'LightingStrategy', strategy);
title(ax2, 'Gouraud Shading');

% 8. Render Mode C: Ray-Traced raster viewport with shadows
ax3 = subplot(1, 3, 3);
rayTracer = RayTraceRenderer('Resolution', [160, 120]);
img = rayTracer.render(scene, camera, 'LightingStrategy', strategy);
image(ax3, img);
axis(ax3, 'image');
title(ax3, 'Ray-Traced viewport');
```

---
 
 ## 10. Running the Interactive 2D Transform Editor (Python)
 This recipe shows how to run the interactive 2D Transform Editor from the command line using the Python virtual environment.
 
 ```bash
 # Navigate to the project root directory
 cd "c:\Ocean Library\Matlab code practice"
 
 # Run the 2D transform editor script using the python virtual environment
 .venv\Scripts\python.exe Python\transform_editor_2d.py
 ```
 
 ---
 
 ## 11. Running the Interactive 3D Primitives Viewer (Python)
 This recipe shows how to run the interactive 3D Primitives Viewer from the command line using the Python virtual environment.
 
 ```bash
 # Navigate to the project root directory
 cd "c:\Ocean Library\Matlab code practice"
 
 # Run the 3D viewer script using the python virtual environment
 .venv\Scripts\python.exe Python\viewer_3d.py
 ```
 
 ---
 
 ## 12. Running the Interactive 2D Transform Editor (MATLAB)
 This recipe shows how to launch the 2D Transform Editor directly in MATLAB.
 
 ```matlab
 % Run this in the MATLAB command window
 run('Experiments/Scripts/Computer Vision/exp_transform_editor_2d.m');
 ```
 
 ---
 
 ## 13. Running the Interactive 3D Primitives Viewer (MATLAB)
 This recipe shows how to launch the 3D Primitives Viewer directly in MATLAB.
 
 ```matlab
 % Run this in the MATLAB command window
 run('Experiments/Scripts/Computer Vision/exp_viewer_3d.m');
 ```

 ---
 
 ## 14. Statistical Background Subtraction & Foreground Extraction (MATLAB)
 This recipe shows how to train the `BackgroundSubtractor` class using sequence frames (with robust MedianMAD estimation), extract a 4-channel RGBA foreground, and composite it onto a new background.
 
 ```matlab
 import OceanMath.Image.Resampling.BackgroundSubtractor
 
 % 1. Create a dummy video training sequence (10 frames of 200x200x3)
 H = 200; W = 200; C = 3; N = 10;
 baseFrame = 0.5 * ones(H, W, C);
 trainFrames = repmat(baseFrame, [1, 1, 1, N]);
 % Introduce transient outliers in 2 frames (20% of duration)
 trainFrames(80:120, 80:120, :, 1) = 1.0;
 trainFrames(80:120, 80:120, :, 2) = 1.0;
 
 % 2. Train the background subtractor using robust MedianMAD method
 bs = BackgroundSubtractor(trainFrames, 'Method', 'MedianMAD');
 
 % 3. Create a test frame with a foreground object (a 15x15 white square at center)
 testFrame = baseFrame;
 testFrame(90:105, 90:105, :) = 1.0;
 
 % 4. Apply difference keying to get 4-channel RGBA output
 [fgMask, fgRGBA] = bs.apply(testFrame, 15.0);
 
 % 5. Apply morphological post-processing and extract centroids
 [cleanMask, centroids] = bs.postProcess(fgMask, 'MinArea', 50, 'MorphSize', 2);
 
 % 6. Composite the extracted RGBA foreground over a new background (e.g. green canvas)
 newBg = zeros(H, W, 3);
 newBg(:, :, 2) = 0.8; % Green background
 composite = bs.blendOn(fgRGBA, newBg);
 
 fprintf('Detected %d connected components.\n', size(centroids, 1));
 if ~isempty(centroids)
     fprintf('Centroid 1: X = %.2f, Y = %.2f\n', centroids(1, 1), centroids(1, 2));
 end
 ```
