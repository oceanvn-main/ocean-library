# Computer Vision Geometric Primitives and Transformations Guide

## Overview
Geometric primitives and spatial transformations form the mathematical foundation of computer vision, image formation, and 3D reconstruction. This guide details the mathematical principles and package implementation of 2D/3D homogeneous transforms, 3D rotations, pinhole camera projections, and lens distortion aberration models.

The mathematical core is isolated in `OceanMath.Vision`, ensuring full reusability for algorithms, and interactive exploration is integrated into `OceanViz.App.Computer Vision.ComputerVisionSandbox`.

---

## Mathematical Models

### 1. 2D Transformations
2D projective transformations (homographies) map points from one plane to another. Using homogeneous coordinates:
`x' = H * x`

Where `H` is a 3x3 matrix:
`[x']   [h11  h12  h13]   [x]`
`[y'] = [h21  h22  h23] * [y]`
`[w']   [h31  h32  h33]   [1]`

Normalized inhomogeneous coordinates are recovered via `(x'/w', y'/w')`. Special cases of homographies include:
- **Translation**: 2 degrees of freedom (tx, ty).
- **Rotation**: 1 degree of freedom (theta).
- **Scale**: 2 degrees of freedom (sx, sy).
- **Shear**: 2 degrees of freedom (shx, shy).
- **Euclidean / Rigid**: 3 degrees of freedom (translation + rotation: tx, ty, theta).
- **Similarity**: 4 degrees of freedom (scaled rotation + translation: tx, ty, theta, s).
- **Stretch/squash**: 4 degrees of freedom (non-uniform scaling + translation: tx, ty, sx, sy).
- **Affine**: 6 degrees of freedom (combines translation, rotation, scale, and shear).

### 2. 3D Rotations
Representing 3D rotations requires addressing singularities (gimbal lock) and interpolation performance:
- **Euler Angles (Roll, Pitch, Yaw)**: Highly intuitive representation using successive rotations around intermediate axes (e.g. ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)). However, it suffers from **Gimbal Lock** when the pitch angle reaches +/- 90 degrees, causing the first and third axes to align and losing a degree of freedom.
- **Unit Quaternions**: A 4D hyper-spherical representation `q = [w, x, y, z]` that completely avoids singularities and supports Spherical Linear Interpolation (SLERP) to find the shortest arc geodesic path between orientations.
- **Axis-Angle**: Defines a unit rotation axis `u` and rotation magnitude `theta`.
- **Exponential Map (so(3))**: A 3x1 vector mapping Lie algebra coordinates directly to the rotation group SO(3) via Rodrigues' formula.

#### Gimbal Lock Mathematical Singularity
In the ZYX Euler convention, when the Pitch angle equals 90 degrees:
* `cos(pitch) = 0`, `sin(pitch) = 1`
Substituting these values into the composite rotation matrix `R = Rz(yaw) * Ry(pitch) * Rx(roll)` yields:
`R = [ 0, -sin(yaw - roll),  cos(yaw - roll); ...`
`      0,  cos(yaw - roll),  sin(yaw - roll); ...`
`     -1,                0,                0 ]`
Notice that the matrix elements depend purely on the difference `yaw - roll`. Yaw and Roll are no longer independent degrees of freedom. Changing Yaw by alpha and Roll by -alpha results in the exact same physical rotation matrix. This collinearity of the first and third gimbal axes reduces the rotation space from 3D to 2D, causing a lock.

### 3. Camera Projection
Pinhole cameras map 3D world points `P_w` to 2D image coordinates `p_img`.
1. **Extrinsics**: Converts world space to local camera frame space:
   `P_cam = R_w2c * P_w + t_w2c`
2. **Intrinsics (K)**: Defines focal lengths (fx, fy), principal offset points (cx, cy), and skew (s):
   `K = [fx,  s, cx; 0, fy, cy; 0,  0,  1]`
3. **Projection Models**:
   - *Perspective*: Standard projection scaling by depth `z`.
   - *Orthographic*: Parallel ray projection ignoring depth.
   - *Weak-Perspective*: Scaled orthographic projection where depth is approximated as a constant average `Z_average` across all points.

### 4. Lens Distortion
Physical glass lenses introduce radial and decentering distortions:
- **Radial**: Pincushion or barrel distortion expanding or compressing points based on radial distance `r^2 = x^2 + y^2` from the principal point:
  `x_distorted = x * (1 + k1*r^2 + k2*r^4 + k3*r^6)`
- **Tangential**: Caused by lens decentering or tilt relative to the sensor, introducing asymmetrical shear:
  `dx = 2*p1*x*y + p2*(r^2 + 2*x^2)`
  `dy = p1*(r^2 + 2*y^2) + 2*p2*x*y`

### 5. Transformation Manifolds (Lie Groups) and Tangent Spaces (Lie Algebras)
In pattern recognition and statistical learning, distance metrics should ideally be invariant to continuous spatial transformations (like SO(2) rotation and SE(2) rigid body motion).
* **The Transformation Manifold (Lie Group)**: Applying a continuous 1-parameter transformation (e.g. rotation by theta) to an image vector x_i (residing in high-dimensional pixel space) traces out a curved 1-dimensional manifold C(theta). Finding the shortest Euclidean distance between two such curved manifolds is a non-convex, computationally expensive optimization problem.
* **The Tangent Space (Lie Algebra)**: The derivative of the image vector with respect to the transformation parameter, evaluated at the identity (theta = 0), generates a tangent vector T_rot = -y * dI/dx + x * dI/dy (where dI/dx, dI/dy are spatial image gradients). The original point plus scalar multiples of this tangent vector form a flat, linear subspace (tangent space) that locally approximates the curved manifold.
* **Tangent Distance**: Measures the shortest Euclidean distance between the two linear tangent subspaces. This is solved extremely efficiently using standard linear least-squares.

### 6. Co-vectors and the Transposed Inverse (H^-T)
In projective geometry, geometric primitives represent two different algebraic structures:
* **Vectors (Points)**: A spatial point coordinate x transforms via standard multiplication: `x' = H * x`.
* **Co-vectors (Lines, Planes, Surface Normals)**: Represent boundary constraints or perpendicular directions. A point x lies on a line l if and only if `l^T * x = 0`. Under a transformation H, the intersection property must be preserved: `(l')^T * x' = 0`.
* **The H^-T Derivation**: Substituting the point transformation gives:
  `(l')^T * H * x = 0`
  `((l')^T * H) * x = 0`
  For this to hold for any point x on the original line, the grouped term must equal the original line co-vector:
  `l^T = (l')^T * H`
  Transposing both sides:
  `l = H^T * l'`
  Isolating the transformed co-vector `l'`:
  `l' = (H^T)^(-1) * l = H^(-T) * l`
Under non-orthogonal operations (like shearing or non-uniform scaling), using the point matrix H to transform a co-vector causes it to physically detach from its points and point in the wrong direction. The transposed inverse `H^(-T)` mathematically guarantees that perpendicularity and intersection properties are preserved.

---

## Usage

The `OceanMath.Vision` sub-package provides clean static and object-oriented interfaces:

### 1. Homogeneous 2D Affine Warping
```matlab
import OceanMath.Geometry.Transform2D

% Define Euclidean / Rigid transform (translation + rotation, 3 DOF)
SE = Transform2D.rigid(2.0, -1.0, 45);

% Define Similarity transform (scaled rotation + translation, 4 DOF)
Sim = Transform2D.similarity(2.0, -1.0, 45, 1.5);

% Define Stretch/squash transform (changes aspect ratio + translation, 4 DOF)
SS = Transform2D.stretchSquash(2.0, -1.0, 1.5, 0.8);

% Define general affine matrix combining translation, rotation, scale, shear (6 DOF)
A = Transform2D.affine(2.0, -1.0, 45, 1.5, 1.5, 0, 0);

% Transform a [2 x N] coordinate grid of points
pts = [1, 2, 3; 4, 5, 6];
warpedPts = Transform2D.transformPoints(A, pts);
```

### 1b. Homogeneous 3D Affine and Projective Warping
```matlab
import OceanMath.Geometry.Transform3D

% Define Translation (tx, ty, tz)
T = Transform3D.translation(1.0, -2.0, 3.5);

% Define Rigid transform (translation + rotation, 6 DOF)
R_mat = [0, -1, 0; 1, 0, 0; 0, 0, 1]; % 90 deg rotation around Z
SE = Transform3D.rigid(1.0, -2.0, 3.5, R_mat);

% Define Similarity transform (translation + rotation + uniform scaling, 7 DOF)
Sim = Transform3D.similarity(1.0, -2.0, 3.5, R_mat, 1.5);

% Define general Affine matrix from 3x3 linear matrix and translation (12 DOF)
A_mat = [1.2, 0.1, 0.0; -0.1, 0.9, 0.2; 0.0, 0.0, 1.0];
A = Transform3D.affine(A_mat, 1.0, -2.0, 3.5);

% Define 3D Shear (6 DOF)
Sh = Transform3D.shear(0.1, 0.0, 0.2, 0.0, 0.0, 0.3);

% Transform a [3 x N] coordinate grid of points
pts = [1, 2, 3; 4, 5, 6; 7, 8, 9];
warpedPts = Transform3D.transformPoints(A, pts);
```

### 2. 3D Rotation Conversions and SLERP
```matlab
import OceanMath.Geometry.Rotation3D

% Euler to matrix
R = Rotation3D.eulerToMatrix(10, -20, 45);

% Matrix to quaternion
q1 = Rotation3D.matrixToQuat(R);
q2 = [0, 1, 0, 0]; % 180 deg around X-axis

% Perform spherical linear interpolation at t = 0.5
q_interp = Rotation3D.slerp(q1, q2, 0.5);
R_interp = Rotation3D.quatToMatrix(q_interp);
```

### 3. Camera Projection modeling
```matlab
import OceanMath.Geometry.CameraProjection

% Setup intrinsics and extrinsics
K = [500, 0, 320; 0, 500, 240; 0, 0, 1];
T_w2c = eye(4); T_w2c(3,4) = -8; % Translate camera 8 units back along Z-axis

proj = CameraProjection('Intrinsics', K, 'Extrinsics', T_w2c, 'ProjectionType', 'Perspective');

% Project a 3D point cloud
pts3D = [0, 1, -1; 0, 2, 2; 5, 5, 6];
[pts2D, inFront] = proj.project(pts3D);
```

### 4. Tangent Distance Invariant Metric
```matlab
% Given reference image x1 and target image x2 (16x16 flattened vectors)
% and horizontal (T1_tx, T2_tx), vertical (T1_ty, T2_ty), and rotational (T1_rot, T2_rot) tangents
V1 = [T1_tx, T1_ty, T1_rot]; % Tangent space 1 (size 256x3)
V2 = [T2_tx, T2_ty, T2_rot]; % Tangent space 2 (size 256x3)

% Solve least-squares orthogonal distance between the two linear hyperplanes:
% Minimize || (x1 + V1*c1) - (x2 + V2*c2) ||^2
e_diff = x1 - x2;
M = [V1, -V2];
c_opt = - (M' * M) \ (M' * e_diff);

% Compute closest points and Tangent Distance
p1 = x1 + V1 * c_opt(1:3);
p2 = x2 + V2 * c_opt(4:6);
d_tangent = norm(p1 - p2);
```

### 5. Co-vector and Surface Normal Transformations
```matlab
% Point transformation matrix (containing shear or scale)
H = [1.5, 1.0, 0.0; ...
     0.0, 0.8, 0.0; ...
     0.5, 0.0, 1.2];

% 1. Points transform via standard matrix multiplication
pts_warped = H * pts;

% 2. Surface normals or line equations (co-vectors) transform via the Transposed Inverse
H_inv_trans = inv(H)';
normals_warped = H_inv_trans * normals;

% Normalize to keep unit vectors
normals_warped_unit = normals_warped ./ sqrt(sum(normals_warped.^2, 1));
```

---

## Examples

### Complete 3D Camera Projection Pipeline
```matlab
% 1. Create a 3D point cloud of a wireframe cube
[X, Y, Z] = meshgrid([-1, 1], [-1, 1], [-1, 1]);
pts3D = [X(:)'; Y(:)'; Z(:)'];

% 2. Place camera in world at (0, 0, -5) looking straight down Z-axis
R_w2c = eye(3);
t_w2c = [0; 0; 5]; % inv(T_c2w) translation
T_w2c = [R_w2c, t_w2c; 0, 0, 0, 1];

% 3. Define standard 640x480 intrinsic settings
K = [500, 0, 320; 0, 500, 240; 0, 0, 1];

% 4. Model and execute projection
proj = OceanMath.Geometry.CameraProjection('Intrinsics', K, 'Extrinsics', T_w2c);
[pts2D, inFront] = proj.project(pts3D);

% pts2D contains the projected [2 x 8] image coordinate mesh
disp(pts2D);
```

## Interactive Exploration Tools

Both MATLAB and Python versions of the interactive applications are available to explore 2D transformations (Exercise 2.2) and 3D projections (Exercise 2.3) visually in a dark-mode user interface.

### 1. Interactive 2D Transform Editor

Allows you to interactively manipulate 2D shapes under Translation, Rigid, Similarity, Affine, and Perspective (Homography) motion models.

- **Implementations**:
  - **MATLAB Class**: [TransformEditor2D.m](file:///c:/Ocean%20Library/Matlab%20code%20practice/+OceanViz/+App/Computer%20Vision/TransformEditor2D.m) (launched via [exp_transform_editor_2d.m](file:///c:/Ocean%20Library/Matlab%20code%20practice/Experiments/Scripts/Computer%20Vision/exp_transform_editor_2d.m))
  - **Python Script**: [transform_editor_2d.py](file:///c:/Ocean%20Library/Matlab%20code%20practice/Python/transform_editor_2d.py)
- **Features**:
  - Draw new shape boundaries via Shift + Left-Click & Drag.
  - Live numeric readouts of the 3x3 homography matrix H.
  - Save and load configurations to disk via JSON files.
- **Controls**:
  - Shift + Left-Click & Drag: Draw a new rectangle on the canvas.
  - Left-Click on a shape: Selects the shape.
  - Drag corner handles (white dots in Python, colored dots in MATLAB): Apply deformation based on the selected motion model.
  - Delete or Backspace: Delete the selected shape.

### 2. Interactive 3D Primitives Viewer

Renders 3D points, lines, and polygons under Perspective and Orthographic camera projection models.

- **Implementations**:
  - **MATLAB Class**: [Viewer3D.m](file:///c:/Ocean%20Library/Matlab%20code%20practice/+OceanViz/+App/Computer%20Vision/Viewer3D.m) (launched via [exp_viewer_3d.m](file:///c:/Ocean%20Library/Matlab%20code%20practice/Experiments/Scripts/Computer%20Vision/exp_viewer_3d.m))
  - **Python Script**: [viewer_3d.py](file:///c:/Ocean%20Library/Matlab%20code%20practice/Python/viewer_3d.py)
- **Features**:
  - Interactive sliders to control camera extrinsics (Yaw/Pitch/Roll, TX/TY/TZ) and object transforms.
  - Click-and-drag directly on the canvas to orbit the camera interactively.
  - Standard graphics pipeline with near-plane clipping.
  - Painter's Algorithm implementation for hidden-surface removal of colored polygons.
  - Pre-loaded presets (House, Cube, Pyramid) and a text box for custom scene command import (P, L, POLY commands).

---

## See Also
- `OceanMath.Quat` — Unit quaternion operations.
- `OceanMath.Slerp` — Spherical linear interpolation.
- `OceanViz.Core` — Visual themes and styles engine.
