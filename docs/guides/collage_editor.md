# 2D Image Collage Editor

## Overview
The 2D Image Collage Editor (Szeliski Exercise 3.26) is an interactive design tool to combine, transform, and layer multiple cropped images on a single digital canvas. Key capabilities include:
1. **Interactive Crop Selection**: Import images from disk and specify custom crop regions using a separate selection window.
2. **Layering & Ordering**: Paste cropped regions as independent layers and re-order them (Bring to Front, Send to Back) or delete layers.
3. **Parametric Transformations**: Interactively drag corner handles to deform layers under **Translation**, **Rigid**, **Similarity**, **Affine**, or **Perspective (Homography)** models.
4. **Real-time Rendering**: Composites warped layers onto a cached canvas background using inverse projective mapping and bilinear sampling at 60+ FPS.

---

## Usage

The collage editor is integrated into the sandbox application as **Tab 10: 2D Collage Editor**.

To run programmatically:
```matlab
% Set up paths
addpath('c:\Ocean Library\Matlab code practice\+OceanViz\+App\Computer Vision');
addpath(genpath('c:\Ocean Library\Matlab code practice'));

% Launch Sandbox
app = ComputerVisionSandbox();
app.show();
```

To run the programmatic collage composition experiment:
```matlab
exp_collage_editor;
```

---

## Technical Details

### 1. Homography Formulation
Each layer's deformation is parameterized by a 3x3 homography matrix `H` mapping coordinates from the local cropped bounding box `[0,0; w,0; w,h; 0,h]` to target canvas locations:

```
[ u ]     [ h00 h01 h02 ] [ x ]
[ v ]  =  [ h10 h11 h12 ] [ y ]
[ w ]     [ h20 h21 h22 ] [ 1 ]
```
Where `(u/w, v/w)` are the final destination coordinates on the 800x600 canvas.

### 2. Constraints Solver
When dragging a handle `k` to mouse position `(x, y)`, the target coordinates `Q` are updated and a new homography `H` is computed:
* **Translation**: Translates all corners equally by `dx, dy`.
* **Rigid**: Solves Procrustes rotation mapping around the opposite corner.
* **Similarity**: Solves a 4-parameter linear system for rotation, scaling, and translation.
* **Affine**: Solves a 6-parameter system mapping three corners (dragging, opposite, and one adjacent).
* **Perspective**: Solves an 8-parameter homography system (Direct Linear Transform) mapping all 4 corners.

---

## See Also
* [Local Image Warping](local_image_warping.md)
* [Parametric Warping & Mipmapping](parametric_warping.md)
