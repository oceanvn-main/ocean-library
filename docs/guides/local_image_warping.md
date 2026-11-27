# Local Image Warping (Ex 3.23)

## Overview

The `+OceanMath/+Image/+Resampling/` package implements five local image warping strategies covering Ex 3.23 of Szeliski's *Computer Vision: Algorithms and Applications*. All methods perform **backward (inverse) mapping** with bilinear interpolation, ensuring no holes.

---

## Classes

### PiecewiseAffineWarp
**File:** `+OceanMath/+Image/+Resampling/PiecewiseAffineWarp.m`  
**Continuity:** C0 (displacements continuous, but derivatives can jump at triangle boundaries)

Partitions the image domain via Delaunay triangulation over control points. Each pixel finds its enclosing source triangle and maps back using barycentric coordinates.

```matlab
tri = delaunay(pts_src(:,1), pts_src(:,2));
I_warp = OceanMath.Image.Resampling.PiecewiseAffineWarp.warp(I_src, pts_src, pts_tgt, tri);
```

---

### ThinPlateSpline
**File:** `+OceanMath/+Image/+Resampling/ThinPlateSpline.m`  
**Continuity:** C-infinity (globally smooth everywhere)

Fits a smooth displacement field by solving a linear system using RBF kernels. Supports kernels: `thinplate`, `gaussian`, `multiquadric`, `inversedistance`.

```matlab
tps = OceanMath.Image.Resampling.ThinPlateSpline(pts_tgt, pts_src, 'thinplate');
I_warp = tps.warpImage(I_src);
```

---

### BeierNeelyWarp  _(Ex 3.23 Part 2)_
**File:** `+OceanMath/+Image/+Resampling/BeierNeelyWarp.m`

Feature-based morphing using corresponding directed line segments. For each target pixel, computes the projection (u, v) onto each line pair and aggregates weighted displacements with Gaussian distance decay.

```matlab
I_warp = OceanMath.Image.Resampling.BeierNeelyWarp.warp(I_src, ...
    lines_tgt_P, lines_tgt_Q, lines_src_P, lines_src_Q);
```

**Parameters:**
| Variable     | Meaning |
|--------------|---------|
| `lines_tgt_P` | Target line start points (L x 2) |
| `lines_tgt_Q` | Target line end points (L x 2) |
| `lines_src_P` | Source line start points (L x 2) |
| `lines_src_Q` | Source line end points (L x 2) |

---

### BSplineWarp  _(Ex 3.23 Part 3)_
**File:** `+OceanMath/+Image/+Resampling/BSplineWarp.m`  
**Continuity:** C2 (curvature-continuous)

Free-Form Deformation (FFD) using a regular grid of control points and tensor-product cubic B-spline basis functions. Each target pixel accumulates weighted displacements from its 4x4 surrounding control point neighborhood, clamped at boundaries.

**Cubic B-spline basis functions** (for t in [0, 1]):

```
B0(t) = (1 - t)^3 / 6
B1(t) = (3*t^3 - 6*t^2 + 4) / 6
B2(t) = (-3*t^3 + 3*t^2 + 3*t + 1) / 6
B3(t) = t^3 / 6
```

```matlab
I_warp = OceanMath.Image.Resampling.BSplineWarp.warp(I_src, pts_src, pts_tgt, [Ry, Rx]);
```

**Parameters:**
| Argument  | Description |
|-----------|-------------|
| `I_src`   | Source image (H x W x C) |
| `pts_src` | Original control grid (N x 2, column-major order) |
| `pts_tgt` | Deformed control grid (N x 2) |
| `grid_sz` | Grid size `[Ry, Rx]` |
| `out_size`| (Optional) output size `[H_tgt, W_tgt]` |

---

### DisplacementFieldWarp  _(Ex 3.23 Part 4 — backend)_
**File:** `+OceanMath/+Image/+Resampling/DisplacementFieldWarp.m`

Forward splatting warp from a dense `(Dx, Dy)` displacement field. Used as the backend for the Paintbrush flow mode in the GUI — the Paintbrush tab performs backward bilinear warp inline for maximum responsiveness (no full re-render on drag).

```matlab
I_warp = OceanMath.Image.Resampling.DisplacementFieldWarp.warp(I_src, Dx, Dy);
```

---

## GUI Integration (Tab 8: Local Warping)

The `ComputerVisionSandbox.m` GUI (Tab 8) supports all five modes interactively:

| Mode | Interaction |
|------|-------------|
| **Piecewise Affine** | Drag red circle handles; Delaunay triangulation redraws in orange |
| **Thin Plate Spline** | Drag red circle handles; smooth global displacement |
| **Gaussian / Multiquadric / Inverse Distance RBF** | Drag red circle handles; localized smooth RBF warp |
| **Beier-Neely (Lines)** | Drag red circle (P start) or orange square (Q end) endpoints of directed orange lines |
| **B-Spline Grid** | Drag red circle handles on the B-spline control grid |
| **Paintbrush (Flow)** | Click and drag anywhere on the canvas; brush size and strength adjustable |

**To run the GUI:**
```matlab
app = ComputerVisionSandbox();
app.show();
```

Then navigate to **Tab 8: Local Warping** and select a mode from the dropdown.

---

## Experiment Script

```matlab
run('Experiments/Scripts/Computer Vision/exp_local_warping.m')
```

Produces a 2x3 dashboard figure comparing all five warp methods side-by-side on a synthetic checkerboard, with timing annotations and mesh/line overlays. Results saved to `Experiments/Scripts/Computer Vision/Results/local_warping_results.png`.

---

## See Also

- `exp_mesh_warping.m` — Piecewise Affine vs TPS comparison
- `exp_forward_vs_inverse_warp.m` — Hole artifacts in forward mapping
- `BeierNeelyWarp.m` — Full Beier-Neely implementation details
- Szeliski §3.6 — Local Image Warping
