# Guide: Mesh-Based Warping & Advanced Geometric Transformations

## Overview

Mesh-based warping and feature-based transformations are methods of image deformation defined by a set of geometric control structures. This guide covers the suite of techniques implemented in the Ocean Library:

1. **Piecewise Affine Warping** (`PiecewiseAffineWarp`): Warps an image defined over a triangular mesh. Within each triangle, coordinate mapping is linear and affine, offering C0 continuity.
2. **Quadrilateral Grid Warping** (`BilinearQuadWarp`): Warps an image using a quadrilateral grid by subdividing each quad into two triangles and calling `PiecewiseAffineWarp`.
3. **Radial Basis Function (RBF) Interpolation** (`ThinPlateSpline`): Smoothly interpolates sparse landmark points using swappable kernels:
   *   **Thin Plate Spline (TPS)**: $U(r) = r^2 \log(r)$ kernel.
   *   **Gaussian RBF**: $\exp(-\beta r^2)$ kernel.
   *   **Multiquadric RBF**: $\sqrt{r^2 + c^2}$ kernel.
   *   **Inverse Distance Weighting (IDW)**: $1 / (r^p + \text{eps})$ kernel.
4. **Forward Displacement Warping** (`DisplacementFieldWarp`): Warps an image forward using displacement fields $(dx, dy)$ via bilinear splatting and neighborhood hole filling.
5. **Line-Based Warping (Beier-Neely Algorithm)** (`BeierNeelyWarp`): Warps an image using oriented directed line segments, mapping points based on proportional projection coordinates.
6. **Content-Adapted Meshes** (`ContentAdaptedMesh`): Generates non-uniform meshes whose vertices are concentrated along high-contrast boundaries to represent image content efficiently.

---

## Usage

### Swappable RBF Interpolation
```matlab
% Fit target-to-source landmarks using a Gaussian RBF kernel
tps = OceanMath.Image.Resampling.ThinPlateSpline(pts_tgt, pts_src, ...
    OceanMath.Image.Resampling.GaussianRBFKernel('Beta', 1.5e-5));

% Warp image
I_warped = tps.warpImage(I_src);
```

### Forward Displacement Warping
```matlab
I_warped = OceanMath.Image.Resampling.DisplacementFieldWarp.warp(I_src, Dx, Dy);
```

### Beier-Neely Line Warping
```matlab
I_warped = OceanMath.Image.Resampling.BeierNeelyWarp.warp(I_src, pts_tgt_P, pts_tgt_Q, pts_src_P, pts_src_Q);
```

### Content-Adapted Mesh Generation
```matlab
[pts, tri] = OceanMath.Image.Resampling.ContentAdaptedMesh.generate(I_src, 100, 'MinDistance', 15);
```

---

## Examples

The script `exp_advanced_mesh_warping.m` runs a full evaluation of all four advanced components: RBF kernel comparisons, forward displacement warping, Beier-Neely line morphing, and content-adapted trimesh generation.

To run the experiment:
```matlab
run('Experiments/Scripts/Computer Vision/exp_advanced_mesh_warping.m')
```

---

## See Also

- [Pixel Transformations Guide](pixel_transforms.md)
- [Spatial Filtering & Lifting Guide](filtering_and_lifting.md)
- [Image Pyramids Guide](pyramids.md)
