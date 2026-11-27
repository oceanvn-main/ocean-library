# 3D Perspective-Correct Software Rasterizer

## Overview
The 3D Texture-Mapped Software Rasterizer (Szeliski Exercise 3.27) implements a perspective-correct triangle rasterization pipeline entirely in software. It is integrated as a custom viewer and programmatic scripts within the Ocean Library to demonstrate how low-level graphics processing units (GPUs) render 3D textured polygons.

To prevent the classic **Affine Texture Mapping distortion** (which manifests as a diagonal crease or kink across rendering polygons), the rasterizer interpolates inverse camera-space depth (1/Z) and pre-divided texture coordinates (u/Z, v/Z) across screen pixels using barycentric coordinates, followed by a per-pixel perspective divide.

---

## Interactive App Usage

The perspective-correct rasterizer is integrated into the interactive **3D Primitives Viewer** app:
```matlab
% Launch the interactive viewer
app = Viewer3D();
```
To enable texture mapping in the app, open the **Object** settings tab, toggle **Texture Mapping** to ON, and select a preset (House, Cube, or Pyramid).

To execute the programmatic software rasterizer script directly:
```matlab
exp_texture_viewer;
```
This script runs a complete 7-step software pipeline, generating a perspective-correct 3D cube rendered to a flat 2D pixel canvas and saving the output to `Results/Computer Vision/ex327_texture_demo.png`.

---

## Technical and Mathematical Details

### 1. The Affine Mapping Artifact
Standard 2D bilinear or affine interpolation maps texture coordinates $(u, v)$ linearly across screen-space coordinates $(x, y)$. However, because perspective projection is non-linear, equal steps in screen space do not correspond to equal steps in 3D space. Under triangulation, this linear screen-space approximation leads to an abrupt change in interpolation direction along the diagonal edges of triangles, creating a visual kink.

### 2. Perspective-Correct Interpolation (Exercise 3.27)
To achieve correct perspective mapping, we interpolate quantities that are linear in screen space:
* **Inverse camera-space depth:** $w = \frac{1}{Z}$
* **Projective texture coordinates:** $u' = \frac{u}{Z}$ and $v' = \frac{v}{Z}$

For any screen-space pixel $(x, y)$ inside a triangle with vertices $P_1, P_2, P_3$:
1. Solve for screen-space barycentric coordinates $(\alpha, \beta, \gamma)$:
   $$\alpha P_1 + \beta P_2 + \gamma P_3 = (x, y)$$
   $$\alpha + \beta + \gamma = 1$$
2. Interpolate the projective attributes at the pixel:
   $$w_{interp} = \alpha \left(\frac{1}{Z_1}\right) + \beta \left(\frac{1}{Z_2}\right) + \gamma \left(\frac{1}{Z_3}\right)$$
   $$u'_{interp} = \alpha \left(\frac{u_1}{Z_1}\right) + \beta \left(\frac{u_2}{Z_2}\right) + \gamma \left(\frac{u_3}{Z_3}\right)$$
   $$v'_{interp} = \alpha \left(\frac{v_1}{Z_1}\right) + \beta \left(\frac{v_2}{Z_2}\right) + \gamma \left(\frac{v_3}{Z_3}\right)$$
3. Perform the **perspective divide** at each pixel to recover true texture coordinates:
   $$u_{true} = \frac{u'_{interp}}{w_{interp}}$$
   $$v_{true} = \frac{v'_{interp}}{w_{interp}}$$
4. Use $(u_{true}, v_{true})$ to sample the texture map (using bilinear interpolation).

---

## See Also
* [Computer Vision Primitives](computer_vision_primitives.md)
* [Image Resampling & Warping](parametric_warping.md)
