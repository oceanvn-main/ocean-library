# 3D Room Light Simulation Guide

## Overview
The 3D Room Light Simulation framework (`OceanMath.Vision`) provides a generic, modular, and extensible environment to simulate physical light interaction inside a room. It decouples the shape geometry, object materials, light source behaviors, reflection equations (shading strategies), and rendering pipelines.

Interactive visualization is provided in `OceanViz.App.Computer Vision.LightSimulationSandbox`.

---

## Architecture and Components

### 1. Shape Geometries
All shapes must implement `OceanMath.Vision.Geometry.IGeometry` defining:
* `[t, normal] = intersect(obj, rayOrigin, rayDir)`: Returns intersection depth and normal vector.
* `[vertices, faces] = toMesh(obj)`: Returns mesh data for polygon rendering.

Standard shapes:
* **PlaneGeometry**: Bounded or infinite flat rectangular segments.
* **SphereGeometry**: 3D quadratic ray-sphere intersections.

### 2. Scene Representation
* **Material**: Manages reflection coefficients:
  * `Ambient` (`Ka`): Background ambient color reflection.
  * `Diffuse` (`Kd`): Scattered diffuse reflection.
  * `Specular` (`Ks`): Mirrored specular highlight color.
  * `Shininess` (`alpha`): Sharpness of specular highlight.
  * `Reflectivity` (`Kr`): Mirror recursive reflection coefficient.
* **SceneObject**: Couples an `IGeometry` shape, a `Material` configuration, and a 4x4 pose coordinate transform matrix.
* **Scene**: Aggregates `SceneObject` elements and light sources.

### 3. Light Sources
All lights implement `OceanMath.Graphics.Lighting.ILight` defining:
* `[intensity, dirToLight, distance] = getLightAt(obj, point)`: Determines illumination vector, magnitude, and distance.

Standard lights:
* **AmbientLight**: Non-directional ambient color intensity.
* **PointLight**: Omni-directional source with quadratic distance attenuation: `1 / (1 + d + d^2)`.
* **DirectionalLight**: Parallel rays without attenuation.
* **SpotLight**: Conical light source with cut-off angle and angular falloff.

### 4. Shading Strategies
Decouples lighting computation from the rendering pipeline (`OceanMath.Graphics.Shading`):
* **LambertianStrategy**: Diffuse Lambertian reflection.
* **PhongStrategy**: Standard specular highlight based on reflection vector: `R = 2 * dot(N, L) * N - L`.
* **BlinnPhongStrategy**: Specular highlight using the half-angle vector: `H = (L + V) / norm(L + V)`.

### 5. Renderers
All renderers implement `OceanMath.Graphics.Renderer.IRenderer`:
* **VectorMeshRenderer**: Renders projected flat or Gouraud shaded wireframe meshes to a MATLAB axes using `patch`.
* **RayTraceRenderer**: Recursive pixel-by-pixel ray caster supporting shadows (occlusion rays) and recursive mirror reflections.

---

## Usage Example

The following example constructs a 3D scene containing a red sphere, a spotlight, and an ambient light, and renders it from a camera viewpoint using the ray tracer:

```matlab
% 1. Create the Scene
scene = OceanMath.Graphics.Scene.Scene();

% 2. Add an Ambient Light
scene.addLight(OceanMath.Graphics.Lighting.AmbientLight('Intensity', [0.15; 0.15; 0.15]));

% 3. Add a Point Light source
scene.addLight(OceanMath.Graphics.Lighting.PointLight(...
    'Position', [-2; 3; 2], ...
    'Intensity', [1.2; 1.2; 1.2]));

% 4. Define Material properties
matSphere = OceanMath.Graphics.Scene.Material(...
    'Ambient', [0.1; 0.0; 0.0], ...
    'Diffuse', [0.8; 0.1; 0.1], ...
    'Specular', [0.6; 0.6; 0.6], ...
    'Shininess', 32, ...
    'Reflectivity', [0.3; 0.3; 0.3]);

% 5. Instantiate a sphere at [0; 0; 4]
pose = OceanMath.Geometry.Transform3D.translation(0, 0, 4);
sphereObj = OceanMath.Graphics.Scene.SceneObject(...
    OceanMath.Vision.Geometry.SphereGeometry('Center', [0;0;0], 'Radius', 1.2), ...
    matSphere, ...
    'Pose', pose);
scene.addObject(sphereObj);

% 6. Define Pinhole Camera
K = [500, 0, 320; 0, 500, 240; 0, 0, 1];
T_w2c = eye(4); T_w2c(3, 4) = 0; % Camera at world origin
camera = OceanMath.Geometry.CameraProjection('Intrinsics', K, 'Extrinsics', T_w2c);

% 7. Define Shading Strategy and Renderer
strategy = OceanMath.Graphics.Shading.BlinnPhongStrategy();
rayTracer = OceanMath.Graphics.Renderer.RayTraceRenderer('Resolution', [160, 120], 'MaxDepth', 2);

% 8. Execute Ray Tracing
img = rayTracer.render(scene, camera, 'LightingStrategy', strategy);

% 9. Display
figure;
image(img);
axis image;
title('Ray-Traced Image');
```

---

## See Also
* `OceanMath.Vision.Geometry.IGeometry` — Spatial primitive interfaces.
* `OceanMath.Graphics.Shading.ILightingStrategy` — Reflection model equations.
* `OceanMath.Graphics.Renderer.IRenderer` — Render pipelines.
* `OceanViz.App.Computer Vision.LightSimulationSandbox` — Real-time interactive UI app.
