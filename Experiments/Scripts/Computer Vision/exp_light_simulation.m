% exp_light_simulation.m
%
% OBJECTIVE:
%   Perform a scientific comparison of 3D surface rendering strategies:
%     1. Vector Flat Shading (constant per-polygon color)
%     2. Vector Gouraud Shading (interpolated per-vertex color)
%     3. Ray-Traced Raster Shading (pixel-by-pixel with occlusion shadow mapping)
%   Under a point light source orbiting a sphere inside a room.
%
% ASSUMPTIONS:
%   - Monochromatic light source (RGB = [1, 1, 1]).
%   - Local illumination models (Lambertian and Blinn-Phong) govern surface reflections.
%   - No secondary ambient inter-reflections (global illumination radiosity is omitted).
%   - Perfect camera lens with zero radial distortion.
%
% AUTHOR: Antigravity AI
% DATE: 2026-06-06

clear; close all; clc;

% Import Vision packages
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
import OceanMath.Geometry.Rotation3D
import OceanViz.Core

fprintf('=== Starting Light Simulation Scientific Experiment ===\n');

%% Step 1: Initialize Scene and Geometries
scene = Scene();

% Sphere at center: Position [0; 0; 4], Radius 1.2
matSphere = Material('Ambient', [0.1; 0; 0], 'Diffuse', [0.8; 0; 0], ...
                     'Specular', [0.7; 0.7; 0.7], 'Shininess', 32.0, 'Reflectivity', [0.2; 0.2; 0.2]);
sphere = SceneObject(SphereGeometry('Center', [0; 0; 0], 'Radius', 1.2), matSphere, ...
                     'Pose', Transform3D.translation(0, 0, 4));

% Bounded Room Planes (floor, ceiling, walls)
matRoom  = Material('Ambient', [0.1; 0.1; 0.1], 'Diffuse', [0.7; 0.7; 0.7]);
matGreen = Material('Ambient', [0.0; 0.1; 0.0], 'Diffuse', [0.0; 0.7; 0.0]);
matBlue  = Material('Ambient', [0.0; 0.0; 0.1], 'Diffuse', [0.0; 0.0; 0.7]);

floorObj = SceneObject(PlaneGeometry('Point', [0; -2; 0], 'Normal', [0; 1; 0], 'IsFinite', true, 'Width', 4, 'Height', 4), matRoom);
backWall = SceneObject(PlaneGeometry('Point', [0; 0; 6], 'Normal', [0; 0; -1], 'IsFinite', true, 'Width', 4, 'Height', 4), matRoom);
leftWall = SceneObject(PlaneGeometry('Point', [-4; 0; 0], 'Normal', [1; 0; 0], 'IsFinite', true, 'Width', 4, 'Height', 4), matGreen);
rightWall = SceneObject(PlaneGeometry('Point', [4; 0; 0], 'Normal', [-1; 0; 0], 'IsFinite', true, 'Width', 4, 'Height', 4), matBlue);

scene.addObject(floorObj);
scene.addObject(backWall);
scene.addObject(leftWall);
scene.addObject(rightWall);
scene.addObject(sphere);

% Lights
ambient = AmbientLight('Intensity', [0.15; 0.15; 0.15]);
pointLight = PointLight('Position', [1.5; 1.5; 2.5], 'Intensity', [1.2; 1.2; 1.2]);
scene.addLight(ambient);
scene.addLight(pointLight);

% Camera setup: 160x120 camera sensor looking forward
K = [120, 0, 80; 0, 120, 60; 0, 0, 1];
camera = CameraProjection('Intrinsics', K, 'Extrinsics', eye(4));

strategy = BlinnPhongStrategy();

%% Step 2: Render Comparison
viz = Core('Theme', 'Dark');
fig = viz.createFigure('Light Shading Strategy Comparison');
set(fig, 'Position', [100, 100, 1100, 450]);

% Subplot A: Vector Flat Shading
axA = subplot(1, 3, 1, 'Parent', fig);
flatRenderer = VectorMeshRenderer('RenderDimension', '2D', 'ShadingMode', 'Flat');
flatRenderer.render(scene, camera, 'Axes', axA, 'LightingStrategy', strategy);
axis(axA, 'equal');
xlim(axA, [0, 160]);
ylim(axA, [0, 120]);
set(axA, 'YDir', 'reverse');
title(axA, 'Flat Shading (Per-Face)', 'Color', 'white');
grid(axA, 'off');

% Subplot B: Vector Gouraud Shading
axB = subplot(1, 3, 2, 'Parent', fig);
gouraudRenderer = VectorMeshRenderer('RenderDimension', '2D', 'ShadingMode', 'Gouraud');
gouraudRenderer.render(scene, camera, 'Axes', axB, 'LightingStrategy', strategy);
axis(axB, 'equal');
xlim(axB, [0, 160]);
ylim(axB, [0, 120]);
set(axB, 'YDir', 'reverse');
title(axB, 'Gouraud Shading (Interpolated)', 'Color', 'white');
grid(axB, 'off');

% Subplot C: Ray-Traced Shadows
axC = subplot(1, 3, 3, 'Parent', fig);
rayTracer = RayTraceRenderer('Resolution', [160, 120], 'MaxDepth', 1);
fprintf('Trigging Ray Trace (160x120 viewport)...\n');
tic;
img = rayTracer.render(scene, camera, 'LightingStrategy', strategy);
toc;
image(axC, img);
axis(axC, 'image');
set(axC, 'XTick', [], 'YTick', []);
title(axC, 'Ray-Traced viewport (with Shadows)', 'Color', 'white');

% Ensure Results directory exists and save output
scriptDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(scriptDir, '..', '..', '..', 'Results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end
saveas(fig, fullfile(resultsDir, 'light_sim_comparison.png'));
fprintf('Saved comparative figure to Results/light_sim_comparison.png\n');

%% Step 3: Quantitative Intensity Profile Sweep (Horizontal Scanline)
% Extract the horizontal intensity profile across the middle of the sphere (row 60)
rowIdx = 60;
flatProfile = zeros(1, 160);
% For Flat/Gouraud, we can evaluate the color at corresponding Z-buffered pixels or sample from the ray-traced image
% Sample Red channel intensity along the scanline
rayTraceProfile = img(rowIdx, :, 1);

fig2 = viz.createFigure('Horizontal Scanline Intensity Profile');
plot(1:160, rayTraceProfile, 'r-', 'LineWidth', 2.0);
xlabel('Pixel X-coordinate');
ylabel('Red Channel Intensity');
title('Scanline Intensity Profile (Z=4 Sphere Centerline)', 'Color', 'white');
grid on;

saveas(fig2, fullfile(resultsDir, 'light_sim_scanline.png'));
fprintf('Saved scanline plot to Results/light_sim_scanline.png\n');
fprintf('=== Experiment Completed Successfully ===\n');
