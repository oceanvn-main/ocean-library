% EXP_MESH_RCNN_GEOMETRY  Mesh R-CNN building blocks, Phase 1: 3-D geometry
%   (Szeliski, Computer Vision 2nd ed., Fig. 5.54 -- Gkioxari, Malik & Johnson 2019).
%
%   QUESTION: Mesh R-CNN predicts a coarse voxel grid, converts it to a mesh
%   with Cubify, and then REFINES the mesh. How much of a real shape does the
%   voxel -> cubify route lose, and does it preserve topology (holes) that a
%   sphere-deforming method (Pixel2Mesh) can never create?
%
%   PIPELINE (all components from the library, no toolboxes):
%     FurnitureShapeFactory  procedural chairs / tables / stools (exact solids)
%     BoxAssembly            exact inside test + exact union-surface samples (GT)
%     Voxelizer              solid -> G^3 occupancy (the voxel branch's target)
%     Cubify                 occupancy -> watertight mesh (Mesh R-CNN Sec. 3.1)
%     TriMesh                genus / volume / watertightness of the result
%     IcoSphere              the genus-0 template of sphere-based methods
%
%   MEASURED (not assumed):
%     - relative volume error of the cubified mesh vs the EXACT union volume
%       (BoxAssembly.volume, coordinate compression -- no sampling), per G
%     - genus of the cubified mesh vs the genus of the solid (fine G=96 grid), per G
%     for two voxelisation rules, 3^3 sub-samples per voxel (odd count, so a
%     voxel is never exactly "half inside" -- a tie that biases 2^3 majority):
%       "majority"  inside fraction >= 1/2
%       "any"       any sub-sample inside (keeps legs thinner than a voxel)
%
%   REPRODUCIBILITY: FurnitureShapeFactory('Seed', 0); SurfaceSampler Seed 1.
%
%   Author: Ocean Library  |  2026-09-23

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
addpath(project_root);
out_dir = fullfile(script_dir, 'Results', 'mesh_rcnn');
if ~isfolder(out_dir)
    mkdir(out_dir);
end

import OceanMath.Geometry.Mesh.*

% === 1. One shape per class/style from the procedural factory ============
fac = OceanData.Generators.Shapes3D.FurnitureShapeFactory('Seed', 0);
wanted = ["chair/frame", "chair/solid", "table/stretchers", "stool/ring"];
shapes = struct('Key', num2cell(wanted), 'Asm', cell(size(wanted)), 'Info', cell(size(wanted)));
found = false(size(wanted));
for attempt = 1:200
    [asm, info] = fac.createRandom();
    k = find(wanted == info.Class + "/" + info.Style, 1);
    if ~isempty(k) && ~found(k)
        shapes(k).Asm = asm;
        shapes(k).Info = info;
        found(k) = true;
    end
    if all(found), break; end
end
assert(all(found), 'Factory did not produce every requested style in 200 draws.');

% Topology of the TRUE solid, measured at a fine grid (not assumed from the style).
refG = 96;
trueGenus = zeros(1, numel(shapes));
refVolume = zeros(1, numel(shapes));
for k = 1:numel(shapes)
    occRef = Voxelizer.fromInsideFunction(@(P) shapes(k).Asm.inside(P), [refG refG refG]);
    refMesh = Cubify('Threshold', 0.5).forward(occRef);
    trueGenus(k) = refMesh.genus();
    refVolume(k) = shapes(k).Asm.volume();              % exact union volume
end

% === 2. Resolution sweep: volume error and genus vs G ====================
Gs = [8 12 16 24 32 48];
rules = struct('Name', {"majority", "any"}, 'Supersample', {3, 3}, 'MinFraction', {0.5, 1/27});
volErr = zeros(numel(rules), numel(Gs), numel(shapes));
genusAt = zeros(numel(rules), numel(Gs), numel(shapes));
for k = 1:numel(shapes)
    for r = 1:numel(rules)
        for g = 1:numel(Gs)
            G = Gs(g);
            occ = Voxelizer.fromInsideFunction(@(P) shapes(k).Asm.inside(P), [G G G], ...
                'Supersample', rules(r).Supersample, 'MinFraction', rules(r).MinFraction);
            m = Cubify('Threshold', 0.5).forward(occ);
            volErr(r, g, k) = (m.volume() - refVolume(k)) / refVolume(k);
            genusAt(r, g, k) = m.genus();
        end
    end
end

fprintf('\n%-18s %-9s | %s\n', 'shape', 'rule', strjoin(compose('G=%-6d', Gs), ' '));
for k = 1:numel(shapes)
    for r = 1:numel(rules)
        fprintf('%-18s %-9s | %s   (volume error %%)\n', shapes(k).Key, rules(r).Name, ...
            strjoin(compose('%+7.1f', 100 * volErr(r, :, k)), ' '));
        fprintf('%-18s %-9s | %s   (genus; true = %d)\n', '', '', ...
            strjoin(compose('%7d', genusAt(r, :, k)), ' '), trueGenus(k));
    end
end

% === 3. Figure 1: the pipeline on one chair + the other classes ==========
viz = OceanViz.Core('Theme', 'Dark');
fig1 = viz.createFigure('Mesh R-CNN geometry: solid -> voxels -> cubify');
fig1.Position = [60 60 1500 820];
tl = tiledlayout(fig1, 2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, 'Mesh R-CNN geometry blocks: exact solid \rightarrow voxels \rightarrow Cubify', ...
    'Color', viz.Colors.Text);
opts = {'FaceAlpha', 1, 'EdgeAlpha', 0.6, 'ViewAngle', [-50 22]};
% Core.plot's styleAxes() calls 'axis tight'; TriMeshStrategy then re-imposes [-1,1]^3.

chair = shapes(1).Asm;
partsMesh = TriMesh.fromGeometry(chair);
viz.plot(nexttile(tl), "TriMesh", struct('Vertices', partsMesh.Vertices, 'Faces', partsMesh.Faces), ...
    'Title', sprintf('Exact solid: %d boxes (true genus %d)', chair.numParts(), trueGenus(1)), opts{:});

[Pgt, Ngt] = chair.sampleSurface(4000, RandStream('mt19937ar', 'Seed', 1));
viz.plot(nexttile(tl), "TriMesh", struct('Points', Pgt, 'PointColor', Ngt(:, 2)), ...
    'Title', 'GT union-surface samples (colour = n_y)', opts{:});

for G = [16 32]
    occ = Voxelizer.fromInsideFunction(@(P) chair.inside(P), [G G G], 'Supersample', 3, 'MinFraction', 1/27);
    m = Cubify('Threshold', 0.5).forward(occ);
    viz.plot(nexttile(tl), "TriMesh", struct('Vertices', m.Vertices, 'Faces', m.Faces, 'ShowEdges', true), ...
        'Title', sprintf('Cubify G=%d: V=%d F=%d genus %d', G, m.numVertices(), m.numFaces(), m.genus()), opts{:});
end

palette = viz.getColorSequence(4);
for k = 2:4
    occ = Voxelizer.fromInsideFunction(@(P) shapes(k).Asm.inside(P), [24 24 24], 'Supersample', 3, 'MinFraction', 1/27);
    m = Cubify('Threshold', 0.5).forward(occ);
    viz.plot(nexttile(tl), "TriMesh", struct('Vertices', m.Vertices, 'Faces', m.Faces, ...
        'FaceColor', palette(k, :), 'ShowEdges', true), ...
        'Title', sprintf('%s, G=24: genus %d (true %d)', shapes(k).Key, m.genus(), trueGenus(k)), opts{:});
end

ico = IcoSphere.create(3, 'Radius', 0.8);
viz.plot(nexttile(tl), "TriMesh", struct('Vertices', ico.Vertices, 'Faces', ico.Faces, ...
    'FaceColor', palette(1, :), 'ShowEdges', true), ...
    'Title', sprintf('Pixel2Mesh template: IcoSphere L3, genus %d', ico.genus()), opts{:});
viz.export(fig1, fullfile(out_dir, 'phase1_geometry_pipeline'));

% === 4. Figure 2: what voxel resolution costs ============================
fig2 = viz.createFigure('Voxel resolution vs fidelity');
fig2.Position = [80 80 1200 460];
tl2 = tiledlayout(fig2, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
lineStyles = ["-o", "--s"];
ax1 = nexttile(tl2); viz.styleAxes(ax1); hold(ax1, 'on');
ax2 = nexttile(tl2); viz.styleAxes(ax2); hold(ax2, 'on');
for k = 1:numel(shapes)
    for r = 1:numel(rules)
        lbl = shapes(k).Key + " (" + rules(r).Name + ")";
        plot(ax1, Gs, 100 * abs(volErr(r, :, k)), lineStyles(r), 'Color', palette(k, :), ...
            'LineWidth', 1.4, 'DisplayName', lbl);
        plot(ax2, Gs, genusAt(r, :, k) - trueGenus(k), lineStyles(r), 'Color', palette(k, :), ...
            'LineWidth', 1.4, 'DisplayName', lbl);
    end
end
set(ax1, 'YScale', 'log');
xlabel(ax1, 'voxel resolution G'); ylabel(ax1, '|volume error| (%)');
title(ax1, 'Cubified volume vs exact union volume', 'Color', viz.Colors.Text);
xlabel(ax2, 'voxel resolution G'); ylabel(ax2, 'genus(cubify) - genus(true)');
title(ax2, 'Topology error (0 = holes preserved)', 'Color', viz.Colors.Text);
legend(ax2, 'Location', 'eastoutside', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
viz.export(fig2, fullfile(out_dir, 'phase1_resolution_sweep'));
