% =========================================================================
% Experiment: Neural Network Graph — Visualization
%
%   Architecture:  12 → 16 → 8 → 10
%   Purpose:       Visual test of Phase 1 NNGraph data structure.
%                  Shows layered node layout with edges colored by weight
%                  sign and thickness proportional to weight magnitude.
%
%   Run:  cd "c:\Ocean Library\Matlab code practice"
%         run Experiments/Scripts/NeuralNet/exp_nngraph_visualize.m
% =========================================================================
clear; clc; close all;

import OceanDL.Graph.*

% =========================================================================
% 1. BUILD THE GRAPH  [12 → 16 → 8 → 10]
% =========================================================================
spec = NNTopologySpec.make( ...
    [12, 16, 8, 10], ...
    {'linear', 'relu', 'relu', 'softmax'}, ...
    'LayerNames', {'Input (12)', 'Hidden 1 (16)', 'Hidden 2 (8)', 'Output (10)'});

graph = NNGraph.fromSpec(spec);

% ---- Forward pass on random input to populate node.Value_ ----
rng(42);
X      = randn(8, 12);         % 8 samples, 12 features
A_out  = graph.forwardPass(X);

fprintf('Graph built: %d total params\n', graph.paramCount());
fprintf('Output shape: %d × %d  (softmax probabilities)\n', size(A_out));
graph.disp();

% =========================================================================
% 2. FIGURE SETUP  — dark theme, wide canvas
% =========================================================================
BG      = [0.10 0.12 0.16];
PANEL   = [0.15 0.17 0.22];
TXTCOL  = [0.93 0.93 0.93];

fig = figure('Name',       'NNGraph — Phase 1 Visualization', ...
             'Color',       BG, ...
             'NumberTitle', 'off', ...
             'Position',    [30, 60, 1500, 820]);

ax = axes('Parent',    fig, ...
          'Color',      BG, ...
          'XColor',    'none', ...
          'YColor',    'none', ...
          'Position',  [0.03 0.08 0.72 0.86]);
hold(ax, 'on');

% Side info panel axes
axInfo = axes('Parent',   fig, ...
              'Color',     PANEL, ...
              'XColor',   'none', ...
              'YColor',   'none', ...
              'Position', [0.77 0.08 0.21 0.86]);
hold(axInfo, 'on');
xlim(axInfo, [0 1]); ylim(axInfo, [0 1]);

% =========================================================================
% 3. COMPUTE NODE POSITIONS  (layered layout)
% =========================================================================
nL         = graph.numLayers();
xPad       = 0.12;
xPositions = linspace(xPad, 1 - xPad, nL);   % layer x-positions

nodePos = cell(1, nL);
for L = 1:nL
    n  = graph.numNodes(L);
    ys = linspace(0.02, 0.98, n);              % evenly spaced vertically
    nodePos{L} = [repmat(xPositions(L), n, 1), ys(:)];
end

% =========================================================================
% 4. LAYER COLOR PALETTE
% =========================================================================
%      Input          Hidden 1        Hidden 2       Output
LAYER_COL = [0.30 0.65 0.95;   % ocean blue
             0.22 0.82 0.50;   % emerald green
             0.99 0.72 0.15;   % amber
             0.95 0.40 0.35];  % coral red

% =========================================================================
% 5. DRAW EDGES  (color = sign, thickness ∝ |weight|)
% =========================================================================
edges = graph.getEdges();
nEdges = numel(edges);

% Find global max |weight| for normalizing thickness
maxW = max(cellfun(@(e) abs(e.Weight_), edges));
maxW = max(maxW, 1e-8);   % guard against all-zero init

for ei = 1:nEdges
    e      = edges{ei};
    L_from = double(e.FromNode.LayerIndex);
    L_to   = double(e.ToNode.LayerIndex);
    ni_f   = double(e.FromNode.NodeIndex);
    ni_t   = double(e.ToNode.NodeIndex);

    p1 = nodePos{L_from}(ni_f, :);
    p2 = nodePos{L_to  }(ni_t, :);

    w  = e.Weight_;
    lw = 0.05 + 2.0 * abs(w) / maxW;   % [0.05 .. 2.05] px
    alpha = 0.06 + 0.18 * abs(w) / maxW;  % [0.06 .. 0.24] opacity

    if w >= 0
        col = [0.30 0.60 0.90 alpha];   % blue = positive weight
    else
        col = [0.95 0.30 0.30 alpha];   % red  = negative weight
    end

    line(ax, [p1(1) p2(1)], [p1(2) p2(2)], ...
        'Color',     col, ...
        'LineWidth', lw);
end

% =========================================================================
% 6. DRAW NODES  (glow halo + filled circle + label)
% =========================================================================
R      = 0.018;    % node radius (normalized axis units)
theta  = linspace(0, 2*pi, 48)';

for L = 1:nL
    baseCol = LAYER_COL(L, :);
    nodes   = graph.getLayer(L).getNodes();
    n       = numel(nodes);

    for ni = 1:n
        node = nodes{ni};
        px   = nodePos{L}(ni, 1);
        py   = nodePos{L}(ni, 2);

        % Brightness driven by mean activation value (0.3 → 1.0 range)
        brightness = 0.35 + 0.65 * min(1.0, abs(node.Value_));

        % --- Outer glow --- (semi-transparent, larger circle)
        xg = px + R * 1.9 * cos(theta);
        yg = py + R * 1.9 * sin(theta);
        fill(ax, xg, yg, baseCol, ...
            'FaceAlpha', 0.10, 'EdgeColor', 'none');

        % --- Main node body ---
        xc = px + R * cos(theta);
        yc = py + R * sin(theta);
        fill(ax, xc, yc, baseCol * brightness, ...
            'EdgeColor', baseCol, ...
            'LineWidth', 1.0, ...
            'FaceAlpha', 0.90);

        % --- Node index label ---  (only if layer is small enough)
        if n <= 16
            text(ax, px, py, sprintf('%d', ni), ...
                'Color',               'w', ...
                'FontSize',            6.5, ...
                'FontWeight',          'bold', ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment',   'middle');
        end
    end
end

% =========================================================================
% 7. LAYER LABELS + CONNECTION COUNT LABELS
% =========================================================================
for L = 1:nL
    col = LAYER_COL(L, :);
    act = spec.Activations{L};

    % Below-axis layer name
    text(ax, xPositions(L), -0.06, ...
        sprintf('%s\n(%s)', spec.LayerNames{L}, act), ...
        'Color',               col, ...
        'FontSize',            9.5, ...
        'FontWeight',          'bold', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment',   'top');

    % Arrow between layers
    if L < nL
        xMid = (xPositions(L) + xPositions(L+1)) / 2;
        nConn = graph.numNodes(L) * graph.numNodes(L+1);
        text(ax, xMid, 1.03, sprintf('%d edges', nConn), ...
            'Color',               [0.55 0.58 0.63], ...
            'FontSize',            8.5, ...
            'HorizontalAlignment', 'center');
    end
end

% Main title
title(ax, 'Neural Network Graph  —  Phase 1 Visualization', ...
    'Color',    TXTCOL, ...
    'FontSize', 14, ...
    'FontWeight', 'bold');

subtitle(ax, sprintf('Architecture: [%s]   |   %d parameters   |   Edge color: ■ positive  ■ negative   |   Brightness ∝ |activation|', ...
    num2str(spec.LayerSizes, '%d → '), graph.paramCount()), ...
    'Color',    [0.55 0.58 0.63], ...
    'FontSize', 9);

xlim(ax, [-0.02, 1.02]);
ylim(ax, [-0.13, 1.08]);

% =========================================================================
% 8. SIDE INFO PANEL
% =========================================================================
function txt(ax_, x, y, s, sz, col, bold)
    fw = 'normal'; if bold, fw = 'bold'; end
    text(ax_, x, y, s, 'Color', col, 'FontSize', sz, ...
        'FontWeight', fw, 'Units', 'normalized', ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
        'Interpreter', 'none');
end

ACCENT  = [0.30 0.65 0.95];
DIM     = [0.50 0.53 0.58];
yy      = 0.97;
dy      = 0.055;

txt(axInfo, 0.06, yy, 'Network Summary', 11, ACCENT, true);  yy = yy - 0.04;
txt(axInfo, 0.06, yy, repmat('─', 1, 28), 8, DIM, false);    yy = yy - dy;

rows = {
    'Input nodes',   num2str(graph.numNodes(1));
    'Hidden 1',      num2str(graph.numNodes(2));
    'Hidden 2',      num2str(graph.numNodes(3));
    'Output nodes',  num2str(graph.numNodes(4));
    'Total edges',   num2str(numel(graph.getEdges()));
    'Total params',  num2str(graph.paramCount());
};
for ri = 1:size(rows, 1)
    txt(axInfo, 0.06, yy, rows{ri,1}, 9,  DIM,    false);
    txt(axInfo, 0.62, yy, rows{ri,2}, 9,  TXTCOL, true);
    yy = yy - dy;
end

yy = yy - 0.01;
txt(axInfo, 0.06, yy, repmat('─', 1, 28), 8, DIM, false); yy = yy - dy;
txt(axInfo, 0.06, yy, 'Weight Init',  11, ACCENT, true);  yy = yy - dy;
txt(axInfo, 0.06, yy, 'ReLU layers',  9, DIM,    false);
txt(axInfo, 0.55, yy, 'He Normal',    9, TXTCOL, true); yy = yy - dy;
txt(axInfo, 0.06, yy, 'Other layers', 9, DIM,    false);
txt(axInfo, 0.55, yy, 'Glorot Unif.', 9, TXTCOL, true); yy = yy - dy * 1.5;

txt(axInfo, 0.06, yy, 'Activations',  11, ACCENT, true);  yy = yy - dy;
for L = 1:nL
    col_ = LAYER_COL(L, :);
    txt(axInfo, 0.06, yy, sprintf('L%d: %s', L, spec.Activations{L}), 9, col_, false);
    yy = yy - dy;
end

yy = yy - 0.01;
txt(axInfo, 0.06, yy, repmat('─', 1, 28), 8, DIM, false); yy = yy - dy;
txt(axInfo, 0.06, yy, 'Forward Pass', 11, ACCENT, true);   yy = yy - dy;
txt(axInfo, 0.06, yy, 'Samples (N)',  9, DIM, false);
txt(axInfo, 0.62, yy, num2str(size(X,1)), 9, TXTCOL, true); yy = yy - dy;
txt(axInfo, 0.06, yy, 'Output range', 9, DIM, false);
txt(axInfo, 0.62, yy, '[0, 1]',       9, TXTCOL, true); yy = yy - dy;
txt(axInfo, 0.06, yy, 'Output sum',   9, DIM, false);
txt(axInfo, 0.62, yy, sprintf('%.4f', mean(sum(A_out, 2))), 9, TXTCOL, true); yy = yy - dy * 1.5;

txt(axInfo, 0.06, yy, 'Legend', 11, ACCENT, true);  yy = yy - dy;
txt(axInfo, 0.06, yy, '■  Positive weight', 9, [0.30 0.60 0.90], false); yy = yy - dy;
txt(axInfo, 0.06, yy, '■  Negative weight', 9, [0.95 0.30 0.30], false); yy = yy - dy;
txt(axInfo, 0.06, yy, '●  Glow = |activation|', 9, DIM, false);

fprintf('\nVisualization complete. %d edges drawn across %d layers.\n', ...
    numel(graph.getEdges()), nL);
