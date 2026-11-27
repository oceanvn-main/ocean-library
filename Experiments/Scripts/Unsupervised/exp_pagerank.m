% EXP_PAGERANK PageRank computation on a small network (ESL Figure 14.47).
%
%   DATASET:        Synthetic 6-node network from Elements of Statistical Learning (Figure 14.47).
%   MODEL:          PageRank (power iteration method).
%   ASSUMPTIONS:    Node 2 points to Node 3 as visually identified in the figure.
%                   Damping factor d = 0.85 as per textbook suggestion.
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; rng(42, 'twister');

% === 1. Data Loading / Construction ===
N = 6;
L = zeros(N, N);

% L_{ij} = 1 if page j points to page i.
% From Figure 14.47:
% 1 points to 2, 3 -> L(2,1)=1, L(3,1)=1
L(2, 1) = 1; L(3, 1) = 1;
% 2 points to 3 -> L(3,2)=1
L(3, 2) = 1;
% 3 points to 1 -> L(1,3)=1
L(1, 3) = 1;
% 4 points to 1, 3 -> L(1,4)=1, L(3,4)=1
L(1, 4) = 1; L(3, 4) = 1;
% 5 points to 3, 6 -> L(3,5)=1, L(6,5)=1
L(3, 5) = 1; L(6, 5) = 1;
% 6 points to 2 -> L(2,6)=1
L(2, 6) = 1;

fprintf('Constructed Link Matrix L (L_ij=1 if j->i):\n');
disp(L);

% === 2. Model Configuration & Training ===
pr = OceanStats.Unsupervised.PageRank();
d = 0.85;
p = pr.fit(L, 'DampingFactor', d);

% === 3. Evaluation ===
sum_p = sum(p);
assert(abs(sum_p - N) < 1e-9, 'Sum of PageRanks must be exactly N.');

% Display PageRanks
[sorted_p, sorted_idx] = sort(p, 'descend');

fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : ESL Figure 14.47 Network (N=6)\n');
fprintf('  Model         : PageRank, d=%.2f\n', d);
fprintf('  Iterations    : %d\n', pr.NumIterations);
fprintf('  Sum of p_i    : %.4f (Theoretical: %d)\n', sum_p, N);
fprintf('--------------------------\n');
for i = 1:N
    node = sorted_idx(i);
    fprintf('  Rank %d : Page %d (Score: %.4f)\n', i, node, sorted_p(i));
end
fprintf('==========================\n');

% === 4. Visualization ===
% Use OceanViz Theme for consistency
viz = OceanViz.Core('Theme', 'Dark');
fig = figure('Name', 'PageRank - ESL Fig 14.47', 'Color', viz.Colors.Background);
ax = axes('Parent', fig, 'Color', viz.Colors.Background, 'XColor', 'none', 'YColor', 'none');

% We use MATLAB's digraph. Note digraph(A) expects A(i,j)=1 if i->j.
% Our L is the transpose, so A = L'.
A = L';
G = digraph(A);

% Node sizes proportional to PageRank
minSize = 10;
maxSize = 40;
% Normalize p to [minSize, maxSize] roughly
nodeSizes = minSize + (p / max(p)) * (maxSize - minSize);

% Define custom coordinates roughly matching Figure 14.47 for better visual
X_pos = [2, 6, 4, 1, 4.5, 6.5];
Y_pos = [6, 5, 4, 2, 1,   2.5];

% Format labels
nodeLabels = cell(N, 1);
for i = 1:N
    nodeLabels{i} = sprintf('Page %d\np=%.2f', i, p(i));
end

p_plot = plot(ax, G, 'XData', X_pos, 'YData', Y_pos, ...
    'NodeLabel', nodeLabels, ...
    'MarkerSize', nodeSizes, ...
    'NodeColor', viz.Colors.Primary, ...
    'EdgeColor', [0.5 0.5 0.5], ...
    'LineWidth', 1.5, ...
    'ArrowSize', 12, ...
    'NodeFontName', 'Arial', ...
    'NodeFontSize', 9);

% Enhance text labels
p_plot.NodeFontSize = 10;
p_plot.NodeFontWeight = 'bold';
% Make labels dark if node is bright, but primary color is usually visible
title(ax, 'PageRank on ESL Figure 14.47', 'Color', viz.Colors.Text, 'FontSize', 14);

% Optional: force rendering to finish
drawnow;
