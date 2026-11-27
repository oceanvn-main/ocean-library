%% HME Linear Diversity: Piecewise & Heteroscedastic Challenges
% Objective: Test HME partition properties when all experts are Linear.
clear; clc; close all;
addpath(genpath(pwd));

import OceanStats.Ensemble.HME.*
import OceanStats.Supervised.Regression.*

% --- Challenge 1: The "Z" Piecewise Linear ---
fprintf('Challenge 1: The "Z" Piecewise Linear...\n');
N = 900;
x_z = linspace(-1, 1, N)';
Y_z = zeros(N, 1);

% Region 1 (-1 to -0.3): Rising
Y_z(x_z < -0.3) = 1.5 * x_z(x_z < -0.3) + 0.5;
% Region 2 (-0.3 to 0.3): Falling
mask2 = (x_z >= -0.3) & (x_z < 0.3);
Y_z(mask2) = -2 * x_z(mask2);
% Region 3 (0.3 to 1): Rising
Y_z(x_z >= 0.3) = 1.5 * x_z(x_z >= 0.3) - 0.5;

Y_z = Y_z + 0.05 * randn(N, 1);
X_z = [ones(N, 1), x_z];

% --- Challenge 2: The Heteroscedastic Jump ---
fprintf('Challenge 2: The Heteroscedastic Jump...\n');
Y_j = zeros(N, 1);
% Left: Low noise, flat
Y_j(x_z < 0) = 2;
Y_j(x_z < 0) = Y_j(x_z < 0) + 0.01 * randn(sum(x_z < 0), 1);
% Right: High noise, jump
Y_j(x_z >= 0) = 5;
Y_j(x_z >= 0) = Y_j(x_z >= 0) + 0.5 * randn(sum(x_z >= 0), 1);
X_j = [ones(N, 1), x_z];

% --- Challenge 3: The Sine Wave (Piecewise Linear Approx) ---
fprintf('Challenge 3: The Sine Wave...\n');
Y_s = sin(2 * pi * x_z) + 0.05 * randn(N, 1);
X_s = [ones(N, 1), x_z];

% --- Setup HME (Linear Only) ---
def_pool = @() CompetitiveExpert({GenericExpert(LinearModel(), 'Density')});

% Helper: Build a balanced tree of a given depth
function node = build_balanced_tree(depth, leaf_gen)
    if depth == 1
        node = LeafNode(leaf_gen());
    else
        node = InternalNode({build_balanced_tree(depth-1, leaf_gen), ...
                             build_balanced_tree(depth-1, leaf_gen)});
    end
end

% Visualization Helper
viz = OceanViz.Core('Theme', 'Dark');

% Standard Model (Depth 3 -> 4 Leaves) for Z and Jump
model_std = HierarchicalMixture(build_balanced_tree(3, def_pool));

% High Resolution Model (Depth 5 -> 16 Leaves) for Sine
model_high = HierarchicalMixture(build_balanced_tree(5, def_pool));

% --- Trial 1: Z-Curve ---
fprintf('Training on Z-Curve...\n');
model_std.fit([X_z, Y_z]);
plot_result(viz, x_z, Y_z, model_std, 'Piecewise Z-Linear', 'hme_z_linear.png');

% --- Trial 2: Jump ---
model_std.reset();
fprintf('Training on Jump Discontinuity...\n');
model_std.fit([X_j, Y_j]);
plot_result(viz, x_z, Y_j, model_std, 'Heteroscedastic Jump', 'hme_jump_linear.png');

% --- Trial 3: Sine Wave (High Res) ---
fprintf('Training High-Resolution Sine (16 Linear Segments)...\n');
model_high.fit([X_s, Y_s]);
plot_result(viz, x_z, Y_s, model_high, 'Sine High-Res (16 Segments)', 'hme_sine_highres.png');

% (Old trials removed)

function plot_result(viz, x, y, model, titleStr, filename)
    fig = viz.createFigure(['Linear Challenge: ' titleStr]);
    ax = gca; viz.styleAxes(ax); hold on;
    
    % 1. Plot Data & Prediction
    s_data = scatter(ax, x, y, 15, 'w', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Noisy Observations');
    
    x_grid = linspace(min(x), max(x), 500)';
    X_grid = [ones(500,1), x_grid];
    y_hat = model.predict(X_grid);
    p_pred = plot(ax, x_grid, y_hat, 'Color', viz.Colors.Secondary, 'LineWidth', 3, 'DisplayName', 'HME Piecewise Consensus');
    
    % 2. Discover and plot Level-1 Gating Boundaries
    % A Depth-3 tree has multiple levels. Let's find where the Root gate switches.
    params = model.getParams();
    % Simple boundary for 1D: x * gamma1 = x * gamma2 -> x = (g2-g1) / ...
    % Better: just numerical root finding on the grid
    root_gate = @(x) model.getRoot().getGate().predict([1, x]);
    g_vals = zeros(500, 2);
    for i = 1:500, g_vals(i,:) = root_gate(x_grid(i)); end
    [~, root_choice] = max(g_vals, [], 2);
    
    switches = find(diff(root_choice) ~= 0);
    h_bound = [];
    for s_idx = 1:length(switches)
        bx = x_grid(switches(s_idx));
        h_bound = line(ax, [bx bx], ylim(ax), 'Color', [1 1 1], 'LineStyle', ':', 'LineWidth', 1, 'DisplayName', 'Major Gate Boundary');
    end
    
    % 3. Labels and Polish
    title(ax, ['HME Linear Consensus: ' titleStr]);
    xlabel(ax, 'Input Feature (X)'); 
    ylabel(ax, 'Target Response (Y)');
    
    leg_entries = [s_data, p_pred];
    if ~isempty(h_bound), leg_entries = [leg_entries, h_bound]; end
    legend(ax, leg_entries, 'Location', 'northeastoutside', 'TextColor', viz.Colors.Text);
    
    % Annotate the "Vertical Jumps" if they occur
    if contains(titleStr, 'Sine') || contains(titleStr, 'Jump')
        text(ax, 0.4, -0.8, '\leftarrow Gating Transition (Switch)', 'Color', viz.Colors.Text, 'FontSize', 9);
    end
    
    saveas(fig, filename);
    fprintf('Audit saved to %s\n', filename);
end
