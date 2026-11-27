%% HME Geometry Challenge: Spirals & Balls
clear; clc; close all;
addpath(genpath(pwd));

% --- Challenge 1: The Concentric Ball ---
fprintf('Challenge 1: The Concentric Ball...\n');
N = 1000;
theta = 2 * pi * rand(N, 1);
r_in = 0.4 * sqrt(rand(N/2, 1));
r_out = 0.6 + 0.4 * sqrt(rand(N/2, 1));
X_ball = [ [r_in .* cos(theta(1:N/2)), r_in .* sin(theta(1:N/2))]; ...
           [r_out .* cos(theta(N/2+1:end)), r_out .* sin(theta(N/2+1:end))] ];
Y_ball = [zeros(N/2, 1); ones(N/2, 1)];
X_ball_intercept = [ones(N, 1), X_ball];

% --- Challenge 2: The Twin Spiral ---
fprintf('Challenge 2: The Twin Spiral...\n');
theta_s = linspace(0, 4*pi, N/2)';
r_s = theta_s / (4*pi);
x1_s = r_s .* cos(theta_s);
y1_s = r_s .* sin(theta_s);
x2_s = -x1_s;
y2_s = -y1_s;

X_spiral = [ [x1_s, y1_s]; [x2_s, y2_s] ] + 0.05 * randn(N, 2);
Y_spiral = [zeros(N/2, 1); ones(N/2, 1)];
X_spiral_intercept = [ones(N, 1), X_spiral];

% --- Training & Visualization Function ---
run_and_plot('Concentric Ball', X_ball_intercept, Y_ball, 'hme_ball_audit.png');
run_and_plot('Twin Spiral', X_spiral_intercept, Y_spiral, 'hme_spiral_audit.png');

function run_and_plot(name, X, Y, filename)
    % Setup HME with Structural Diversity (Linear vs Polynomial)
    import OceanStats.Ensemble.HME.*
    import OceanStats.Supervised.Regression.*

    % SOLUTION A: UseLocalGating = true
    % SOLUTION C: Alpha = 0.5 (Moderate sparsity)
    def_pool = @() CompetitiveExpert({...
        GenericExpert(LogisticModel(), 'Mass'), ...
        GenericExpert(PolynomialLogisticModel(2), 'Mass') ...
    }, 'UseLocalGating', true, 'Alpha', 0.5);

    % Binary Tree Depth 2 -> 2 Leaves -> 4 Total Potential Specialists
    e1 = LeafNode(def_pool());
    e2 = LeafNode(def_pool());
    root = InternalNode({e1, e2});
    model = HierarchicalMixture(root);

    fprintf('Training HME for %s...\n', name);
    model.fit([X, Y]);

    % Visualize
    viz = OceanViz.Core('Theme', 'Dark');
    fig = viz.createFigure(['HME Topology & Hiring: ' name]);
    fig.Position = [100 100 1200 500]; % Manual size for tiling
    t = tiledlayout(fig, 1, 2, 'TileSpacing', 'Compact');
    
    [gx, gy] = meshgrid(linspace(-1.2, 1.2, 100));
    X_grid = [ones(numel(gx), 1), gx(:), gy(:)];
    
    % --- Tile 1: Global Decision Boundary ---
    ax1 = nexttile(t); viz.styleAxes(ax1); hold(ax1, 'on');
    Y_prob = model.predict(X_grid);
    Z = reshape(Y_prob, size(gx));
    contourf(ax1, gx, gy, Z, 20, 'LineStyle', 'none'); 
    colormap(ax1, 'turbo'); 
    
    s0 = scatter(ax1, X(Y==0, 2), X(Y==0, 3), 15, 'w', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Class 0');
    s1 = scatter(ax1, X(Y==1, 2), X(Y==1, 3), 15, 'y', 'filled', 'MarkerEdgeColor', 'k', 'DisplayName', 'Class 1');
    
    xlabel(ax1, 'X_1'); ylabel(ax1, 'X_2');
    title(ax1, 'Class Probability P(Y=1|X)');
    cb1 = colorbar(ax1); ylabel(cb1, 'Probability');
    
    % --- Tile 2: Global Specialist Map ---
    % Reveals who is the "Active Worker" at each coordinate across ALL leaves.
    ax2 = nexttile(t); viz.styleAxes(ax2); hold(ax2, 'on');
    
    % 1. Get Leaf-level gating probabilities [N_grid x 2]
    % Root is InternalNode, so it has a Gate
    params = model.getParams();
    gate_probs = root.getGate().predict(X_grid); 
    
    % 2. Get Pool-level hiring weights for each leaf
    w1 = e1.getExpert().getWorkerWeights(X_grid); % [N_grid x 2]
    w2 = e2.getExpert().getWorkerWeights(X_grid); % [N_grid x 2]
    
    % 3. Calculate Joint Probability of each specialist (P(Leaf) * P(Expert|Leaf))
    joint = [gate_probs(:, 1) .* w1(:, 1), ... % Specialist 1 (Leaf 1, Linear)
             gate_probs(:, 1) .* w1(:, 2), ... % Specialist 2 (Leaf 1, Poly)
             gate_probs(:, 2) .* w2(:, 1), ... % Specialist 3 (Leaf 2, Linear)
             gate_probs(:, 2) .* w2(:, 2)];    % Specialist 4 (Leaf 2, Poly)
             
    [~, best_spec] = max(joint, [], 2);
    Z_s = reshape(best_spec, size(gx));
    
    % Plot categorical regions
    imagesc(ax2, [-1.2 1.2], [-1.2 1.2], Z_s);
    set(ax2, 'YDir', 'normal');
    
    % Diverse Colormap for 4 specialists
    specMap = [0.2 0.6 1.0; ... % Specialist 1: Blue
               1.0 0.5 0.0; ... % Specialist 2: Orange
               0.2 0.8 0.4; ... % Specialist 3: Green
               0.9 0.2 0.2];    % Specialist 4: Red
    colormap(ax2, specMap);
    
    xlabel(ax2, 'X_1'); title(ax2, 'Global Specialist Allocation (All Leaves)');
    cb2 = colorbar(ax2); 
    set(cb2, 'Ticks', 1:4, 'TickLabels', ...
        {'L1: Linear', 'L1: Poly', 'L2: Linear', 'L2: Poly'});

    saveas(fig, filename);
    fprintf('Global Specialists Audit saved to %s\n', filename);
end
