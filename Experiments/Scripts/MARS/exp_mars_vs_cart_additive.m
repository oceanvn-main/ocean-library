% exp_mars_vs_cart_additive.m
% Experiment to demonstrate ESL 9.5: Relationship of MARS to CART.

import OceanStats.Supervised.MARS.*
import OceanStats.Supervised.Tree.*

rng(100); % Reproducibility

%% 1. Generate Additive Data
% True Model: Y = X1 + X2 + Noise
N = 300;
X = rand(N, 2); % Uniform [0,1]
epsilon = 0.1 * randn(N, 1);
Y_true = X(:, 1) + X(:, 2);
Y = Y_true + epsilon; 

%% 2. Train Models

% 2.1 Train Standard MARS
% We limit degree to 1 (purely additive model) for a fair comparison.
fprintf('Training Standard MARS...\n');
mars_std = MARS('MaxInteractionDegree', 1, 'MaxTerms', 20, 'PenaltyCost', 2.0);
mars_std.fit(X, Y);

% 2.2 Train CART (Decision Tree)
% Standard regression tree
fprintf('Training CART...\n');
cart = DecisionTree('Criterion', 'regression', 'MaxDepth', 6, 'MinLeafSize', 5);
cart.fit(X, Y);

%% 3. Generate 2D Grid for Partition Visualization
[x1_grid, x2_grid] = meshgrid(linspace(0, 1, 150), linspace(0, 1, 150));
X_grid = [x1_grid(:), x2_grid(:)];

Y_grid_mars = mars_std.predict(X_grid);
Y_grid_cart = cart.predict(X_grid);

Z_mars = reshape(Y_grid_mars, size(x1_grid));
Z_cart = reshape(Y_grid_cart, size(x1_grid));

%% 4. Plot Comparison: 2D Partitioning Contours
fig = figure('Name', 'Space Partitioning: MARS vs CART', 'Position', [100, 100, 1200, 500]);

% Initialize OceanViz context to access Theme Colormaps
viz = OceanViz.Core('Theme', 'Dark'); % Dark theme provides good contrast for partitions

% CART Partitioning
subplot(1, 2, 1);
contourf(x1_grid, x2_grid, Z_cart, 15, 'LineColor', 'k');
colormap(viz.Colors.Gradient);
cb1 = colorbar;
cb1.Color = viz.Colors.Text;
cb1.Label.String = 'Response (Y)';
hold on;
scatter(X(:,1), X(:,2), 20, Y, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
title('CART Partitioning');
subtitle('Blocky, discontinuous step-function regions');
xlabel('X_1'); ylabel('X_2');
axis square;
grid on;

% MARS Partitioning
subplot(1, 2, 2);
contourf(x1_grid, x2_grid, Z_mars, 15, 'LineColor', 'k', 'LineWidth', 0.5);
colormap(viz.Colors.Gradient);
cb2 = colorbar;
cb2.Color = viz.Colors.Text;
cb2.Label.String = 'Response (Y)';
hold on;
% Draw the explicit Hinge lines (Knot locations) that MARS found
basisList = mars_std.BasisFunctions;
for i = 1:numel(basisList)
    bf = basisList{i};
    % If it is a basic hinge function (not the intercept)
    if isa(bf, 'OceanStats.Supervised.MARS.Basis.HingeBasis')
        feat = bf.FeatureIndex;
        knot = bf.Knot;
        % Draw a red dashed line where the hinge occurs
        if feat == 1
            xline(knot, 'r--', 'LineWidth', 1.5, 'Alpha', 0.6);
        elseif feat == 2
            yline(knot, 'r--', 'LineWidth', 1.5, 'Alpha', 0.6);
        end
    end
end
scatter(X(:,1), X(:,2), 20, Y, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
title('MARS Partitioning');
subtitle('Continuous additive hinges (Red dashed = knot locations)');
xlabel('X_1'); ylabel('X_2');
axis square;

sgtitle('ESL 9.5: Visualizing Feature Space Partitions', 'FontSize', 16, 'FontWeight', 'bold');

% Save Plot
saveas(fig, 'mars_vs_cart_partitions.png');
fprintf('Experiment complete. Plot saved to mars_vs_cart_partitions.png\n');

