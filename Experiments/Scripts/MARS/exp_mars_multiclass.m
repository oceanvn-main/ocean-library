% exp_mars_multiclass.m
% Demonstrates MARS Section 4.2 Multi-Response Classification

rng(123);

% 1. Create a 3-class dataset using DataFactory
factory = OceanStats.Data.DataFactory('Seed', 123);
[X, Y] = factory.create('Shapes', 'Shape', 'Moons', 'N', 300, 'Noise', 0.2);

% DataFactory generates binary Moons, let's artificially shift a portion to Class 3
% to test the multi-class (K=3) indicator functionality.
Y(X(:,1) > 1.0 & Y == 0) = 2; % Class 2 (0-indexed base)

fprintf('Dataset generated: %d samples, 3 classes (0, 1, 2)\n', size(X,1));
unique_classes = unique(Y);
disp(unique_classes');

%% 2. Train MARS
fprintf('Training MARS via Indicator Mapping (Max Interaction = 2)...\n');
import OceanStats.Supervised.MARS.*

% MARS natively detects <=15 unique integer targets and converts to Indicator
mars = MARS('MaxInteractionDegree', 2, 'MaxTerms', 35, 'PenaltyCost', 3.0);
mars.fit(X, Y);

fprintf('\nMARS Multiclass Training Complete.\n');

%% 3. Evaluate Predictions
Y_pred = mars.predict(X);

% Simple Accuracy
accuracy = sum(Y_pred == Y) / length(Y);
fprintf('Training Accuracy: %.2f%%\n', accuracy * 100);

%% 4. Plot Decision Boundaries
margin = 0.5;
[x1_grid, x2_grid] = meshgrid(linspace(min(X(:,1))-margin, max(X(:,1))+margin, 600), ...
                              linspace(min(X(:,2))-margin, max(X(:,2))+margin, 600));
X_grid = [x1_grid(:), x2_grid(:)];

% Predict on Grid
Y_grid_pred = mars.predict(X_grid);
Z = reshape(Y_grid_pred, size(x1_grid));

% Visualization
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('MARS Multi-Class Boundaries');
ax = axes(fig);
viz.styleAxes(ax);

palette = viz.getColorSequence(3);

% Contour Regions
contourf(ax, x1_grid, x2_grid, Z, [-0.5 0.5 1.5 2.5], 'LineColor', 'none');
colormap(ax, palette .* 0.3); % Dim the background
hold(ax, 'on');

% Draw class boundaries
contour(ax, x1_grid, x2_grid, Z, [0.5 1.5], 'LineColor', viz.Colors.Boundary, 'LineWidth', 2);

% Scatter Training Data
for c = 0:2
    idx = (Y == c);
    scatter(ax, X(idx,1), X(idx,2), 35, palette(c+1,:), 'filled', ...
            'MarkerEdgeColor', viz.Colors.Text, 'DisplayName', sprintf('Class %d', c));
end

title(ax, 'MARS Multi-Class (K=3) Classification', 'Interpreter', 'latex', 'Color', viz.Colors.Text);
subtitle(ax, sprintf('Training Accuracy: %.1f%%', accuracy * 100), 'Color', viz.Colors.Text);
xlabel(ax, 'Feature 1 ($X_1$)', 'Interpreter', 'latex');
ylabel(ax, 'Feature 2 ($X_2$)', 'Interpreter', 'latex');
legend(ax, 'Location', 'best', 'TextColor', viz.Colors.Text);
axis(ax, 'square');

saveas(fig, 'mars_multiclass.png');
fprintf('Experiment complete. Plot saved to mars_multiclass.png\n');
