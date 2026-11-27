% exp_mars_mixed_inputs.m
% Demonstrates MARS handling categorical and continuous variables simultaneously.
% As described in ESL 9.4.3 "Mixed Inputs".

rng(42);
N = 300;

% X1 is Categorical with 2 levels: 1 (Group A), 2 (Group B)
X1 = randi([1, 2], N, 1);
% X2 is Continuous in [0, 1]
X2 = rand(N, 1);

X = [X1, X2];

% Generate Target based on category (Interaction!)
% If X1=1: Linear ramp UP
% If X1=2: Linear ramp DOWN
Y_true = zeros(N, 1);
Y_true(X1 == 1) = 5 * X2(X1 == 1);
Y_true(X1 == 2) = 5 * (1 - X2(X1 == 2));

% Add Noise
Y = Y_true + 0.2 * randn(N, 1);

fprintf('Training MARS with Mixed Inputs (Categorical X1, Continuous X2)...\n');
import OceanStats.Supervised.MARS.*

% Create MARS instance. Enable interaction degree 2 to allow X1 * X2 interaction.
% We set PenaltyCost low to ensure the interactions are kept.
mars = MARS('MaxInteractionDegree', 2, 'MaxTerms', 21, 'PenaltyCost', 0.5, 'CategoricalFeatures', [1]);
mars.fit(X, Y);

fprintf('\n--- Mixed Inputs MARS Model ---\n');
mars.printModel({'Category', 'Continuous'});

%% Plotting with OceanViz
viz = OceanViz.Core('Theme', 'Dark');
fig = viz.createFigure('MARS Mixed Inputs (ESL 9.4.3)');
ax = axes(fig);
viz.styleAxes(ax);
hold(ax, 'on');

palette = viz.getColorSequence(2);

for c = 1:2
    idx = (X1 == c);
    % Plot Data
    scatter(ax, X2(idx), Y(idx), 30, palette(c,:), 'filled', 'DisplayName', sprintf('Data (Cat %d)', c));
    
    % Generate Prediction Curve
    % x2_test = linspace(0, 1, 100)';
    % X_test = [repmat(c, 100, 1), x2_test];
    % y_pred = mars.predict(X_test);
    % 
    % plot(ax, x2_test, y_pred, 'Color', palette(c,:), 'LineWidth', 2, 'DisplayName', sprintf('MARS (Cat %d)', c));
end

title(ax, 'MARS: Categorical Partitions and Tensor Products', 'Color', viz.Colors.Text);
xlabel(ax, 'Continuous Predictor (X2)', 'Color', viz.Colors.Text);
ylabel(ax, 'Response (Y)', 'Color', viz.Colors.Text);
legend(ax, 'TextColor', viz.Colors.Text);

saveas(fig, 'mars_mixed_inputs_esl.png');
fprintf('\nExperiment complete. Plot saved to mars_mixed_inputs_esl.png\n');
