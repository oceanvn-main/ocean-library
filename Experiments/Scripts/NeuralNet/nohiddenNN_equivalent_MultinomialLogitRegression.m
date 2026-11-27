% Comprehensive Experiment: Multinomial Logistic Regression vs 0-Hidden-Layer NN
clear; clc; close all;

% 1. Generate Synthetic Objective Data (3 classes in 2D space)
rng(42); 
N = 150;
X1 = randn(N, 2) + [2, 2]; % 150x2
X2 = randn(N, 2) + [-2, 2];
X3 = randn(N, 2) + [0, -2];
X = [X1; X2; X3]; % 450 x 2
Y = [ones(N, 1); 2*ones(N, 1); 3*ones(N, 1)]; % 450 x 1

% Train logistic regression
B = mnrfit(X, Y); % CORRECTED: changed fitmnr to mnrfit

% 0-Hidden-Layer NN
net = patternnet([]); % Softmax, Cross-entropy
net.trainParam.showWindow = false;
T = dummyvar(Y)'; % 3 x 450
net = train(net, X', T);

% Create grid
x1Range = linspace(min(X(:,1))-1, max(X(:,1))+1, 100);
x2Range = linspace(min(X(:,2))-1, max(X(:,2))+1, 100);
[x1Grid, x2Grid] = meshgrid(x1Range, x2Range);
XGrid = [x1Grid(:), x2Grid(:)];

% Evaluate probs on the grid & training data
probLogit = mnrval(B, XGrid); % CORRECTED: changed B.predict to mnrval
probNN = net(XGrid')';

% Training Data prob
trainProbLogit = mnrval(B, X); % CORRECTED: changed B.predict to mnrval
trainProbNN = net(X')';

probLogit_Class1 = reshape(probLogit(:, 1), size(x1Grid)); % 100 x 100
probNN_Class1 = reshape(probNN(:, 1), size(x1Grid));

% Explicit decision boundaries
[~, classLogit] = max(probLogit, [], 2); % Across row
classLogitGrid = reshape(classLogit, size(x1Grid)); % CORRECTED: Typo fixed (Logic -> Logit)

[~, classNN] = max(probNN, [], 2);
classNNGrid = reshape(classNN, size(x1Grid));

[~, trainClassLogit] = max(trainProbLogit, [], 2);
[~, trainClassNN] = max(trainProbNN, [], 2);

%% visual

disp('--------------------------------------------------');
disp('Empirical Equivalence Statistics:');
disp('--------------------------------------------------');

accLogit = sum(trainClassLogit == Y) / length(Y) * 100;
accNN = sum(trainClassNN == Y) / length(Y) * 100;
fprintf('Training Accuracy (Logit): %0.2f%%\n', accLogit);
fprintf('Training Accuracy (NN):    %0.2f%%\n', accNN);

%Grid boundary
gridAgreement = sum(classLogitGrid(:) == classNNGrid(:)) / numel(classLogitGrid) * 100;
fprintf('Boundary Agreement on Grid: %0.2f%%\n', gridAgreement);

%Cons probs diff
mse = mean((probLogit(:) - probNN(:)).^2);
maxDiff = max(abs(probLogit(:) - probNN(:)));
fprintf('Probability Mean Squared Error (MSE): %e\n', mse);
fprintf('Maximum Absolute Probability Diff:    %e\n', maxDiff);
disp('--------------------------------------------------');

%% Plotting

fig = figure('Name', 'Comprehensive Model Equivalence Verification', 'Position', [100, 50, 1000, 1000]);
set(fig, 'Color', [0.15 0.15 0.15]); 

% --- ROW 1: Probability Contours (Class 1) ---
ax1 = subplot(3, 2, 1);
contourf(x1Grid, x2Grid, probLogit_Class1, 20, 'LineColor', 'none');
hold on;
scatter(X(Y==1, 1), X(Y==1, 2), 15, 'w', 'filled', 'MarkerEdgeColor', 'k');
title('Logit: Probability Contour', 'Color', 'w');
configureDarkModeAxes(ax1);

ax2 = subplot(3, 2, 2);
contourf(x1Grid, x2Grid, probNN_Class1, 20, 'LineColor', 'none');
hold on;
scatter(X(Y==1, 1), X(Y==1, 2), 15, 'w', 'filled', 'MarkerEdgeColor', 'k');
title('NN: Probability Contour', 'Color', 'w');
configureDarkModeAxes(ax2);

% --- ROW 2: 3D Probability Surfaces (Class 1) ---
ax3 = subplot(3, 2, 3);
surf(x1Grid, x2Grid, probLogit_Class1, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
title('Logit: 3D Surface', 'Color', 'w');
zlabel('Probability', 'Color', 'w');
configureDarkModeAxes(ax3);
view(-30, 45);

ax4 = subplot(3, 2, 4);
surf(x1Grid, x2Grid, probNN_Class1, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
title('NN: 3D Surface', 'Color', 'w');
zlabel('Probability', 'Color', 'w');
configureDarkModeAxes(ax4);
view(-30, 45);

% --- ROW 3: Discrete Decision Boundaries (All Classes) ---
ax5 = subplot(3, 2, 5);
contourf(x1Grid, x2Grid, classLogitGrid, [1 2 3], 'LineColor', 'k');
hold on;
gscatter(X(:,1), X(:,2), Y, 'ryg', '.', 10, 'doleg', 'off');
title('Logit: Decision Boundary', 'Color', 'w');
configureDarkModeAxes(ax5);

ax6 = subplot(3, 2, 6);
contourf(x1Grid, x2Grid, classNNGrid, [1 2 3], 'LineColor', 'k');
hold on;
gscatter(X(:,1), X(:,2), Y, 'ryg', '.', 10, 'doleg', 'off');
title('NN: Decision Boundary', 'Color', 'w');
configureDarkModeAxes(ax6);

colormap(parula);

% Dark Mode Configuration Helper
function configureDarkModeAxes(ax)
    xlabel(ax, 'Feature 1', 'Color', 'w');
    ylabel(ax, 'Feature 2', 'Color', 'w');
    set(ax, 'Color', [0.1 0.1 0.1], ...
            'XColor', [0.8 0.8 0.8], ...
            'YColor', [0.8 0.8 0.8], ...
            'ZColor', [0.8 0.8 0.8], ...
            'GridColor', [0.3 0.3 0.3], ...
            'GridAlpha', 0.5);
    grid on;
end