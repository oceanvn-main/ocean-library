% EXP_KLR_MIXTURE  Kernel Logistic Regression on ESL mixture data
%
%   DATASET:        ESL Mixture Data (10 centroids per class, N=200 train)
%   MODELS:         1. KLR (Polynomial Degree 4, C=1)
%                   2. KLR (RBF, Gamma=0.5 -> KernelScale=1, C=1)
%   Ref: ESL Chapter 12.3.3, Figure 12.5
%
%   This implements log-likelihood loss (BinaryBCE) with the identical kernels
%   and penalties as the SVM experiment from Figure 12.3.

clc; clear; close all; 
rng(42, 'twister');

ROOT = 'c:\Ocean Library\Matlab code practice';
addpath(ROOT);

fprintf('=== EXP_KLR_MIXTURE: ESL Figure 12.5 Reproduction ===\n\n');

% === 1. Data Generation (ESL Mixture) ===
fprintf('Generating ESL Mixture Data (N=200 Train, N=10000 Test)...\n');

function [X, Y, m1, m2] = generateESLMixture(N, m1, m2)
    if nargin < 2
        m1 = randn(10, 2) + [1, 0];  % Class +1
        m2 = randn(10, 2) + [0, 1];  % Class -1
    end
    X = zeros(N, 2); Y = zeros(N, 1);
    ids1 = randi(10, N/2, 1);
    X(1:N/2, :) = m1(ids1, :) + randn(N/2, 2) * sqrt(1/5);
    Y(1:N/2) = 1;
    ids2 = randi(10, N/2, 1);
    X(N/2+1:N, :) = m2(ids2, :) + randn(N/2, 2) * sqrt(1/5);
    Y(N/2+1:N) = -1;
end

[X_tr, Y_tr, m1, m2] = generateESLMixture(200);
[X_te, Y_te]         = generateESLMixture(10000, m1, m2);

% === 2. Data Preprocessing (Scaling) ===
% Polynomial Kernels require features scaled to [-1, 1] or mean 0 / std 1.
muX = mean(X_tr);
sigmaX = std(X_tr);
X_tr_scaled = (X_tr - muX) ./ sigmaX;
X_te_scaled = (X_te - muX) ./ sigmaX;

% === 3. Model Training ===
solver = OceanStats.Supervised.KernelMethods.BuiltInKernelSolver();
loss   = OceanStats.Loss.Classification.BinaryBCELoss();

fprintf('Training Kernel Logistic Regression (Polynomial Degree 4)...\n');
kernelPoly = OceanStats.Supervised.SVM.Kernels.PolynomialKernel('Degree', 4, 'Gamma', 0.5, 'Coef0', 1.0);
klrPoly = OceanStats.Supervised.KernelMethods.KernelMachine(...
    'C', 1, 'Kernel', kernelPoly, 'Loss', loss, 'Solver', solver);
modelPoly = klrPoly.train(X_tr_scaled, Y_tr);

fprintf('Training Kernel Logistic Regression (Radial Basis Gamma=0.5)...\n');
kernelRBF = OceanStats.Supervised.SVM.Kernels.RBFKernel('Gamma', 0.5);
klrRBF = OceanStats.Supervised.KernelMethods.KernelMachine(...
    'C', 1, 'Kernel', kernelRBF, 'Loss', loss, 'Solver', solver);
modelRBF = klrRBF.train(X_tr_scaled, Y_tr);

% === 4. Evaluation ===
errTrPoly = klrPoly.evaluate(Y_tr, klrPoly.predict(modelPoly, X_tr_scaled));
errTePoly = klrPoly.evaluate(Y_te, klrPoly.predict(modelPoly, X_te_scaled));

errTrRBF = klrRBF.evaluate(Y_tr, klrRBF.predict(modelRBF, X_tr_scaled));
errTeRBF = klrRBF.evaluate(Y_te, klrRBF.predict(modelRBF, X_te_scaled));

function p = evalMix(X, m)
    p = zeros(size(X,1), 1);
    for k = 1:10
        d2 = sum((X - m(k,:)).^2, 2);
        p = p + exp(-d2 * 2.5);
    end
    p = p / 10;
end

p1 = evalMix(X_te, m1); p2 = evalMix(X_te, m2);
bayesPred = ones(size(Y_te)); bayesPred(p2 > p1) = -1;
bayesErr = mean(Y_te ~= bayesPred);

% === 5. Visualization ===
fprintf('Rendering OceanViz Figure...\n');
viz = OceanViz.Core('Theme', 'Scientific');
fig = figure('Name', 'Kernel Logistic Regression (ESL Fig 12.5)', 'Units','normalized','Position',[0.1 0.1 0.6 0.8]);
fig.Color = viz.Colors.Background;

[xx, yy] = meshgrid(linspace(min(X_tr(:,1))-1, max(X_tr(:,1))+1, 150), ...
                    linspace(min(X_tr(:,2))-1, max(X_tr(:,2))+1, 150));
Xgrid = [xx(:), yy(:)];
Xgrid_scaled = (Xgrid - muX) ./ sigmaX;

% Contour levels for probabilities 0.25 (f = -log(3)), 0.5 (f = 0), and 0.75 (f = log(3))
p_75_level = log(3);
p_25_level = -log(3);

% --- Subplot 1: Poly 4 ---
% Re-evaluate f(x) via KernelMachine (which handles the coeff weighting automatically)
K_poly = modelPoly.kernel.compute(modelPoly.sv_X, Xgrid_scaled);
fPoly = ((modelPoly.sv_alpha .* modelPoly.sv_y)' * K_poly)' + modelPoly.b;
zzPoly = reshape(fPoly, size(xx));

ax1 = subplot(2, 1, 1); viz.styleAxes(ax1); hold(ax1, 'on');
scatter(ax1, X_tr(Y_tr==1, 1), X_tr(Y_tr==1, 2), 30, [0.9 0.6 0.1], 'o', 'LineWidth', 1.5);
scatter(ax1, X_tr(Y_tr==-1, 1), X_tr(Y_tr==-1, 2), 30, [0.4 0.7 0.9], 'o', 'LineWidth', 1.5);

contour(ax1, xx, yy, zzPoly, [0 0], 'Color', 'k', 'LineWidth', 1.5);
contour(ax1, xx, yy, zzPoly, [p_75_level p_75_level], '--', 'Color', 'k', 'LineWidth', 1.0);
contour(ax1, xx, yy, zzPoly, [p_25_level p_25_level], '--', 'Color', 'k', 'LineWidth', 1.0);

p1g = evalMix(Xgrid, m1); p2g = evalMix(Xgrid, m2);
contour(ax1, xx, yy, reshape(p1g - p2g, size(xx)), [0 0], '--', 'Color', [0.6 0.2 0.8], 'LineWidth', 1.5);

title(ax1, 'LR - Degree-4 Polynomial in Feature Space', 'Color', viz.Colors.Text);
xlabel(ax1, 'Feature 1 (X_1)', 'Color', viz.Colors.Text);
ylabel(ax1, 'Feature 2 (X_2)', 'Color', viz.Colors.Text);
legend(ax1, {'Class +1 (Orange)', 'Class -1 (Blue)', '0.5 Probability Boundary', '0.75 / 0.25 Prob', '', 'Bayes Optimal'}, 'Location', 'best', 'TextColor', viz.Colors.Text, 'Color', viz.Colors.Background);
text(ax1, 0.05, 0.15, sprintf('Training Error: %.3f\nTest Error:     %.3f\nBayes Error:    %.3f', errTrPoly, errTePoly, bayesErr), 'Units', 'normalized', 'Color', viz.Colors.Text, 'FontName', 'Courier');

% --- Subplot 2: RBF ---
K_rbf = modelRBF.kernel.compute(modelRBF.sv_X, Xgrid_scaled);
fRBF = ((modelRBF.sv_alpha .* modelRBF.sv_y)' * K_rbf)' + modelRBF.b;
zzRBF = reshape(fRBF, size(xx));

ax2 = subplot(2, 1, 2); viz.styleAxes(ax2); hold(ax2, 'on');
scatter(ax2, X_tr(Y_tr==1, 1), X_tr(Y_tr==1, 2), 30, [0.9 0.6 0.1], 'o', 'LineWidth', 1.5);
scatter(ax2, X_tr(Y_tr==-1, 1), X_tr(Y_tr==-1, 2), 30, [0.4 0.7 0.9], 'o', 'LineWidth', 1.5);

contour(ax2, xx, yy, zzRBF, [0 0], 'Color', 'k', 'LineWidth', 1.5);
contour(ax2, xx, yy, zzRBF, [p_75_level p_75_level], '--', 'Color', 'k', 'LineWidth', 1.0);
contour(ax2, xx, yy, zzRBF, [p_25_level p_25_level], '--', 'Color', 'k', 'LineWidth', 1.0);
contour(ax2, xx, yy, reshape(p1g - p2g, size(xx)), [0 0], '--', 'Color', [0.6 0.2 0.8], 'LineWidth', 1.5);

title(ax2, 'LR - Radial Kernel in Feature Space', 'Color', viz.Colors.Text);
xlabel(ax2, 'Feature 1 (X_1)', 'Color', viz.Colors.Text);
ylabel(ax2, 'Feature 2 (X_2)', 'Color', viz.Colors.Text);
text(ax2, 0.05, 0.15, sprintf('Training Error: %.3f\nTest Error:     %.3f\nBayes Error:    %.3f', errTrRBF, errTeRBF, bayesErr), 'Units', 'normalized', 'Color', viz.Colors.Text, 'FontName', 'Courier');

if ~exist('Results', 'dir'); mkdir('Results'); end
saveas(fig, 'Results/exp_klr_mixture_fig12_5.png');

fprintf('\n=== EXPERIMENT SUMMARY ===\n');
fprintf('  Dataset       : ESL Mixture Data\n');
fprintf('  Bayes Error   : %.3f\n\n', bayesErr);
fprintf('  Model (Poly4) : KLR Newton\n');
fprintf('  Train Err     : %.3f\n', errTrPoly);
fprintf('  Test  Err     : %.3f\n\n', errTePoly);
fprintf('  Model (RBF)   : KLR Newton\n');
fprintf('  Train Err     : %.3f\n', errTrRBF);
fprintf('  Test  Err     : %.3f\n', errTeRBF);
fprintf('==========================\n');
