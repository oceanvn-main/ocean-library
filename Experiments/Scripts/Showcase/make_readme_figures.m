% MAKE_README_FIGURES  Small, reproducible experiments that produce the README figures.
%
%   Runs four short experiments on the CPU (a few minutes in total) and writes
%   PNG figures to docs/images/. Every number shown in a figure is computed
%   here; nothing is loaded from a results cache.
%
%     1. Robust image denoising   — Tikhonov (L2) vs total variation (L1),
%                                   OceanMath.Regularization + IRLS / Chambolle-Pock
%     2. Prototype classifiers    — K-means vs LVQ vs an MLP trained with the
%                                   OceanDL autograd engine (ESL Figure 13.1 setting)
%     3. MLP learning curves      — train / test cross-entropy and error per step,
%                                   without and with AdamW weight decay
%     4. Jack's car rental        — policy iteration (Sutton & Barto, Example 4.2)
%
%   Requirements: Image Processing Toolbox (for the cameraman test image).
%
%   Usage (from the repository root):
%       run('Experiments/Scripts/Showcase/make_readme_figures.m')
%
%   Author: Ocean Library  |  2026-11-28

libraryRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(libraryRoot);
outDir = fullfile(libraryRoot, 'docs', 'images');
if ~isfolder(outDir), mkdir(outDir); end

viz = OceanViz.Core('Theme', 'Dark');

%% ========================================================================
%  1. Robust image denoising: L2 (Tikhonov) vs L1 (total variation)
%  ========================================================================
%  Energy: E(f) = (1/2)||f - g||^2 + lambda * sum_k rho((D f)_k),
%  D = 2-D forward differences. L2 is solved exactly by IRLS (constant
%  weights), TV by the Chambolle-Pock primal-dual algorithm.
fprintf('\n=== 1. Image denoising ===\n');
rng(0);
cleanImg   = im2double(imread('cameraman.tif'));
noiseSigma = 0.10;
noisyImg   = cleanImg + noiseSigma * randn(size(cleanImg));

psnrOf = @(f) 10 * log10(1 / mean((f(:) - cleanImg(:)).^2));

% lambda chosen by a small PSNR grid search against the clean image
% (a demonstration setting: the clean image is known here).
lambdaL2 = [0.05 0.1 0.25 0.5];
lambdaTV = [0.04 0.06 0.08 0.12];

[bestL2, lamL2] = bestOverGrid(lambdaL2, psnrOf, @(lam) ...
    OceanMath.Image.Variational.Solvers.IRLSSolver('MaxIter', 5, 'Tol', 1e-8).solve( ...
        OceanMath.Image.Variational.ImageRegularizationEnergy( ...
            OceanMath.Regularization.QuadraticPenalty(), 'Lambda', lam), noisyImg));

[bestTV, lamTV] = bestOverGrid(lambdaTV, psnrOf, @(lam) ...
    OceanMath.Image.Variational.Solvers.PrimalDualSolver('MaxIter', 400, 'Tol', 1e-6).solve( ...
        OceanMath.Image.Variational.ImageRegularizationEnergy( ...
            OceanMath.Regularization.L1Penalty(), 'Lambda', lam), noisyImg));

fprintf('  noisy PSNR %.2f dB | L2 %.2f dB (lambda %.2f) | TV %.2f dB (lambda %.2f)\n', ...
    psnrOf(noisyImg), psnrOf(bestL2), lamL2, psnrOf(bestTV), lamTV);

fig = viz.createFigure('Robust image denoising');
fig.Position = [40 80 1600 470];
panels = {cleanImg, 'Clean image'; ...
          noisyImg, sprintf('Noisy (\\sigma = %.2f), PSNR %.1f dB', noiseSigma, psnrOf(noisyImg)); ...
          bestL2,   sprintf('L2 / Tikhonov, PSNR %.1f dB', psnrOf(bestL2)); ...
          bestTV,   sprintf('Total variation, PSNR %.1f dB', psnrOf(bestTV))};
for k = 1:4
    ax = axes('Parent', fig, 'Position', [0.01 + (k-1)*0.2475, 0.04, 0.235, 0.80]);
    viz.plot(ax, "Image", struct('Image', min(max(panels{k,1}, 0), 1), 'Colormap', 'gray'), ...
        'Title', panels{k,2}, 'ShowLegend', false);
end
saveFigure(viz, fig, fullfile(outDir, 'denoising'));

%% ========================================================================
%  2-3. Prototype classifiers vs an MLP on the ESL 3-class mixture
%  ========================================================================
fprintf('\n=== 2. Prototype methods vs MLP ===\n');
seed = 42;
gen  = OceanData.Generators.ESLMixtureGenerator('N', 300, 'Seed', seed, ...
    'NumClasses', 3, 'CentersPerClass', 10);
[X, y] = gen.generate();
split  = OceanData.DataArch.DataSplitter('Strategy', 'holdout', 'Ratio', 0.375).split(X, y);
Xtr = X(split.trainIdx, :);  ytr = y(split.trainIdx);
Xte = X(split.testIdx, :);   yte = y(split.testIdx);
K = 3;  R = 5;   % classes, prototypes per class

% --- K-means prototypes, fitted per class ---
km = OceanStats.Unsupervised.KMeans();
kmProto = zeros(K*R, 2);  kmLab = zeros(K*R, 1);
for k = 1:K
    km.fit(Xtr(ytr == k, :), R);
    kmProto((k-1)*R+1:k*R, :) = km.Centroids;
    kmLab((k-1)*R+1:k*R)      = k;
end

% --- LVQ1 refinement starting from the K-means prototypes ---
lvq = OceanStats.Supervised.Neighbors.LVQ1('R', R, 'LearningRate', 0.1, ...
    'MaxEpochs', 50, 'Decay', 0.02, 'Verbose', false);
lvq.fit(Xtr, ytr, 'InitPrototypes', kmProto, 'InitLabels', kmLab);

% --- MLP trained with the OceanDL autograd engine (full-batch AdamW) ---
% Trained twice: without weight decay (it overfits the 113 training points)
% and with decoupled weight decay 1.0 (Loshchilov & Hutter 2019).
numSteps = 1500;
[~,   histFree] = trainMLP(Xtr, ytr, Xte, yte, K, numSteps, 0,   seed);
[net, hist]     = trainMLP(Xtr, ytr, Xte, yte, K, numSteps, 1.0, seed);

% --- Test errors and decision regions on a dense grid ---
errKM  = mean(nearestPrototype(Xte, kmProto, kmLab) ~= yte);
errLVQ = mean(lvq.predict(Xte) ~= yte);
errMLP = hist.testErr(end);
fprintf('  MLP without weight decay: train error %.3f, test error %.3f, test loss %.3f\n', ...
    histFree.trainErr(end), histFree.testErr(end), histFree.testLoss(end));
fprintf('  MLP with weight decay 1.0: train error %.3f, test error %.3f, test loss %.3f\n', ...
    hist.trainErr(end), hist.testErr(end), hist.testLoss(end));
[~, errBayes] = gen.bayesPredict(Xte, yte);
fprintf('  test error: K-means %.3f | LVQ %.3f | MLP %.3f | Bayes %.3f\n', errKM, errLVQ, errMLP, errBayes);

pad = 0.5;
[xx, yy] = meshgrid(linspace(min(X(:,1))-pad, max(X(:,1))+pad, 200), ...
                    linspace(min(X(:,2))-pad, max(X(:,2))+pad, 200));
G = [xx(:), yy(:)];
[~, zzBayes] = max(gen.bayesDensity(G), [], 2);
[~, zzMLP]   = max(net.forward(OceanDL.Autograd.Tensor(G')).Data, [], 1);
regions = { reshape(nearestPrototype(G, kmProto, kmLab), size(xx)), kmProto, kmLab, ...
                sprintf('K-means, %d prototypes/class | test error %.3f', R, errKM); ...
            reshape(lvq.predict(G), size(xx)), lvq.Prototypes, lvq.PrototypeLabels, ...
                sprintf('LVQ, %d prototypes/class | test error %.3f', R, errLVQ); ...
            reshape(zzMLP, size(xx)), zeros(0, 2), zeros(0, 1), ...
                sprintf('MLP 2-32-32-3, AdamW (OceanDL) | test error %.3f', errMLP) };

fig = viz.createFigure('Prototype methods vs MLP');
fig.Position = [40 80 1650 560];
for k = 1:3
    ax = axes('Parent', fig, 'Position', [0.04 + (k-1)*0.325, 0.11, 0.28, 0.78]);
    d = struct('X', X, 'Y', y, 'Prototypes', regions{k,2}, 'ProtoLabels', regions{k,3}, ...
        'Grid', struct('xx', xx, 'yy', yy, 'zz', regions{k,1}), ...
        'BayesGrid', struct('xx', xx, 'yy', yy, 'zz', reshape(zzBayes, size(xx))), ...
        'DataPointSize', 10, 'ProtoSize', 55, 'StippleDensity', 70);
    viz.plot(ax, 'PrototypeClassifier', d, 'Title', regions{k,4}, ...
        'XLabel', 'x_1', 'YLabel', 'x_2', 'ShowLegend', k == 1);
end
saveFigure(viz, fig, fullfile(outDir, 'prototypes_vs_mlp'));

fig = viz.createFigure('MLP learning curves');
fig.Position = [40 80 1300 470];
steps = (1:numSteps)';
ax = axes('Parent', fig, 'Position', [0.06 0.14 0.42 0.74]);
viz.plot(ax, 'LearningCurve', struct('X', steps, 'Lines', {{ ...
        struct('Y', histFree.trainLoss, 'Name', 'Train, no decay'), ...
        struct('Y', histFree.testLoss,  'Name', 'Test, no decay', 'LineStyle', '--'), ...
        struct('Y', hist.trainLoss,     'Name', 'Train, weight decay 1.0'), ...
        struct('Y', hist.testLoss,      'Name', 'Test, weight decay 1.0', 'LineStyle', '--')}}), ...
    'Title', 'Cross-entropy loss', 'XLabel', 'AdamW step', 'YLabel', 'Loss');
ax = axes('Parent', fig, 'Position', [0.56 0.14 0.42 0.74]);
viz.plot(ax, 'LearningCurve', struct('X', steps, 'Lines', {{ ...
        struct('Y', histFree.trainErr, 'Name', 'Train, no decay'), ...
        struct('Y', histFree.testErr,  'Name', 'Test, no decay', 'LineStyle', '--'), ...
        struct('Y', hist.trainErr,     'Name', 'Train, weight decay 1.0'), ...
        struct('Y', hist.testErr,      'Name', 'Test, weight decay 1.0', 'LineStyle', '--')}}, ...
        'RefLines', {{struct('Y', errBayes, 'Name', 'Bayes error')}}), ...
    'Title', 'Misclassification rate', 'XLabel', 'AdamW step', 'YLabel', 'Error');
saveFigure(viz, fig, fullfile(outDir, 'mlp_learning_curves'));

%% ========================================================================
%  4. Jack's car rental: policy iteration (Sutton & Barto, Example 4.2)
%  ========================================================================
fprintf('\n=== 4. Jack''s car rental ===\n');
env = OceanStats.RL.Env.Classic.ModifiedCarRentalEnv('MaxCars', 20, 'MaxMove', 5, ...
    'RentReward', 10, 'MoveCost', 2, 'ParkingThreshold', 10, 'ParkingPenalty', 0, ...
    'FreeShuttle', false);
solver = OceanStats.RL.DP.PolicyIterationSolver(env, 'DiscountFactor', 0.9, 'Theta', 1e-4);
tic;
[V, policy, info] = solver.run();
fprintf('  converged in %d policy iterations (%.1f s)\n', info.PolicySteps, toc);

fig = viz.createFigure('Jack''s car rental');
fig.Position = [40 80 1300 520];
rl = OceanViz.Strategies.TabularRLStrategy();
rlData = struct('PolicyMatrix', policy, 'ValueMatrix', V);
ax = axes('Parent', fig, 'Position', [0.06 0.12 0.38 0.76]);
rl.draw(ax, rlData, struct('Mode', "PolicyHeatmap", ...
    'Title', sprintf('Optimal policy (cars moved), %d iterations', info.PolicySteps)), viz.Colors);
ax = axes('Parent', fig, 'Position', [0.56 0.12 0.40 0.76]);
rl.draw(ax, rlData, struct('Mode', "ValueSurface", 'Title', 'Optimal state value v_*(s)'), viz.Colors);
saveFigure(viz, fig, fullfile(outDir, 'car_rental_policy_iteration'));

fprintf('\n[DONE] Figures written to %s\n', outDir);

%% ========================================================================
%  Local helpers
%  ========================================================================
function [bestF, bestLam] = bestOverGrid(lambdas, scoreFn, solveFn)
% BESTOVERGRID  Solve for each lambda and keep the highest-scoring result.
bestScore = -inf;
for lam = lambdas
    f = solveFn(lam);
    s = scoreFn(f);
    fprintf('    lambda %.3f -> PSNR %.2f dB\n', lam, s);
    if s > bestScore
        bestScore = s;  bestF = f;  bestLam = lam;
    end
end
end

function [net, hist] = trainMLP(Xtr, ytr, Xte, yte, K, numSteps, weightDecay, seed)
% TRAINMLP  Full-batch AdamW training of a 2-32-32-K ReLU network with the
%   OceanDL autograd engine; records train / test loss and error per step.
rng(seed);
net = OceanDL.nn.Sequential( ...
    OceanDL.nn.Linear(2, 32), OceanDL.nn.Activation("relu"), ...
    OceanDL.nn.Linear(32, 32), OceanDL.nn.Activation("relu"), ...
    OceanDL.nn.Linear(32, K));
opt = OceanDL.optim.ModuleOptimizer(net.parameters(), ...
    @() OceanStats.Optim.AdamWOptimizer('LR', 1e-2, 'WeightDecay', weightDecay));
xTr = OceanDL.Autograd.Tensor(Xtr');
xTe = OceanDL.Autograd.Tensor(Xte');
hist = struct('trainLoss', zeros(numSteps,1), 'testLoss', zeros(numSteps,1), ...
              'trainErr', zeros(numSteps,1), 'testErr', zeros(numSteps,1));
for step = 1:numSteps
    opt.zeroGrad();
    logits = net.forward(xTr);
    loss   = OceanDL.Autograd.F.crossEntropy(logits, ytr');
    backward(loss);
    opt.step();

    hist.trainLoss(step) = loss.item();
    [~, pTr] = max(logits.Data, [], 1);
    hist.trainErr(step) = mean(pTr(:) ~= ytr);
    zTe = net.forward(xTe);
    hist.testLoss(step) = OceanDL.Autograd.F.crossEntropy(zTe, yte').item();
    [~, pTe] = max(zTe.Data, [], 1);
    hist.testErr(step) = mean(pTe(:) ~= yte);
end
end

function yhat = nearestPrototype(X, P, labels)
% NEARESTPROTOTYPE  1-nearest-prototype classification (squared Euclidean).
D = sum(X.^2, 2) + sum(P.^2, 2).' - 2 * (X * P.');
[~, idx] = min(D, [], 2);
yhat = labels(idx);
end

function saveFigure(viz, fig, fileStem)
% SAVEFIGURE  Export through OceanViz, then cap the width at 1800 px so the
%   README images stay small.
viz.export(fig, fileStem);
img = imread(fileStem + ".png");
if size(img, 2) > 1800
    img = imresize(img, 1800 / size(img, 2));
    imwrite(img, fileStem + ".png");
end
close(fig);
end
