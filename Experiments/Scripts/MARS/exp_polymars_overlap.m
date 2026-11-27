%% exp_polymars_overlap.m
% Experiment: Compare Standard (Indicator) MARS vs PolyMARS on a 3-class
% overlapping problem.
%
% Run from project root:
%   cd('c:\Ocean Library\Matlab code practice')
%   run('Experiments/exp_polymars_overlap.m')
%
% Output: Results/polymars_overlap.png

addpath(genpath(pwd));

%% 1. Generate overlapping 3-class data
rng(42);
N    = 300;                  % samples per class
mu   = [0 0; 3 0; 1.5 2.6]; % class centroids
Sig  = 1.4 * eye(2);        % shared covariance (overlap intentional)

X = []; Y = [];
for k = 1:3
    Xk = mvnrnd(mu(k,:), Sig, N);
    X  = [X; Xk];                  %#ok<AGROW>
    Y  = [Y; k * ones(N, 1)];      %#ok<AGROW>
end

% Shuffle
idx = randperm(size(X,1));
X = X(idx,:);
Y = Y(idx);

fprintf('Data: %d samples, 3 classes, 2 features.\n', length(Y));

%% 2. Fit standard Indicator MARS
fprintf('\n--- Standard Indicator MARS ---\n');
mars_ind = OceanStats.Supervised.MARS.MARS( ...
    'Objective',           'LeastSquares', ...
    'MaxTerms',            15, ...
    'MaxInteractionDegree', 1);
mars_ind.fit(X, Y);
yhat_ind  = mars_ind.predict(X);
acc_ind   = mean(yhat_ind == Y);
fprintf('Training Accuracy: %.2f%%\n', acc_ind * 100);

%% 3. Fit PolyMARS
fprintf('\n--- PolyMARS (Multinomial Log-Likelihood) ---\n');
mars_poly = OceanStats.Supervised.MARS.MARS( ...
    'Objective',           'Multinomial', ...
    'MaxTerms',            15, ...
    'MaxInteractionDegree', 1);
mars_poly.fit(X, Y);
yhat_poly = mars_poly.predict(X);
acc_poly  = mean(yhat_poly == Y);
fprintf('Training Accuracy: %.2f%%\n', acc_poly * 100);

%% 4. Build decision boundary grid
x1_range = linspace(min(X(:,1)) - 0.5, max(X(:,1)) + 0.5, 200);
x2_range = linspace(min(X(:,2)) - 0.5, max(X(:,2)) + 0.5, 200);
[G1, G2]  = meshgrid(x1_range, x2_range);
Xgrid     = [G1(:), G2(:)];

% Indicator MARS: hard labels on grid
yg_ind = mars_ind.predict(Xgrid);
Zind   = reshape(yg_ind, size(G1));

% PolyMARS: use the winning class probability for contour shading
Pg_poly  = mars_poly.predictProba(Xgrid);   % (Ngrid x 3)
[~, Zpc] = max(Pg_poly, [], 2);             % hard label from max proba
Zpoly    = reshape(Zpc, size(G1));

% For smooth probability shading: max proba value
Pmax_poly = reshape(max(Pg_poly, [], 2), size(G1));

%% 5. Plot
colors = [0.2 0.5 1.0;    % class 1 - blue
          1.0 0.4 0.2;    % class 2 - orange
          0.2 0.75 0.4];  % class 3 - green

fig = figure('Color', [0.12 0.12 0.15], 'Position', [100 100 1200 520]);

% --- Left panel: Indicator MARS ---
ax1 = subplot(1,2,1);
hold on;

% Soft background fill via imagesc
imagesc(x1_range, x2_range, Zind);
colormap(ax1, brighten(colors, 0.65));
clim([1 3]);
alpha(0.45);

% Decision boundary contours
contour(G1, G2, Zind, [1.5 2.5], 'w-', 'LineWidth', 1.5);

% Scatter data
for k = 1:3
    idx_k = Y == k;
    scatter(X(idx_k,1), X(idx_k,2), 28, colors(k,:), 'filled', ...
        'MarkerEdgeColor', 'w', 'LineWidth', 0.5, 'MarkerFaceAlpha', 0.8);
end

title(sprintf('Indicator MARS  (acc = %.1f%%)', acc_ind*100), ...
    'Color', 'w', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x_1', 'Color', [0.8 0.8 0.8]);
ylabel('x_2', 'Color', [0.8 0.8 0.8]);
ax1.Color = [0.18 0.18 0.22];
ax1.XColor = [0.7 0.7 0.7]; ax1.YColor = [0.7 0.7 0.7];
ax1.GridColor = [0.4 0.4 0.4]; grid on;

% --- Right panel: PolyMARS ---
ax2 = subplot(1,2,2);
hold on;

% Shade by max-probability strength
imagesc(x1_range, x2_range, Zpoly);
colormap(ax2, brighten(colors, 0.65));
clim([1 3]);
alpha(0.45);

% Overlay confidence contours of PolyMARS
contour(G1, G2, Pmax_poly, [0.55 0.70 0.85], ...
    'LineColor', [0.9 0.9 0.9], 'LineWidth', 1.0, 'LineStyle', ':');

% Decision boundary
contour(G1, G2, Zpoly, [1.5 2.5], 'w-', 'LineWidth', 1.5);

% Scatter data
for k = 1:3
    idx_k = Y == k;
    scatter(X(idx_k,1), X(idx_k,2), 28, colors(k,:), 'filled', ...
        'MarkerEdgeColor', 'w', 'LineWidth', 0.5, 'MarkerFaceAlpha', 0.8);
end

title(sprintf('PolyMARS  (acc = %.1f%%)', acc_poly*100), ...
    'Color', 'w', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('x_1', 'Color', [0.8 0.8 0.8]);
ylabel('x_2', 'Color', [0.8 0.8 0.8]);
ax2.Color = [0.18 0.18 0.22];
ax2.XColor = [0.7 0.7 0.7]; ax2.YColor = [0.7 0.7 0.7];
ax2.GridColor = [0.4 0.4 0.4]; grid on;

% Dotted-contour legend note
annotation(fig, 'textbox', [0.52 0.04 0.45 0.05], ...
    'String', 'Dotted lines: PolyMARS confidence contours (55%, 70%, 85%)', ...
    'Color', [0.7 0.7 0.7], 'FontSize', 9, 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center');

sgtitle('Standard MARS vs PolyMARS — 3-Class Overlap', ...
    'Color', 'w', 'FontSize', 15, 'FontWeight', 'bold');

%% 6. Save
if ~exist('Results', 'dir'); mkdir('Results'); end
outPath = fullfile('Results', 'polymars_overlap.png');
exportgraphics(fig, outPath, 'Resolution', 150);
fprintf('\nFigure saved to: %s\n', outPath);
