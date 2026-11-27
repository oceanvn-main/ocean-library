% EXP_FACTOR_ANALYSIS  ESL Ex. 14.15 — Factor Analysis vs PCA.
%
%   REFERENCE:  ESL Exercise 14.15 (p.559), §14.7.1.
%   GOAL:       PCA aligns with max-variance X3 (Var=100).
%               FA recovers correlated latent structure X1≈X2 (shared Z1).
%   MODEL:      X1=Z1, X2=X1+0.001*Z2, X3=10*Z3  (N=200, p=3)
%   REPRODUCIBILITY: rng(42, 'twister')
%
clc; clear; rng(42, 'twister');

% === 1. Data Generation ===
N  = 200;
Z  = randn(N, 3);
X1 = Z(:,1);
X2 = X1 + 0.001 * Z(:,2);
X3 = 10 * Z(:,3);
X  = [X1, X2, X3];

fprintf('=== Data ===\n');
fprintf('  Var(X1)=%.4f  Var(X2)=%.4f  Var(X3)=%.4f\n', var(X1), var(X2), var(X3));
fprintf('  Cor(X1,X2)=%.6f  Cor(X1,X3)=%.6f\n', corr(X1,X2), corr(X1,X3));

% === 2. PCA ===
fprintf('\n=== PCA ===\n');
mu_x = mean(X, 1);
Xc   = X - mu_x;
[~, S_svd, V] = svd(Xc, 'econ');
pc1 = V(:,1);
ev  = diag(S_svd).^2 / (N-1);
pct = 100 * ev / sum(ev);
fprintf('  PC1 direction: [%.4f, %.4f, %.4f]\n', pc1(1), pc1(2), pc1(3));
fprintf('  Variance:  PC1=%.1f%%  PC2=%.1f%%  PC3=%.1f%%\n', pct(1), pct(2), pct(3));

% === 3. Factor Analysis (q=1, no rotation) ===
fprintf('\n=== Factor Analysis (q=1) ===\n');
fa = OceanStats.Unsupervised.FactorAnalysis();
fa.fit(X, 1, 'Rotation', 'none');
fa.printSummary();
A1 = fa.getLoadings();   % (3 x 1)
fprintf('  FA1 loading: [%.4f, %.4f, %.4f]\n', A1(1), A1(2), A1(3));

% Normalize for direction comparison
pc1_n = pc1 / norm(pc1);
fa1_n = A1  / norm(A1);

% === 4. Visualization ===
viz = OceanViz.Core('Theme', 'Dark');
bg  = [0.10 0.10 0.13];

% --- Figure 1: 3D scatter with PC1 and FA1 arrows ---
fig1 = figure('Name', 'ESL Ex 14.15 — PCA vs FA (3D)', ...
    'Color', bg, 'Position', [30 50 800 680]);
ax3d = axes(fig1);

scatter3(ax3d, X1, X2, X3, 14, 'filled', ...
    'MarkerFaceColor', [0.5 0.6 0.9], 'MarkerFaceAlpha', 0.3);
hold(ax3d, 'on');

scale = 28;
q3d = @(dir, col) quiver3(ax3d, 0, 0, 0, ...
    dir(1)*scale, dir(2)*scale, dir(3)*scale, ...
    'Color', col, 'LineWidth', 3, 'MaxHeadSize', 0.5, 'AutoScale', 'off');
q3d(pc1_n, [1.0 0.25 0.25]);   % PC1 red
q3d(fa1_n, [0.2 0.9 0.9]);     % FA1 cyan
hold(ax3d, 'off');

xlabel(ax3d, 'X_1', 'Color', 'w', 'FontSize', 11);
ylabel(ax3d, 'X_2', 'Color', 'w', 'FontSize', 11);
zlabel(ax3d, 'X_3', 'Color', 'w', 'FontSize', 11);
title(ax3d, 'PC1 (red) aligns with X3 axis | FA1 (cyan) aligns with X1-X2 plane', ...
    'Color', 'w', 'FontSize', 9);
legend(ax3d, {'Data', 'PC1 (max variance)', 'FA1 (latent Z_1)'}, ...
    'TextColor', 'w', 'Color', [0.15 0.15 0.2], 'FontSize', 9);
view(ax3d, [-40, 18]); grid(ax3d, 'on');
viz.styleAxes(ax3d);
sgtitle(fig1, 'ESL Ex 14.15 — Factor Analysis vs PCA', ...
    'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

% --- Figure 2: Loading analysis (2x3 panel) ---
fig2 = figure('Name', 'ESL Ex 14.15 — Analysis', ...
    'Color', bg, 'Position', [860 50 1050 680]);
tl2  = tiledlayout(fig2, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
vn   = {'X_1', 'X_2', 'X_3'};

% (A) Empirical covariance
ax = nexttile(tl2, 1);
Semp = cov(X);
imagesc(ax, Semp); colormap(ax, parula); colorbar(ax);
set(ax, 'XTick', 1:3, 'XTickLabel', vn, 'YTick', 1:3, 'YTickLabel', vn);
title(ax, 'Empirical Cov', 'Color', 'w', 'FontSize', 10);
for i=1:3; for j=1:3
    text(ax,j,i,sprintf('%.1f',Semp(i,j)),'HorizontalAlignment','center',...
        'Color','k','FontSize',9,'FontWeight','bold');
end; end
viz.styleAxes(ax);

% (B) FA model covariance
ax = nexttile(tl2, 2);
Sfa = fa.getCovModel();
imagesc(ax, Sfa); colormap(ax, parula); colorbar(ax);
set(ax, 'XTick', 1:3, 'XTickLabel', vn, 'YTick', 1:3, 'YTickLabel', vn);
title(ax, 'FA Model Cov = A*A'' + diag(Psi)', 'Color', 'w', 'FontSize', 10);
for i=1:3; for j=1:3
    text(ax,j,i,sprintf('%.1f',Sfa(i,j)),'HorizontalAlignment','center',...
        'Color','k','FontSize',9,'FontWeight','bold');
end; end
viz.styleAxes(ax);

% (C) PC1 vs FA1 loading magnitudes
ax = nexttile(tl2, 3);
b = bar(ax, [abs(pc1_n), abs(fa1_n)], 'grouped');
b(1).FaceColor = [1.0 0.25 0.25];
b(2).FaceColor = [0.2 0.9  0.9];
set(ax, 'XTickLabel', vn, 'FontSize', 10);
ylabel(ax, '|Component|', 'Color', 'w', 'FontSize', 10);
title(ax, 'PC1 vs FA1 Direction', 'Color', 'w', 'FontSize', 10);
legend(ax, {'PC1','FA1'}, 'TextColor', 'w', 'Color', [0.15 0.15 0.2], 'FontSize', 9);
ylim(ax, [0, 1.1]); viz.styleAxes(ax);

% (D) Communality
ax = nexttile(tl2, 4);
comm = fa.Communality;
b2 = bar(ax, comm, 'FaceColor', 'flat');
b2.CData = [0.3 0.85 0.4; 0.3 0.85 0.4; 0.9 0.35 0.3];
set(ax, 'XTickLabel', vn, 'FontSize', 10);
ylabel(ax, 'Communality h^2', 'Color', 'w', 'FontSize', 10);
title(ax, 'FA Communality (X3 \approx 0)', 'Color', 'w', 'FontSize', 10);
yline(ax, 1, '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
ylim(ax, [0, 1.15]); viz.styleAxes(ax);

% (E) Specific variances
ax = nexttile(tl2, 5);
b3 = bar(ax, fa.SpecificVar, 'FaceColor', 'flat');
b3.CData = [0.9 0.6 0.2; 0.9 0.6 0.2; 0.3 0.6 0.95];
set(ax, 'XTickLabel', vn, 'FontSize', 10);
ylabel(ax, 'Specific Var (Psi)', 'Color', 'w', 'FontSize', 10);
title(ax, 'FA Specific Variances (X3 high)', 'Color', 'w', 'FontSize', 10);
viz.styleAxes(ax);

% (F) Summary text
ax = nexttile(tl2, 6); axis(ax, 'off');
txt = sprintf([...
    'PC1: [%+.3f, %+.3f, %+.3f]\n'...
    'FA1: [%+.3f, %+.3f, %+.3f]\n\n'...
    'Variance:\n'...
    '  PC1=%4.1f%%  PC2=%4.1f%%  PC3=%4.1f%%\n\n'...
    'Communality:\n'...
    '  X1=%.4f  X2=%.4f  X3=%.4f\n\n'...
    'Key insight:\n'...
    '  PCA sees Var(X3)=100\n'...
    '  FA sees Cor(X1,X2)~1\n'...
    '  FA ignores uncorrelated X3\n'...
    '  as specific noise'],...
    pc1_n(1),pc1_n(2),pc1_n(3),...
    fa1_n(1),fa1_n(2),fa1_n(3),...
    pct(1),pct(2),pct(3),...
    fa.Communality(1),fa.Communality(2),fa.Communality(3));
text(ax, 0.05, 0.95, txt, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'FontSize', 9, 'Color', 'w', ...
    'FontName', 'Courier', 'Interpreter', 'none');
viz.styleAxes(ax);

sgtitle(fig2, 'ESL Ex 14.15 — Loading Analysis: PCA vs Factor Analysis', ...
    'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

% === 5. Summary ===
fprintf('\n=== Key Results (N=%d) ===\n', N);
fprintf('  PC1 (normalized): [%+.4f, %+.4f, %+.4f]\n', pc1_n(1), pc1_n(2), pc1_n(3));
fprintf('  FA1 (normalized): [%+.4f, %+.4f, %+.4f]\n', fa1_n(1), fa1_n(2), fa1_n(3));
fprintf('  Communality: X1=%.4f  X2=%.4f  X3=%.4f\n', ...
    fa.Communality(1), fa.Communality(2), fa.Communality(3));
fprintf('\n  Conclusion:\n');
fprintf('    PC1 ~ e3: dominated by Var(X3)=100\n');
fprintf('    FA1 ~ (e1+e2)/sqrt(2): recovers shared latent Z1\n');
fprintf('    X3 communality ~ 0: FA treats it as uncorrelated specific noise\n');
