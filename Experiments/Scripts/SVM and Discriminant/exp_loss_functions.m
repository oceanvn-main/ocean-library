% EXP_LOSS_FUNCTIONS  Reproduces ESL Figure 12.4
%
%   Plots various loss functions:
%     1. Hinge Loss: max(0, 1 - yf)
%     2. Binomial Deviance: log(1 + exp(-yf))
%     3. Squared Error: (1 - yf)^2
%     4. Class Huber:
%           -yf               if yf < -1
%           (1 - yf)^2 / 4    if -1 <= yf <= 1
%           0                 if yf > 1
%
%   Ref: ESL Chapter 12.3.2, Figure 12.4

clc; clear; close all;

ROOT = 'c:\Ocean Library\Matlab code practice';
addpath(ROOT);

fprintf('=== EXP_LOSS_FUNCTIONS: ESL Figure 12.4 Reproduction ===\n\n');

% 1. Data points for yf (margin)
yf = linspace(-3.2, 3.2, 1000);

% 2. Instantiate Ocean Library Loss Functions
lossObjs = {
    OceanStats.Loss.Classification.HingeLoss(), ...
    OceanStats.Loss.Classification.MulticlassDevianceLoss(), ...
    OceanStats.Loss.Classification.MarginSquaredLoss(), ...
    OceanStats.Loss.Classification.HuberisedSquareHingeLoss()
};

% True labels are all +1, so the prediction 'yf' is just 'f'
Y_true = ones(size(yf));
Y_pred = yf;

% Calculate the loss curves
loss_hinge = zeros(size(yf));
loss_binom = zeros(size(yf));
loss_sq    = zeros(size(yf));
loss_huber = zeros(size(yf));

for i = 1:length(yf)
    % The Loss classes aggregate with mean(), so we pass scalars to get the point-wise curve
    loss_hinge(i) = lossObjs{1}.forward(1, yf(i));
    
    % MulticlassDeviance takes (Y_true, Probabilities) where Y_true is 1D class labels
    % To map 'f' back to probability for class 1: p = 1 / (1 + exp(-f))
    p = 1 / (1 + exp(-yf(i)));
    % To match the slope of -1 on the left tail, we use negative log likelihood directly: log(1 + exp(-f))
    % The deviance loss uses -log(p) which is identical.
    loss_binom(i) = lossObjs{2}.forward(1, p);
    
    loss_sq(i)    = lossObjs{3}.forward(1, yf(i));
    loss_huber(i) = lossObjs{4}.forward(1, yf(i));
end

% Note on scaling: The Huberized Square Hinge Loss in our library is implemented 
% as max(0, 1-m)^2, which has a left tail slope of -4 (not -1 like ESL Figure 12.4).
% To align the plot visually exactly with ESL Figure 12.4, we divide it by 4 just for the plot.
loss_huber = loss_huber / 4;

% 3. Rendering via OceanViz.Core
try
    viz = OceanViz.Core('Theme', 'Scientific');
catch
    % Fallback if Light theme is missing
    viz = OceanViz.Core('Theme', 'Dark');
end

fig = figure('Name', 'Loss Functions (ESL Fig 12.4)', 'Units', 'normalized', 'Position', [0.2, 0.2, 0.5, 0.6]);
fig.Color = viz.Colors.Background;
ax = axes('Parent', fig);
viz.styleAxes(ax);
hold(ax, 'on'); 

% Manually override to match ESL colors roughly
color_hinge = [0.65, 0.15, 0.15];  % Dark red
color_binom = [0.45, 0.75, 0.45];  % Light green
color_sq    = [1.00, 0.55, 0.15];  % Orange
color_huber = [0.25, 0.35, 0.85];  % Blue

% Plot the curves
plot(ax, yf, loss_hinge, 'Color', color_hinge, 'LineWidth', 2.0, 'DisplayName', 'Hinge Loss');
plot(ax, yf, loss_binom, 'Color', color_binom, 'LineWidth', 2.0, 'DisplayName', 'Binomial Deviance');
plot(ax, yf, loss_sq,    'Color', color_sq,    'LineWidth', 2.0, 'DisplayName', 'Squared Error');
plot(ax, yf, loss_huber, 'Color', color_huber, 'LineWidth', 2.0, 'DisplayName', 'Class Huber');

% Axes Limits and Zero Lines
xlim(ax, [-3.2, 3.2]);
ylim(ax, [-0.2, 3.2]);

xline(ax, 0, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0, 'HandleVisibility', 'off');
yline(ax, 0, '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0, 'HandleVisibility', 'off');

% Labels and Title
xlabel(ax, 'yf', 'FontSize', 14, 'Color', viz.Colors.Text);
ylabel(ax, 'Loss', 'FontSize', 14, 'Color', viz.Colors.Text);
title(ax, '12. Flexible Discriminants', 'FontSize', 16, 'Color', viz.Colors.Text, 'HorizontalAlignment', 'center', 'FontWeight', 'normal');

% Legend
lgd = legend(ax, 'Location', 'northeast', 'FontSize', 12);
lgd.TextColor = viz.Colors.Text;
lgd.Color = viz.Colors.Background;
lgd.Box = 'on';

% Additional formatting to match the book
set(ax, 'TickDir', 'out', 'Box', 'on');
if isequal(fig.Color, [1 1 1])
    set(ax, 'XColor', 'k', 'YColor', 'k');
end

hold(ax, 'off');

% 4. Save Image
outDir = fullfile(ROOT, 'Results');
if ~exist(outDir, 'dir'), mkdir(outDir); end
savePath = fullfile(outDir, 'exp_loss_functions_fig12_4.png');
saveas(fig, savePath);

fprintf('Successfully saved Figure 12.4 to:\n  %s\n\n', savePath);
