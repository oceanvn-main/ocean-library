% EXP_SIGMOID_SATURATION  Explore the failure of sigmoid at high confidence.
%
%   PHENOMENON:
%     The sigmoid function sigma(z) = 1/(1+exp(-z)) maps any real input to (0,1).
%     When |z| is large, the output is near 0 or 1 (high "confidence").
%     At these saturation points, the gradient sigma'(z) = sigma(z)*(1-sigma(z))
%     collapses toward zero — the neuron stops learning regardless of how large
%     the error is. This is the root cause of the vanishing gradient problem
%     in deep sigmoid networks.
%
%   ASSUMPTIONS:
%     - Gradients computed analytically from closed-form derivatives (no autograd).
%     - Deep network experiment uses L layers of width 1 (scalar chain) so
%       the gradient product is exact and visually interpretable.
%     - Weight update approximation: delta_w = -lr * gradient (SGD step 0).
%     - All computations derived from inputs; no hardcoded expected outputs.
%     - Comparison vs ReLU and tanh derived from their own analytical gradients.
%
%   REPRODUCIBILITY: rng(2024, 'twister')
%
%   PANELS:
%     1. Sigmoid curve + derivative + gradient dead zones
%     2. Gradient magnitude as a function of pre-activation z
%     3. Vanishing gradients through depth: sigmoid vs tanh vs ReLU
%     4. Weight update collapse: how much does a neuron learn at high confidence?
%
%   Location: Experiments/Scripts/NeuralNet/exp_sigmoid_saturation.m

clc; clear; close all;
rng(2024, 'twister');

% =========================================================================
% DESIGN PARAMETERS  (all explicit — no silent choices)
% =========================================================================
Z_RANGE     = linspace(-10, 10, 2000);   % pre-activation sweep range
N_LAYERS    = 10;                         % depth for vanishing gradient test
LAYER_WIDTH = 32;                         % neurons per hidden layer (scalar variant too)
LR          = 0.01;                       % learning rate for update-collapse panel
HIGH_CONF_THRESHOLD = 0.02;              % gradient < this = "dead zone"

% =========================================================================
% ANALYTICAL FUNCTIONS  (derived, not hardcoded)
% =========================================================================
sigmoid       = @(z) 1 ./ (1 + exp(-z));
sigmoid_grad  = @(z) sigmoid(z) .* (1 - sigmoid(z));   % σ'(z) = σ(1-σ)

relu          = @(z) max(0, z);
relu_grad     = @(z) double(z > 0);                     % 1 for z>0, 0 otherwise

tanh_fn       = @(z) tanh(z);
tanh_grad     = @(z) 1 - tanh(z).^2;                   % sech^2(z)

% =========================================================================
% FIGURE SETUP  — dark theme
% =========================================================================
BG   = [0.10 0.12 0.16];
PANEL= [0.14 0.16 0.21];
TXT  = [0.93 0.93 0.93];
DIM  = [0.50 0.53 0.60];
BLUE = [0.30 0.65 0.95];
RED  = [0.95 0.35 0.30];
GRN  = [0.22 0.82 0.50];
AMB  = [0.99 0.72 0.15];
PURP = [0.75 0.40 0.95];

fig = figure('Name',       'Sigmoid Saturation — Failure at High Confidence', ...
             'Color',       BG, ...
             'NumberTitle', 'off', ...
             'Position',    [30, 30, 1500, 900]);

% =========================================================================
% PANEL 1: Sigmoid curve + derivative + dead zones
% =========================================================================
ax1 = subplot(2, 2, 1);
styleAx(ax1, BG, TXT, DIM);
hold(ax1, 'on');

% Dead-zone shading: |gradient| < HIGH_CONF_THRESHOLD
deadMask = sigmoid_grad(Z_RANGE) < HIGH_CONF_THRESHOLD;
fillBand(ax1, Z_RANGE, deadMask, -0.05, 1.05, [0.95 0.30 0.30], 0.12);

% Sigmoid curve
plot(ax1, Z_RANGE, sigmoid(Z_RANGE), 'Color', BLUE, 'LineWidth', 2.5, ...
    'DisplayName', '\sigma(z) — output');

% Derivative curve
plot(ax1, Z_RANGE, sigmoid_grad(Z_RANGE), 'Color', AMB, 'LineWidth', 2.0, ...
    'LineStyle', '--', 'DisplayName', "\sigma'(z) — gradient");

% Max gradient reference line
[~, maxGidx] = max(sigmoid_grad(Z_RANGE));
yline(ax1, 0.25, ':', 'Color', [DIM 0.8], 'LineWidth', 1.2);
text(ax1, 6, 0.27, 'max gradient = 0.25', 'Color', DIM, 'FontSize', 8.5);

% Zero gradient reference
yline(ax1, HIGH_CONF_THRESHOLD, ':', 'Color', [RED 0.9], 'LineWidth', 1.0);
text(ax1, 3.5, HIGH_CONF_THRESHOLD + 0.02, ...
    sprintf('dead threshold = %.2f', HIGH_CONF_THRESHOLD), ...
    'Color', RED, 'FontSize', 8.5);

% Annotations on dead zones
text(ax1, -8.5, 0.5, sprintf('DEAD\n(σ≈0)'), ...
    'Color', RED, 'FontSize', 9, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');
text(ax1, 8.5, 0.5, sprintf('DEAD\n(σ≈1)'), ...
    'Color', RED, 'FontSize', 9, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');

legend(ax1, 'TextColor', TXT, 'Color', PANEL, 'EdgeColor', 'none', ...
    'FontSize', 9, 'Location', 'east');
title(ax1, 'Panel 1: Sigmoid Curve + Gradient', 'Color', TXT, 'FontSize', 11);
xlabel(ax1, 'Pre-activation  z  =  Wx + b', 'Color', DIM);
ylabel(ax1, 'Value', 'Color', DIM);
xlim(ax1, [-10 10]); ylim(ax1, [-0.05 1.05]);

% =========================================================================
% PANEL 2: Gradient magnitude comparison — sigmoid vs tanh vs ReLU
% =========================================================================
ax2 = subplot(2, 2, 2);
styleAx(ax2, BG, TXT, DIM);
hold(ax2, 'on');

% Dead-zone shading for sigmoid
fillBand(ax2, Z_RANGE, deadMask, -0.05, 1.10, [0.95 0.30 0.30], 0.10);

plot(ax2, Z_RANGE, sigmoid_grad(Z_RANGE), ...
    'Color', BLUE, 'LineWidth', 2.2, 'DisplayName', "σ'(z)  [sigmoid]");
plot(ax2, Z_RANGE, tanh_grad(Z_RANGE), ...
    'Color', GRN,  'LineWidth', 2.2, 'DisplayName', "tanh'(z)  [tanh]");
plot(ax2, Z_RANGE, relu_grad(Z_RANGE), ...
    'Color', AMB,  'LineWidth', 2.2, 'DisplayName', "relu'(z)  [ReLU]");

% Gradient ceiling lines
yline(ax2, 0.25, ':', 'Color', [BLUE 0.7], 'LineWidth', 1.0);
yline(ax2, 1.00, ':', 'Color', [GRN  0.7], 'LineWidth', 1.0);
text(ax2, 6, 0.27, 'σ max = 0.25', 'Color', BLUE, 'FontSize', 8);
text(ax2, 6, 1.02, 'tanh max = 1.0', 'Color', GRN, 'FontSize', 8);
text(ax2, 2.5, 1.02, 'ReLU = 1.0 (flat)', 'Color', AMB, 'FontSize', 8);

legend(ax2, 'TextColor', TXT, 'Color', PANEL, 'EdgeColor', 'none', ...
    'FontSize', 9, 'Location', 'northeast');
title(ax2, "Panel 2: Gradient Comparison — Why σ' Is Fatally Small", ...
    'Color', TXT, 'FontSize', 11);
xlabel(ax2, 'Pre-activation  z', 'Color', DIM);
ylabel(ax2, "Gradient magnitude  f'(z)", 'Color', DIM);
xlim(ax2, [-10 10]); ylim(ax2, [-0.05 1.10]);

% =========================================================================
% PANEL 3: Vanishing gradients through depth
%   Scenario: gradient signal travelling backward through L layers.
%   At each layer, the gradient is multiplied by f'(z_l) for that layer.
%   We sweep over different pre-activation magnitudes |z| and show how
%   the gradient at the INPUT layer decays with depth.
% =========================================================================
ax3 = subplot(2, 2, 3);
styleAx(ax3, BG, TXT, DIM);
hold(ax3, 'on');

depths = 1:N_LAYERS;

% Test three representative pre-activation regimes
z_cases = struct( ...
    'label', {'z = ±0.5 (near center)', 'z = ±2 (moderate)', 'z = ±5 (saturated)'}, ...
    'z',     {0.5,                        2.0,                  5.0});

styles = {'--', '-', '-.'};
alphas = [0.7, 1.0, 1.0];

for ci = 1:numel(z_cases)
    zv = z_cases(ci).z;

    % Sigmoid: gradient shrinks by σ'(z) per layer
    g_sig = sigmoid_grad(zv) .^ depths;

    % Tanh: gradient shrinks by tanh'(z) per layer
    g_tanh = tanh_grad(zv)   .^ depths;

    % ReLU: gradient stays at 1 (for z > 0) → no vanishing
    g_relu = relu_grad(zv)    .^ depths;

    plot(ax3, depths, g_sig,  'Color', [BLUE  alphas(ci)], ...
        'LineWidth', 2.0, 'LineStyle', styles{ci}, ...
        'DisplayName', sprintf('sigmoid  %s', z_cases(ci).label));
    plot(ax3, depths, g_tanh, 'Color', [GRN   alphas(ci)], ...
        'LineWidth', 2.0, 'LineStyle', styles{ci}, ...
        'DisplayName', sprintf('tanh     %s', z_cases(ci).label));
    if ci == 2
        plot(ax3, depths, g_relu, 'Color', AMB, ...
            'LineWidth', 2.5, 'LineStyle', '-', ...
            'DisplayName', 'ReLU     z>0  (no vanishing)');
    end
end

% Compute meaningful threshold: when gradient < 1% of layer 1 gradient
[~] = yline(ax3, 1e-4, ':', 'Color', [RED 0.8], 'LineWidth', 1.2);
text(ax3, 8, 1.5e-4, 'effective zero (1e-4)', 'Color', RED, 'FontSize', 8);

set(ax3, 'YScale', 'log');
legend(ax3, 'TextColor', TXT, 'Color', PANEL, 'EdgeColor', 'none', ...
    'FontSize', 8, 'Location', 'southwest');
title(ax3, 'Panel 3: Gradient Vanishing Through Depth  [log scale]', ...
    'Color', TXT, 'FontSize', 11);
xlabel(ax3, 'Layer depth (from output backward)', 'Color', DIM);
ylabel(ax3, 'Gradient magnitude reaching this layer', 'Color', DIM);
xlim(ax3, [1 N_LAYERS]);

% =========================================================================
% PANEL 4: Weight update collapse at high confidence
%   For a single neuron with true label y=0, current output a=sigmoid(z):
%     loss    = 0.5 * (a - y)^2
%     dL/dw   = (a - y) * sigma'(z) * x
%   We sweep z from -10 to 10, with x = 1, and show how |dL/dw|
%   collapses even though the ERROR (a - y) is large.
% =========================================================================
ax4 = subplot(2, 2, 4);
styleAx(ax4, BG, TXT, DIM);
hold(ax4, 'on');

y_true = 0;   % true label: negative class
x_val  = 1;   % input feature = 1 (unit input)
z_sweep = linspace(-10, 10, 2000);

a_out     = sigmoid(z_sweep);
error_sq  = (a_out - y_true).^2;          % squared error (loss surface)
grad_loss = (a_out - y_true) .* sigmoid_grad(z_sweep) .* x_val;  % dL/dw
weight_upd = abs(LR * grad_loss);          % |Δw| per gradient step

% Normalize for joint display
error_norm = error_sq ./ max(error_sq);
grad_norm  = abs(grad_loss) ./ max(abs(grad_loss) + 1e-12);

% High-confidence shading (saturated positive side: σ(z) > 0.98 and wrong)
wrongConf = z_sweep > 4;   % sigmoid(4) ≈ 0.98 — confidently wrong
fillBand(ax4, z_sweep, wrongConf, -0.05, 1.05, RED, 0.15);
text(ax4, 7, 0.85, sprintf('Confidently\nWRONG\n(σ ≈ 1 but y=0)'), ...
    'Color', RED, 'FontSize', 8.5, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');

yyaxis(ax4, 'left');
ax4.YAxis(1).Color = BLUE;
plot(ax4, z_sweep, error_norm, 'Color', RED, 'LineWidth', 2.2, ...
    'DisplayName', 'Squared error (normalized)');
plot(ax4, z_sweep, a_out, 'Color', [BLUE 0.8], 'LineWidth', 1.5, ...
    'LineStyle', '--', 'DisplayName', '\sigma(z) — model output');
ylabel(ax4, 'Error / Output ', 'Color', DIM);

yyaxis(ax4, 'right');
ax4.YAxis(2).Color = AMB;
plot(ax4, z_sweep, weight_upd, 'Color', AMB, 'LineWidth', 2.5, ...
    'DisplayName', sprintf('|Δw| per step  (lr=%.3f)', LR));
ylabel(ax4, '|Δw|  weight update magnitude', 'Color', DIM);

% Find z where weight update is max
[~, maxUpdIdx] = max(weight_upd);
xline(ax4, z_sweep(maxUpdIdx), ':', 'Color', [AMB 0.8], 'LineWidth', 1.2);
text(ax4, z_sweep(maxUpdIdx) - 0.3, max(weight_upd) * 0.55, ...
    sprintf('max Δw\n at z≈%.1f', z_sweep(maxUpdIdx)), ...
    'Color', AMB, 'FontSize', 8, 'HorizontalAlignment', 'right');

legend(ax4, 'TextColor', TXT, 'Color', PANEL, 'EdgeColor', 'none', ...
    'FontSize', 9, 'Location', 'northwest');
title(ax4, 'Panel 4: Weight Update Collapse at High Confidence  (y_{true}=0)', ...
    'Color', TXT, 'FontSize', 11);
xlabel(ax4, 'Pre-activation  z  (larger = more confident)', 'Color', DIM);
xlim(ax4, [-10 10]);

% =========================================================================
% SUPER TITLE
% =========================================================================
annotation(fig, 'textbox', [0 0.96 1 0.04], ...
    'String',           'Sigmoid Saturation — Failure at High Confidence', ...
    'Color',            TXT, ...
    'FontSize',         15, ...
    'FontWeight',       'bold', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment',   'middle', ...
    'EdgeColor',        'none', ...
    'BackgroundColor',  BG);

% =========================================================================
% SUMMARY PRINTOUT
% =========================================================================
fprintf('\n=== SIGMOID SATURATION EXPERIMENT SUMMARY ===\n');
fprintf('  z range swept         : [%.0f, %.0f]  (%d points)\n', ...
    Z_RANGE(1), Z_RANGE(end), numel(Z_RANGE));
fprintf('  Max sigmoid gradient  : %.4f  (at z = 0)\n', max(sigmoid_grad(Z_RANGE)));
fprintf('  Max tanh gradient     : %.4f  (at z = 0)\n', max(tanh_grad(Z_RANGE)));
fprintf('  Dead zone (σ grade<%.2f): %.1f%%  of z-range\n', ...
    HIGH_CONF_THRESHOLD, 100 * mean(deadMask));
fprintf('  Depth tested          : %d layers\n', N_LAYERS);
fprintf('\n  Gradient reaching input layer (sigmoid, z=±5):\n');
fprintf('    After %d layers: %.2e  (effectively zero)\n', ...
    N_LAYERS, sigmoid_grad(5)^N_LAYERS);
fprintf('\n  Gradient reaching input layer (ReLU, z>0):\n');
fprintf('    After %d layers: %.4f  (no decay)\n', N_LAYERS, 1.0);
fprintf('\n  KEY INSIGHT:\n');
fprintf('    At z=5 (σ≈0.99), gradient per layer = %.6f\n', sigmoid_grad(5));
fprintf('    Through 5 layers:  gradient = %.2e\n', sigmoid_grad(5)^5);
fprintf('    Through 10 layers: gradient = %.2e\n', sigmoid_grad(5)^10);
fprintf('    The neuron is stuck — high error, near-zero update.\n');
fprintf('==============================================\n');

% =========================================================================
% LOCAL HELPER FUNCTIONS
% =========================================================================
function styleAx(ax, bg, txt, dim)
    set(ax, 'Color', bg, 'XColor', dim, 'YColor', dim, ...
        'GridColor', [0.3 0.3 0.35], 'GridAlpha', 0.4);
    ax.FontSize = 9;
    grid(ax, 'on');
end

function fillBand(ax, x, mask, ylo, yhi, col, alpha)
    % Fill vertical bands where mask == true
    inBand = false;
    for i = 1:numel(x)
        if mask(i) && ~inBand
            x0 = x(i); inBand = true;
        elseif ~mask(i) && inBand
            fill(ax, [x0 x(i-1) x(i-1) x0], [ylo ylo yhi yhi], col, ...
                'FaceAlpha', alpha, 'EdgeColor', 'none');
            inBand = false;
        end
    end
    if inBand
        fill(ax, [x0 x(end) x(end) x0], [ylo ylo yhi yhi], col, ...
            'FaceAlpha', alpha, 'EdgeColor', 'none');
    end
end
