% EXPERIMENT: Time Series Analysis — ARMA Simulations and Autocorrelations
%
% PURPOSE:
%   This script runs a series of simulations to study the behavior of Autoregressive
%   and Moving Average (ARMA) processes. It compares empirical and theoretical
%   Autocorrelation Functions (ACF) and Partial Autocorrelation Functions (PACF),
%   and verifies the MA(1) duality described in Section 3.7 of the textbook.
%
% ASSUMPTIONS:
%   1. White noise process e(t) is independent and identically distributed (i.i.d.)
%      Gaussian with zero mean and variance sigma^2 = 1.0.
%   2. All processes simulated are stationary (roots of AR characteristic polynomial
%      modulus < 1) to ensure theoretical ACF/PACF are well-defined.
%   3. A burn-in period of 100 steps is sufficient to eliminate initialization bias.
%
% REPRODUCIBILITY:
%   A fixed random seed (rng(42)) is set at the start of the experiment.
%
% See also: OceanMath.TimeSeries.Models.ARMA, OceanMath.TimeSeries.Diagnostics.autocorr,
%           OceanMath.TimeSeries.Diagnostics.parcorr, OceanViz.Core

clear;
clc;
close all;

% Set seed for reproducibility
rng(42);

% --- Simulation Parameters ---
N = 300;        % Length of time series realizations
sigma = 1.0;    % Standard deviation of white noise
maxLag = 20;    % Maximum lags for ACF and PACF

% Initialize OceanViz engine
viz = OceanViz.Core('Theme', 'Dark');

%% Part 1: Standard ARMA Processes (From Textbook Figure 3.1)
% Define the processes:
% (a) White Noise: Y_t = e_t
% (b) MA(1): Y_t = e_t + 0.8*e_{t-1}
% (c) AR(1) Positive: Y_t = 0.8*Y_{t-1} + e_t
% (d) AR(1) Negative: Y_t = -0.8*Y_{t-1} + e_t

models = { ...
    struct('name', 'White Noise', 'AR', [], 'MA', [], 'const', 0), ...
    struct('name', 'MA(1) theta=0.8', 'AR', [], 'MA', 0.8, 'const', 0), ...
    struct('name', 'AR(1) phi=0.8', 'AR', 0.8, 'MA', [], 'const', 0), ...
    struct('name', 'AR(1) phi=-0.8', 'AR', -0.8, 'MA', [], 'const', 0) ...
};

fprintf('=== PART 1: SIMULATING TEXTBOOK ARMA MODELS ===\n\n');

for i = 1:length(models)
    m = models{i};
    
    % 1. Create model object
    armaObj = OceanMath.TimeSeries.Models.ARMA(...
        'ARCoeffs', m.AR, ...
        'MACoeffs', m.MA, ...
        'Constant', m.const ...
    );
    
    % 2. Simulate realization
    y = armaObj.simulate(N, sigma);
    
    % 3. Compute empirical ACF and PACF
    empACF = OceanMath.TimeSeries.Diagnostics.autocorr(y, maxLag);
    empPACF = OceanMath.TimeSeries.Diagnostics.parcorr(y, maxLag);
    
    % 4. Compute theoretical ACF and PACF
    theoACF = armaObj.theoreticalACF(maxLag);
    theoPACF = armaObj.theoreticalPACF(maxLag);
    
    % Print summary stats
    fprintf('Model: %s\n', m.name);
    fprintf('  Equation:       %s\n', armaObj.getEquation());
    fprintf('  Stationary:     %s\n', logicalStr(armaObj.isStationary()));
    fprintf('  Invertible:     %s\n', logicalStr(armaObj.isInvertible()));
    fprintf('  Empirical Mean: %.4f (Expected: %.4f)\n', mean(y), m.const / (1 - sum(m.AR)));
    fprintf('  Empirical Var:  %.4f\n\n', var(y));
    
    % 5. Render dashboard
    fig = viz.createFigure(m.name);
    ax = axes(fig);
    
    plotData = struct();
    plotData.Y = y;
    plotData.ACF = empACF;
    plotData.PACF = empPACF;
    plotData.TheoreticalACF = theoACF;
    plotData.TheoreticalPACF = theoPACF;
    plotData.Lags = maxLag;
    plotData.ModelName = m.name;
    
    viz.plot(ax, 'TimeSeries', plotData, 'ShowLegend', true);
end

%% Part 2: Duality and Invertibility of MA(1) (From Textbook Section 3.7)
% Section 3.7 states that the processes:
%   Model A: Y_t = e_t + 0.5*e_{t-1}   (invertible)
%   Model B: Y_t = e_t + 2.0*e_{t-1}   (non-invertible)
% yield the exact same theoretical ACF, with rho_1 = 0.4 and rho_k = 0 for k > 1.

fprintf('=== PART 2: MA(1) DUALITY VERIFICATION ===\n\n');

modelA = OceanMath.TimeSeries.Models.ARMA('ARCoeffs', [], 'MACoeffs', 0.5);
modelB = OceanMath.TimeSeries.Models.ARMA('ARCoeffs', [], 'MACoeffs', 2.0);

% Compute theoretical ACFs
acfA = modelA.theoreticalACF(maxLag);
acfB = modelB.theoreticalACF(maxLag);

fprintf('Model A (theta = 0.5):\n');
fprintf('  Invertible: %s\n', logicalStr(modelA.isInvertible()));
fprintf('  Theoretical rho_1: %.4f\n', acfA(2));
fprintf('  Theoretical rho_2: %.4f\n\n', acfA(3));

fprintf('Model B (theta = 2.0):\n');
fprintf('  Invertible: %s\n', logicalStr(modelB.isInvertible()));
fprintf('  Theoretical rho_1: %.4f\n', acfB(2));
fprintf('  Theoretical rho_2: %.4f\n\n', acfB(3));

% Assert equivalence of ACF
maxDiff = max(abs(acfA - acfB));
fprintf('Maximum difference between theoretical ACFs: %.1e\n', maxDiff);
if maxDiff < 1e-12
    fprintf('SUCCESS: Models A and B share the exact same theoretical ACF!\n');
else
    fprintf('FAILURE: Autocorrelation functions do not match!\n');
end

% Plot comparison dashboard
figDual = viz.createFigure('MA(1) Duality comparison');
tiled = tiledlayout(figDual, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Simulate realizations for both
yA = modelA.simulate(N, sigma);
yB = modelB.simulate(N, sigma);

% Plot model A
axA1 = nexttile(tiled, [1, 2]);
viz.styleAxes(axA1);
hold(axA1, 'on');
plot(axA1, yA, 'Color', viz.Colors.Primary, 'LineWidth', 1.0, 'DisplayName', 'Model A (theta=0.5)');
plot(axA1, yB, 'Color', viz.Colors.Secondary, 'LineWidth', 1.0, 'DisplayName', 'Model B (theta=2.0)');
title(axA1, 'Realization Overlay', 'Color', viz.Colors.Text);
legend(axA1, 'TextColor', viz.Colors.Text, 'EdgeColor', 'none', 'Color', 'none');

axA2 = nexttile(tiled);
viz.styleAxes(axA2);
hold(axA2, 'on');
stem(axA2, 0:5, acfA(1:6), 'o', 'Color', viz.Colors.Primary, 'LineWidth', 2.0, 'DisplayName', 'Model A');
stem(axA2, 0:5, acfB(1:6), 'x--', 'Color', viz.Colors.Boundary, 'LineWidth', 1.0, 'DisplayName', 'Model B');
title(axA2, 'First 5 Lags ACF Comparison', 'Color', viz.Colors.Text);
legend(axA2, 'TextColor', viz.Colors.Text, 'EdgeColor', 'none', 'Color', 'none');
ylim(axA2, [-0.2, 1.1]);

% Compute PACFs (which differ because one is invertible and one is not!)
pacfA = modelA.theoreticalPACF(10);
pacfB = modelB.theoreticalPACF(10);

axA3 = nexttile(tiled);
viz.styleAxes(axA3);
hold(axA3, 'on');
plot(axA3, 1:10, pacfA, '-o', 'Color', viz.Colors.Primary, 'LineWidth', 1.5, 'DisplayName', 'Model A (theta=0.5)');
plot(axA3, 1:10, pacfB, '-x', 'Color', viz.Colors.Secondary, 'LineWidth', 1.5, 'DisplayName', 'Model B (theta=2.0)');
title(axA3, 'Theoretical PACF Comparison', 'Color', viz.Colors.Text);
legend(axA3, 'TextColor', viz.Colors.Text, 'EdgeColor', 'none', 'Color', 'none');
ylim(axA3, [-1.1, 1.1]);

fprintf('\n=== EXPERIMENT COMPLETE ===\n');

function str = logicalStr(val)
    if val
        str = 'True';
    else
        str = 'False';
    end
end
