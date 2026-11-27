% EXPERIMENT: Spectral Analysis — Identifying Hidden Cycles in Time Series
%
% PURPOSE:
%   This script verifies that time-domain autocovariances (and autocorrelations) of 
%   a cyclic series can look like messy, noisy oscillations, whereas the frequency-domain 
%   power spectrum concentrates cyclical behavior, making hidden periodicities (like 
%   the 11-year solar cycle) immediately obvious as a massive spike on a graph.
%
% DATASET:
%   Built-in MATLAB 'sunspot.dat' containing:
%     - Column 1: Years (1700 - 1987)
%     - Column 2: Annual average sunspot numbers
%
% ASSUMPTIONS:
%   1. The series is sampled annually with a sampling interval of T = 1 year,
%      so the sampling frequency Fs = 1 cycle/year.
%   2. The periodic behavior of the sunspot counts is stable over time but 
%      subject to stochastic noise.
%   3. Centering the data (subtracting the sample mean) removes the DC component
%      (0 Hz frequency), allowing cleaner visualization of the periodic cycles.
%
% REPRODUCIBILITY:
%   No random simulation is needed since the script uses the deterministic 
%   real-world sunspot historical dataset.
%
% See also: OceanMath.TimeSeries.Diagnostics.autocorr, OceanViz.Core

clear;
clc;
close all;

fprintf('=== STARTING SPECTRAL VS TIME-DOMAIN EXPERIMENT ===\n\n');

% 1. Load the built-in dataset
fprintf('Loading built-in sunspot dataset...\n');
load sunspot.dat;
years = sunspot(:, 1);
counts = sunspot(:, 2);

N = length(counts);
Fs = 1; % Sampling rate: 1 sample per year

fprintf('Dataset loaded: %d annual observations from %d to %d.\n\n', ...
    N, min(years), max(years));

% 2. Preprocess: Center the series (subtract the mean)
% Centering removes the massive spike at 0 Hz (DC offset) in the frequency spectrum.
countsMean = mean(counts);
countsCentered = counts - countsMean;

% 3. Time-Domain Autocovariance and Autocorrelation (ACF)
fprintf('Computing time-domain autocorrelation function (ACF)...\n');
maxLag = 50; % Lag up to 50 years to show multi-cycle oscillations
acf = OceanMath.TimeSeries.Diagnostics.autocorr(countsCentered, maxLag);

% Compute raw autocovariance
autocov = zeros(maxLag + 1, 1);
for k = 0:maxLag
    autocov(k+1) = sum(countsCentered(1:N-k) .* countsCentered(k+1:end)) / N;
end

% 4. Frequency-Domain Power Spectral Density (Periodogram)
fprintf('Computing frequency-domain power spectrum via Fast Fourier Transform (FFT)...\n');

% Compute FFT of centered data
fftOut = fft(countsCentered);

% Power spectrum (magnitude squared, scaled by N)
Pxx = (abs(fftOut).^2) / N;

% Since the signal is real, the spectrum is symmetric. We take the first half.
numUniquePts = ceil((N + 1) / 2);
Pxx_half = Pxx(1:numUniquePts);

% Scale the spectrum to conserve total variance (excluding DC and Nyquist if even)
if mod(N, 2) == 0
    Pxx_half(2:end-1) = 2 * Pxx_half(2:end-1);
else
    Pxx_half(2:end) = 2 * Pxx_half(2:end);
end

% Frequency vector (cycles per year)
freqs = (0:numUniquePts-1) * (Fs / N);

% 5. Identify the dominant frequency and cycle length
[peakPower, peakIdx] = max(Pxx_half(2:end)); % ignore DC (0 Hz) at index 1
peakIdx = peakIdx + 1; % adjust index for offset

dominantFreq = freqs(peakIdx);
dominantPeriod = 1 / dominantFreq;

fprintf('\n=== EXPERIMENT ANALYSIS RESULTS ===\n');
fprintf('Dominant Frequency: %.4f cycles/year\n', dominantFreq);
fprintf('Dominant Periodicity: %.2f years (solar cycle)\n', dominantPeriod);
fprintf('Spectral Peak Power: %.2f\n\n', peakPower);

% 6. Render Premium Dark-Themed Layout using OceanViz
fprintf('Generating visualization plots...\n');
viz = OceanViz.Core('Theme', 'Dark');
colors = viz.Colors;

fig = viz.createFigure('Spectral vs Time-Domain Cycle Identification');
layout = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Centered Sunspot Time Series
ax1 = nexttile(layout);
viz.styleAxes(ax1);
hold(ax1, 'on');
plot(ax1, years, countsCentered, 'Color', colors.Primary, 'LineWidth', 1.2, 'DisplayName', 'Centered Sunspots');
title(ax1, 'Time Series: Centered Sunspot Data (Oscillations with Noise)', 'Color', colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
xlabel(ax1, 'Year', 'Color', colors.Text);
ylabel(ax1, 'Centered Count', 'Color', colors.Text);
hold(ax1, 'off');

% Panel 2: Autocovariance (Time-Domain)
ax2 = nexttile(layout);
viz.styleAxes(ax2);
hold(ax2, 'on');
stem(ax2, 0:maxLag, autocov, 'Filled', 'Color', colors.Boundary, 'MarkerFaceColor', colors.Boundary, 'MarkerSize', 4, 'LineWidth', 1.2);
plot(ax2, 0:maxLag, autocov, 'Color', colors.Boundary, 'LineStyle', ':', 'LineWidth', 1.0, 'HandleVisibility', 'off');
title(ax2, 'Time-Domain: Autocovariances (Noisy, Messy Decaying Oscillations)', 'Color', colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
xlabel(ax2, 'Lag (Years)', 'Color', colors.Text);
ylabel(ax2, 'Autocovariance', 'Color', colors.Text);
hold(ax2, 'off');

% Panel 3: Power Spectrum (Frequency-Domain)
ax3 = nexttile(layout);
viz.styleAxes(ax3);
hold(ax3, 'on');

% Plot spectrum line
plot(ax3, freqs, Pxx_half, 'Color', colors.Secondary, 'LineWidth', 1.5, 'DisplayName', 'Power Spectrum');

% Highlight the dominant periodicity peak with a stem and text label
stem(ax3, dominantFreq, peakPower, 'Filled', 'Color', colors.Boundary, ...
    'MarkerFaceColor', colors.Boundary, 'MarkerSize', 6, 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Peak (Period = %.2f yrs)', dominantPeriod));

% Label the peak explicitly without LaTeX
text(ax3, dominantFreq + 0.01, peakPower * 0.9, ...
     sprintf('Peak: %.2f Year Cycle (Freq: %.3f)', dominantPeriod, dominantFreq), ...
     'Color', colors.Text, 'FontSize', 10, 'FontWeight', 'bold');

title(ax3, 'Frequency-Domain: Power Spectrum (Concentrated Peak of Periodic Cycle)', 'Color', colors.Text, 'FontSize', 11, 'FontWeight', 'bold');
xlabel(ax3, 'Frequency (Cycles / Year)', 'Color', colors.Text);
ylabel(ax3, 'Power Spectral Density', 'Color', colors.Text);
legend(ax3, 'Location', 'northeast', 'TextColor', colors.Text, 'EdgeColor', 'none', 'Color', 'none');
hold(ax3, 'off');

% 7. Save the visualization figure
fprintf('Saving visualization figure...\n');
if ~exist('Results', 'dir')
    mkdir('Results');
end
try
    viz.export(fig, 'Results/exp_hidden_cycles_spectral.png');
    fprintf('Figure successfully saved to Results/exp_hidden_cycles_spectral.png\n');
catch ME
    saveas(fig, 'Results/exp_hidden_cycles_spectral.png');
    fprintf('Figure saved using saveas fallback to Results/exp_hidden_cycles_spectral.png. Error: %s\n', ME.message);
end

fprintf('=== EXPERIMENT SCRIPT COMPLETED ===\n');

