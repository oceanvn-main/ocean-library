%% Recipe: Using Online Accumulators (`+OceanEngine/+Accumulators`)
% This recipe demonstrates how to use the generic online accumulators
% for sample averaging, weighted importance sampling, and Welford variance.

clear; clc;

%% 1. Weighted Importance Sampling Accumulator
% Create a 2D lookup table accumulator [100 states x 2 actions]
q_acc = OceanEngine.Accumulators.WeightedMeanAccumulator([100 2]);

% Simulate streaming updates for state 15, action 1
q_acc.update(10, 0.5, 15, 1);
q_acc.update(20, 1.5, 15, 1);

Q_table = q_acc.getValue();
C_table = q_acc.getWeight();

fprintf('Q(15, 1) = %.2f (expected 17.50)\n', Q_table(15, 1));
fprintf('C(15, 1) = %.2f (expected 2.00)\n', C_table(15, 1));

%% 2. Welford Online Variance Accumulator
% Track mean and variance of a streaming Gaussian signal
welford_acc = OceanEngine.Accumulators.WelfordVarianceAccumulator([1 1]);
rng(42);
data = randn(1, 1000) * 2.5 + 5.0; % mean = 5.0, std = 2.5, var = 6.25

for k = 1:numel(data)
    welford_acc.update(data(k));
end

fprintf('\nOnline Mean: %.4f (sample mean: %.4f)\n', welford_acc.getValue(), mean(data));
fprintf('Online Var:  %.4f (sample var:  %.4f)\n', welford_acc.getVariance('Type', 'sample'), var(data));
fprintf('Online Std:  %.4f (sample std:  %.4f)\n', welford_acc.getStd('Type', 'sample'), std(data));

%% 3. Exponential Moving Average Accumulator
ema_acc = OceanEngine.Accumulators.AccumulatorFactory.create('EMA', [1 1], 'Alpha', 0.1);
ema_acc.update(100);
ema_acc.update(200);
fprintf('\nEMA Value: %.2f\n', ema_acc.getValue());

%% 4. 3-Pillar Diagram & Prototype Introspection (Canvas Palette)
proto = OceanEngine.Accumulators.AccumulatorFactory.getPrototypeSpec('Weighted', [10 2]);
fprintf('\nStrategy: %s\n', proto.Strategy);
fprintf('Symbolic Formula: %s\n', proto.Symbolic.Formula);
fprintf('Diagram Symbol:   %s\n', proto.Diagram.DiagramSymbol.LabelText);

