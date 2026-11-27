% Exercise 11.5: Single Hidden Layer Neural Network with Weight Decay
clear; clc; close all;

% --- Part (b): Data Generation ---
rng(42); % For reproducibility

% True parameters
a1 = [3; 3];
a2 = [3; -3];

% Generate Training Data (100 observations)
N_train = 100;
X_train = randn(N_train, 2);
Z_train = randn(N_train, 1);
% Y = sigmoid(X*a1) + (X*a2)^2 + 0.30*Z
Y_train = 1./(1 + exp(-X_train*a1)) + (X_train*a2).^2 + 0.30 * Z_train;

% Generate Test Data (1000 observations)
N_test = 1000;
X_test = randn(N_test, 2);
Z_test = randn(N_test, 1);
Y_test = 1./(1 + exp(-X_test*a1)) + (X_test*a2).^2 + 0.30 * Z_test;

% Combine for MATLAB's divider to track test error natively per epoch
X_all = [X_train; X_test]'; % Transpose for neural network (features x samples)
Y_all = [Y_train; Y_test]';

train_indices = 1:N_train;
test_indices = (N_train + 1):(N_train + N_test);


% --- Part (a) & (b): Back-propagation with Weight Decay ---
% We will test multiple weight decay values to observe overfitting
weight_decays = [0, 0.01, 0.1, 0.5]; 
num_decays = length(weight_decays);

fig1 = figure('Name', 'Learning Curves vs Weight Decay', 'Position', [100, 100, 1000, 800]);
set(fig1, 'Color', [0.15 0.15 0.15]);

for i = 1:num_decays
    % Create a single hidden layer feedforward network with 10 units
    % 'trainscg' (Scaled Conjugate Gradient) is an efficient backprop algorithm
    net = fitnet(10, 'trainscg'); 
    net.trainParam.showWindow = false; % Run silently
    net.trainParam.epochs = 500;       % Ensure enough epochs to see curves
    
    % Implement Weight Decay via Regularization parameter
    % 0 = pure MSE, >0 incorporates Mean Squared Weights into the cost function
    net.performParam.regularization = weight_decays(i); 
    
    % Explicitly assign our train and test sets to track performance
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = train_indices;
    net.divideParam.valInd = []; % Skip validation to stick strictly to train/test
    net.divideParam.testInd = test_indices;
    
    % Train the network
    [~, tr] = train(net, X_all, Y_all);
    
    % Plotting the Learning Curves
    ax = subplot(2, 2, i);
    % Filter out empty epoch padding that MATLAB sometimes adds
    valid_epochs = 1:tr.num_epochs+1; 
    
    semilogy(tr.epoch(valid_epochs), tr.perf(valid_epochs), 'LineWidth', 2, 'Color', [0.2 0.8 0.8]); hold on;
    semilogy(tr.epoch(valid_epochs), tr.tperf(valid_epochs), 'LineWidth', 2, 'Color', [0.9 0.4 0.4]);
    
    title(sprintf('Weight Decay = %.2f', weight_decays(i)), 'Color', 'w');
    xlabel('Epochs', 'Color', 'w');
    ylabel('Mean Squared Error (Log Scale)', 'Color', 'w');
    legend('Train Error', 'Test Error', 'TextColor', 'w', 'Location', 'northeast');
    configureDarkModeAxes(ax);
end


% --- Part (c): Varying the Number of Hidden Units ---
hidden_units_range = 1:10;
test_errors_units = zeros(length(hidden_units_range), 1);
train_errors_units = zeros(length(hidden_units_range), 1);

% Fix weight decay to a modest value that prevented severe overfitting above
fixed_decay = 0.05; 

for h = hidden_units_range
    net = fitnet(h, 'trainscg');
    net.trainParam.showWindow = false;
    net.trainParam.epochs = 500;
    net.performParam.regularization = fixed_decay;
    
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = train_indices;
    net.divideParam.valInd = [];
    net.divideParam.testInd = test_indices;
    
    [~, tr] = train(net, X_all, Y_all);
    
    % Capture the final empirical error
    train_errors_units(h) = tr.perf(end);
    test_errors_units(h)  = tr.tperf(end);
end

fig2 = figure('Name', 'Performance vs Hidden Units', 'Position', [200, 200, 600, 400]);
set(fig2, 'Color', [0.15 0.15 0.15]);
ax2 = axes;
plot(hidden_units_range, train_errors_units, '-o', 'LineWidth', 2, 'Color', [0.2 0.8 0.8], 'MarkerFaceColor', [0.2 0.8 0.8]); hold on;
plot(hidden_units_range, test_errors_units, '-s', 'LineWidth', 2, 'Color', [0.9 0.4 0.4], 'MarkerFaceColor', [0.9 0.4 0.4]);
title('Effect of Hidden Units on Model Error', 'Color', 'w');
xlabel('Number of Hidden Units', 'Color', 'w');
ylabel('Final MSE', 'Color', 'w');
legend('Train Error', 'Test Error', 'TextColor', 'w');
configureDarkModeAxes(ax2);


% --- Dark Mode Helper Function ---
function configureDarkModeAxes(ax)
    set(ax, 'Color', [0.1 0.1 0.1], ...
            'XColor', [0.8 0.8 0.8], ...
            'YColor', [0.8 0.8 0.8], ...
            'GridColor', [0.3 0.3 0.3], ...
            'GridAlpha', 0.5);
    grid on;
end