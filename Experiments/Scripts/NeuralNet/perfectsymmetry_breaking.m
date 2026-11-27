% Objective Symmetry Breaking Experiment using Deep Learning Toolbox

% 1. Dynamically Generate Objective Training Data (Non-linear clusters)
num_samples = 400;
rng('shuffle'); % Ensure truly random generation each run

% Generate Class 0 (Center cluster)
X_class0 = randn(2, num_samples/2) * 0.5;
Y_class0 = zeros(1, num_samples/2);

% Generate Class 1 (Ring/Outer cluster)
angles = rand(1, num_samples/2) * 2 * pi;
radii = randn(1, num_samples/2) * 0.5 + 3;
X_class1 = [radii .* cos(angles); radii .* sin(angles)];
Y_class1 = ones(1, num_samples/2);

% Combine and convert to dlarray for Autograd tracking
% Format 'CB' means Channels (features) x Batch (observations)
X_train = dlarray([X_class0, X_class1], 'CB'); 
Y_train = dlarray([Y_class0, Y_class1], 'CB');

% 2. Define Objective Network Architecture
layers = [
    featureInputLayer(2, 'Name', 'input')
    fullyConnectedLayer(4, 'Name', 'hidden') % 4 hidden neurons
    sigmoidLayer('Name', 'sigmoid1')
    fullyConnectedLayer(1, 'Name', 'output')
    sigmoidLayer('Name', 'sigmoid2')
];

% Initialize Case B: Standard Toolbox Init (Glorot/Symmetry Broken)
net_rand = dlnetwork(layers);

% Initialize Case A: Forced Perfect Symmetry (Exact Zeros)
net_zero = net_rand;
learnables_zero = net_zero.Learnables;

% Objectively force all learnable weights and biases to 0.00 as dlarrays
for i = 1:height(learnables_zero)
    % Extract the size of the current parameter
    param_size = size(learnables_zero.Value{i});
    
    % Create exact zeros and explicitly cast to dlarray
    learnables_zero.Value{i} = dlarray(zeros(param_size));
end

% Reassign the strictly formatted table back to the network
net_zero.Learnables = learnables_zero;

% 3. Training Hyperparameters
epochs = 100;
learnRate = 0.5;
velocity_zero = [];
velocity_rand = [];

% Tracking arrays
dist_zero = zeros(epochs, 1);
dist_rand = zeros(epochs, 1);

% Dynamically find the index for the hidden layer weights to avoid hardcoding
hidden_idx = find(strcmp(net_zero.Learnables.Layer, 'hidden') & strcmp(net_zero.Learnables.Parameter, 'Weights'));

% 4. Objective Training Loop with Autograd
for epoch = 1:epochs
    
    % --- Train Case A (Zero Init) ---
    % dlfeval objectively evaluates the function and tracks gradients
    [loss_z, gradients_z] = dlfeval(@modelLoss, net_zero, X_train, Y_train);
    
    % Standard SGDM update (no manual math)
    [net_zero, velocity_zero] = sgdmupdate(net_zero, gradients_z, velocity_zero, learnRate, 0.9);
    
    % Measure objective distance between Hidden Neuron 1 and Neuron 2 weights
    % extractdata() detaches the values from the Autograd graph
    w_z = extractdata(net_zero.Learnables.Value{hidden_idx}); 
    dist_zero(epoch) = norm(w_z(1, :) - w_z(2, :));

    % --- Train Case B (Random Init) ---
    [loss_r, gradients_r] = dlfeval(@modelLoss, net_rand, X_train, Y_train);
    [net_rand, velocity_rand] = sgdmupdate(net_rand, gradients_r, velocity_rand, learnRate, 0.9);
    
    w_r = extractdata(net_rand.Learnables.Value{hidden_idx}); 
    dist_rand(epoch) = norm(w_r(1, :) - w_r(2, :));
    
end

% 5. Data Visualization (Dark Mode)
fig = figure('Color', [0.15 0.15 0.15], 'Name', 'Objective Symmetry Analysis');
ax = axes('Parent', fig, 'Color', [0.2 0.2 0.2], 'XColor', 'w', 'YColor', 'w', 'GridColor', [0.5 0.5 0.5]);
hold on; grid on;

plot(1:epochs, dist_zero, 'LineWidth', 2, 'Color', [0.9 0.4 0.4], 'DisplayName', 'Forced Zero Init (Paralyzed)');
plot(1:epochs, dist_rand, 'LineWidth', 2, 'Color', [0.3 0.8 0.9], 'DisplayName', 'Glorot Init (Learning)');

title('Objective Divergence of Neuron Weights Over Time', 'Color', 'w');
xlabel('Training Epoch', 'Color', 'w');
ylabel('Euclidean Distance ||Neuron_1 - Neuron_2||', 'Color', 'w');
legend('TextColor', 'w', 'Color', [0.25 0.25 0.25], 'Location', 'northwest');
hold off;

% --- Objective Loss Function ---
function [loss, gradients] = modelLoss(net, X, Y)
    % Forward pass through the dynamically defined network
    Y_pred = forward(net, X);
    % Calculate Mean Squared Error objectively
    loss = mse(Y_pred, Y);
    % Let Autograd handle the complex calculus dynamically
    gradients = dlgradient(loss, net.Learnables);
end