% EXPERIMENT_9_4: Empirical verification of backfitting upward convergence.
% This script constructs a constrained smoothing matrix, executes the
% backfitting algorithm, and plots the RSS trajectory against the analytical limit.

function verify_exercise_9_4()
    % --- Parameters ---
    N = 100;           % Dimension of the vector space
    max_iter = 15;     % Number of backfitting cycles
    
    % --- Step 1: Construct symmetric S with eigenvalues in [0, 1) ---
    % Generate a random orthogonal basis using QR decomposition
    [Q, ~] = qr(randn(N, N)); 
    
    % Generate random eigenvalues strictly between 0 and 0.99
    eigenvalues = rand(N, 1) * 0.99; 
    Lambda = diag(eigenvalues);
    
    % Reconstruct the symmetric smoothing matrix
    S = Q * Lambda * Q';
    
    % --- Step 2: Initialize Data and Variables ---
    y = randn(N, 1);
    f1 = zeros(N, 1);
    f2 = zeros(N, 1);
    
    I = eye(N);
    rss_history = zeros(max_iter, 1);
    
    % --- Step 3: Execute Backfitting Loop ---
    for m = 1:max_iter
        % Sequential updates
        f1 = S * (y - f2);
        f2 = S * (y - f1);
        
        % Calculate and store the Residual Sum of Squares (RSS)
        r_current = y - (f1 + f2);
        rss_history(m) = sum(r_current.^2);
    end
    
    % --- Step 4: Calculate Analytical Limit ---
    % Solve (I + S) * r_star = (I - S) * y using optimized linear solver (\)
    rhs_vector = (I - S) * y;
    r_star = (I + S) \ rhs_vector; 
    rss_limit = sum(r_star.^2);
    
    % --- Step 5: Dark Mode Visualization ---
    figure('Name', 'Backfitting Convergence', 'Color', [0.12 0.12 0.12]);
    axes_handle = gca;
    
    % Plot the empirical RSS trajectory
    plot(1:max_iter, rss_history, '-o', ...
        'Color', [0.0, 0.8, 0.4], ...     % Bright spring green
        'LineWidth', 2, ...
        'MarkerFaceColor', [0.0, 0.8, 0.4], ...
        'MarkerSize', 6);
    hold on;
    
    % Plot the theoretical limit
    yline(rss_limit, '--', ...
        'Color', [1.0, 0.6, 0.2], ...     % Bright orange
        'LineWidth', 2, ...
        'Label', 'Theoretical Limit $r^*$', ...
        'LabelHorizontalAlignment', 'left', ...
        'Interpreter', 'latex');
    
    % Configure dark mode aesthetics
    set(axes_handle, ...
        'Color', [0.18 0.18 0.18], ...    % Dark grey background
        'XColor', [0.9 0.9 0.9], ...      % Off-white axes
        'YColor', [0.9 0.9 0.9], ...
        'GridColor', [0.5 0.5 0.5], ...
        'GridAlpha', 0.3, ...
        'FontSize', 11);
    grid on;
    
    title('Exercise 9.4: Upward Convergence of RSS', 'Color', 'w', 'FontSize', 14);
    xlabel('Iteration ($m$)', 'Interpreter', 'latex', 'Color', 'w', 'FontSize', 12);
    ylabel('Residual Sum of Squares ($\| \mathbf{r} \|^2$)', 'Interpreter', 'latex', 'Color', 'w', 'FontSize', 12);
    
    legend({'Empirical RSS', 'Analytical Limit'}, ...
        'TextColor', 'w', ...
        'Color', [0.2 0.2 0.2], ...
        'EdgeColor', 'none', ...
        'Location', 'southeast');
end