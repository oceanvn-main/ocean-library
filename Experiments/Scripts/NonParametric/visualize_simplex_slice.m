% Monte Carlo Simulation: Geometric Marginalization on a 3D Simplex
% Optimized for dark mode visibility and vectorized performance.

function visualize_simplex_slice()
    % 1. Initialization
    N = 10000; % Number of random darts to throw
    n = 3;     % Dimensions (p1, p2, p3)
    
    % 2. Generate perfectly uniform points on the simplex
    % We use the exponential distribution trick to avoid center-clustering bias
    X = -log(rand(N, n)); 
    P = X ./ sum(X, 2); % Normalize so every row sums strictly to 1.0
    
    % 3. Define our target cross-section
    target_p1 = 0.2;
    tolerance = 0.01; % The "thickness" of our geometric slice
    
    % 4. Logical Filtering (Identify the dart hits)
    slice_mask = abs(P(:,1) - target_p1) < tolerance;
    
    % Split the data into the specific slice vs the rest of the pyramid
    P_slice = P(slice_mask, :);
    P_base  = P(~slice_mask, :);
    
    % 5. Visualization (Dark Mode Optimized)
    fig = figure('Name', 'Simplex Cross-Section', 'Color', [0.12 0.12 0.12]);
    hold on; grid on;
    
    % Plot the base simplex points (Dim Cyan, slight transparency)
    scatter3(P_base(:,1), P_base(:,2), P_base(:,3), 8, [0 0.8 0.8], 'filled', ...
        'MarkerFaceAlpha', 0.15);
    
    % Plot the isolated cross-section (Bright Neon Red/Orange, fully opaque)
    scatter3(P_slice(:,1), P_slice(:,2), P_slice(:,3), 30, [1 0.3 0.1], 'filled', ...
        'MarkerEdgeColor', [1 1 1], 'LineWidth', 0.5);
    
    % 6. Formatting the Dark Theme Axes
    ax = gca;
    ax.Color = [0.15 0.15 0.15];       % Dark axis background
    ax.XColor = [0.8 0.8 0.8];         % Light gray axis lines
    ax.YColor = [0.8 0.8 0.8];
    ax.ZColor = [0.8 0.8 0.8];
    ax.GridColor = [0.5 0.5 0.5];      % Visible grid
    ax.GridAlpha = 0.3;
    
    % Set limits and strict viewing angle to see the 2D plane in 3D space
    xlim([0 1]); ylim([0 1]); zlim([0 1]);
    view(135, 30); % Rotated to clearly see the triangle flat on
    
    % Labels and Title
    xlabel('p_1 (Our Target Variable)', 'Color', 'w', 'FontWeight', 'bold');
    ylabel('p_2', 'Color', 'w', 'FontWeight', 'bold');
    zlabel('p_3', 'Color', 'w', 'FontWeight', 'bold');
    title(sprintf('Simplex Cross Section at p_1 \\approx %.1f', target_p1), ...
        'Color', 'w', 'FontSize', 14);
    
    % Legend
    lgd = legend('All Valid Probabilities', 'The Cross-Section Slice');
    lgd.TextColor = 'w';
    lgd.Color = [0.2 0.2 0.2];
    
    hold off;
end