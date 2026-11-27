function exp_TreeDF_Ex95bc_20260228()
    % EXP_TREEDF_EX95BC_20260228
    % ESL Exercise 9.5 (b) and (c) - Degrees of Freedom of a Regression Tree
    %
    % THEORY (Ex 9.5):
    % The generic definition of degrees of freedom for a fitting operator
    % y -> y_hat is:
    %      df = sum_i cov(y_i, y_hat_i) / sigma^2
    %
    % For a regression tree with m terminal nodes, if the leaf assignments
    % were fixed, the tree would be a linear operator S onto an m-dimensional
    % subspace, and df = trace(S) = m.
    % However, because the tree structure adapts to the y values (searching
    % for optimal splits), the actual df > m.
    %
    % EXPERIMENT:
    % (b) Generate N=100 observations with 10 standard Gaussian predictors.
    %     Fix X for all simulations.
    % (c) Generate y ~ N(0, 1), independent of X.
    %     Fit regression trees with exact terminal nodes m = 1, 5, 10.
    %     Estimate df via Monte Carlo:
    %       cov(y_i, y_hat_i) approx (1/(B-1)) * sum_b (y_i^(b) - E[y_i]) * (y_hat_i^(b) - E[y_hat_i])
    %
    % Note: We use MATLAB's fitrtree with 'MaxNumSplits' = m-1 to guarantee
    % exactly m terminal nodes, as custom tree implementations without a
    % best-first growth queue cannot strictly enforce a leaf count.
    
    clc; close all;
    
    %% Setup Experiment Parameters
    N = 100;    % Number of observations
    P = 10;     % Number of predictors
    B = 50;     % Monte Carlo Simulations (increased from 10 for better estimate stability)
                % Ex 9.5 suggests 10, but 50 runs fast and gives much clearer results.
    
    target_nodes = [1, 5, 10];
    K = length(target_nodes);
    
    %% (b) Generate Fixed Predictors
    rng(42); % For reproducibility
    X_fixed = randn(N, P);
    
    %% (c) Monte Carlo Simulations
    fprintf('Running Monte Carlo simulations (B=%d)...\n', B);
    fprintf('m (leaves) | Theoretical df | Estimated df\n');
    fprintf('-------------------------------------------\n');
    
    df_estimated = zeros(K, 1);
    
    % Storage for simulations: 
    % We need to store y and y_hat for all N points across B simulations.
    % Y_all: N x B matrix
    Y_all = randn(N, B);  % y ~ N(0, 1) -> sigma^2 = 1
    
    for k = 1:K
        m = target_nodes(k);
        max_splits = m - 1;
        
        Yhat_all = zeros(N, B);
        
        for b = 1:B
            y_b = Y_all(:, b);
            
            % Fit tree with exactly m terminal nodes.
            % 'MinLeafSize', 1 prevents arbitrary stopping before m nodes.
            tree = fitrtree(X_fixed, y_b, ...
                'MaxNumSplits', max_splits, ...
                'MinLeafSize', 1, ...     
                'SplitCriterion', 'mse');
            
            Yhat_all(:, b) = predict(tree, X_fixed);
            
            % Optional: Assert tree size is correct (unless impossible data)
            % actual_leaves = sum(~tree.IsBranch);
            % if actual_leaves ~= m
            %    warning('Tree generated %d leaves instead of %d', actual_leaves, m);
            % end
        end
        
        % Calculate Covariance across B simulations for each point i
        % cov(y_i, y_hat_i)  for i = 1...N
        % Since sigma^2 = 1, sum of these is the df.
        
        cov_sum = 0;
        for i = 1:N
            % Extracts 1xB vector for point i
            y_i      = Y_all(i, :);
            y_hat_i  = Yhat_all(i, :);
            
            % Sample covariance formula for two vectors:
            c = cov(y_i, y_hat_i);
            cov_sum = cov_sum + c(1, 2); % off-diagonal is the cross-covariance
        end
        
        df_estimated(k) = cov_sum; % since sigma^2 = 1, df = cov_sum / 1
        
        fprintf('    %-6d |      %-9d |    %.2f\n', m, m, df_estimated(k));
    
    core = OceanViz.Core('Theme', 'Dark');
    C = core.Colors;
    
    %% (d) Visualization
    
    f = figure('Color', C.Background, 'Position', [200 200 800 600], 'Name', 'Tree Degrees of Freedom');
    ax = axes(f);
    set(ax, 'Color', C.Background, 'XColor', C.Text, 'YColor', C.Text, ...
        'FontSize', 12, 'GridColor', C.Grid, 'GridAlpha', 0.2);
    hold on; grid on;
    
    % Bar chart for Estimated DF
    bplot = bar(target_nodes, df_estimated, 0.6, 'FaceColor', [0.2 0.6 1.0], 'EdgeColor', 'none');
    try bplot.FaceAlpha = 0.8; catch; end
    
    % Line for Theoretical DF (df = m)
    % Draw a line y = x points
    x_line = [0, max(target_nodes)+2];
    y_line = x_line;
    pline = plot(x_line, y_line, '--', 'Color', [1.0 0.3 0.3], 'LineWidth', 2);
    
    % Formatting
    title('Degrees of Freedom of a Regression Tree (ESL Ex 9.5)', 'Color', C.Text, 'FontSize', 16);
    xlabel('Number of Terminal Nodes (m)', 'Color', C.Text);
    ylabel('Degrees of Freedom (df)', 'Color', C.Text);
    
    xticks(target_nodes);
    xlim([0, max(target_nodes)+2]);
    ylim([0, max(df_estimated)*1.2]);
    
    % Note text explaining Adaptive vs Fixed
    txt = sprintf(['<strong>Discussion (Part D):</strong>\n\n', ...
                   'Theoretical DF (fixed structure): df = m\n', ...
                   'Estimated DF (adaptive structure): df >> m\n\n', ...
                   'Because the tree actively searches the data\n', ...
                   'for optimal split variables and split points,\n', ...
                   'it consumes far more degrees of freedom\n', ...
                   'than a fixed m-dimensional projection.']);
               
    text(0.05, 0.75, txt, 'Units', 'normalized', 'Color', C.Text, ...
         'VerticalAlignment', 'top', 'FontSize', 12, 'BackgroundColor', [0.15 0.15 0.15], ...
         'Margin', 8, 'EdgeColor', [0.3 0.3 0.3]);
     
    legend([bplot, pline], {'Estimated df (Monte Carlo, Adaptive)', 'Theoretical df (Fixed Structure)'}, ...
           'Location', 'northwest', 'TextColor', C.Text, 'Color', C.Background, 'EdgeColor', [0.3 0.3 0.3]);
       
    hold off;
    
    % Save output image
    % exportgraphics(f, 'exp_TreeDF_Ex95bc.png', 'Resolution', 300);
end
