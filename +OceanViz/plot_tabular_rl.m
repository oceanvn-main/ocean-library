function fig = plot_tabular_rl(ValueMatrix, PolicyMatrix, opts)
    % PLOT_TABULAR_RL Visualizes Tabular RL Policy and Value functions using OceanViz.
    %
    % USAGE:
    %   fig = OceanViz.plot_tabular_rl(V, pi)
    %   fig = OceanViz.plot_tabular_rl(solver)
    %   fig = OceanViz.plot_tabular_rl(V, pi, 'Title', 'Ex 4.7 Policy Iteration')
    
    arguments
        ValueMatrix
        PolicyMatrix = []
        opts.Title (1,1) string = "Jack's Car Rental RL Solution"
        opts.Theme (1,1) string = "Dark"
        opts.FigureHandle = []
    end
    
    % Support passing solver instance directly
    if isa(ValueMatrix, 'OceanStats.RL.DP.PolicyIterationSolver') || isa(ValueMatrix, 'PolicyIterationSolver')
        solver = ValueMatrix;
        V_mat = solver.ValueMatrix;
        pi_mat = solver.PolicyMatrix;
    else
        V_mat = ValueMatrix;
        pi_mat = PolicyMatrix;
    end
    
    % Initialize OceanViz Core Engine
    viz = OceanViz.Core('Theme', opts.Theme);
    theme = viz.Colors;
    
    if isempty(opts.FigureHandle) || ~isvalid(opts.FigureHandle)
        fig = figure('Name', opts.Title, 'Color', theme.Background, 'Position', [100, 100, 1100, 480]);
    else
        fig = opts.FigureHandle;
        set(fig, 'Color', theme.Background);
    end
    
    % Subplot 1: 2D Policy Heatmap
    ax1 = subplot(1, 2, 1, 'Parent', fig);
    set(ax1, 'Color', theme.Background);
    
    grid_size = size(pi_mat, 1);
    cars_range = 0:(grid_size - 1);
    
    hold(ax1, 'on');
    imagesc(ax1, cars_range, cars_range, pi_mat');
    set(ax1, 'YDir', 'normal');
    
    % Decision boundary contours
    min_act = min(pi_mat(:));
    max_act = max(pi_mat(:));
    [C, hc] = contour(ax1, cars_range, cars_range, pi_mat', min_act:max_act, 'k-', 'LineWidth', 1.2);
    if ~isempty(C) && size(C, 1) == 2
        try
            clabel(ax1, C, hc, 'Color', [1 1 1], 'FontSize', 9, 'FontWeight', 'bold');
        catch
        end
    end
    
    colormap(ax1, jet);
    c1 = colorbar(ax1);
    c1.Label.String = 'Net Cars Moved (1 \rightarrow 2)';
    c1.Label.Color = theme.Text;
    c1.Color = theme.Text;
    c1.Ticks = min_act:max_act;
    
    grid(ax1, 'on');
    set(ax1, 'Color', theme.Background, 'XColor', theme.Text, 'YColor', theme.Text, 'GridColor', theme.Grid);
    xlabel(ax1, '# Cars at second location', 'Color', theme.Text, 'FontSize', 10);
    ylabel(ax1, '# Cars at first location', 'Color', theme.Text, 'FontSize', 10);
    title(ax1, sprintf('%s: Optimal Policy \\pi(s)', opts.Title), 'Color', theme.Text, 'FontSize', 11, 'FontWeight', 'bold');
    
    legend(ax1, 'Location', 'northeast', 'TextColor', theme.Text, 'Color', [theme.Background 0.8], 'EdgeColor', 'none');
    hold(ax1, 'off');
    
    % Subplot 2: 3D State Value Surface
    ax2 = subplot(1, 2, 2, 'Parent', fig);
    set(ax2, 'Color', theme.Background);
    
    [X, Y] = meshgrid(cars_range, cars_range);
    
    hold(ax2, 'on');
    s_val = surf(ax2, X, Y, V_mat', 'EdgeColor', [0.2 0.2 0.2], 'EdgeAlpha', 0.4, 'FaceAlpha', 0.95);
    s_val.DisplayName = 'State Value V(s)';
    
    colormap(ax2, turbo);
    c2 = colorbar(ax2);
    c2.Label.String = 'Expected Return V(s)';
    c2.Label.Color = theme.Text;
    c2.Color = theme.Text;
    
    view(ax2, [-35, 30]);
    grid(ax2, 'on');
    set(ax2, 'Color', theme.Background, 'XColor', theme.Text, 'YColor', theme.Text, 'ZColor', theme.Text, 'GridColor', theme.Grid);
    
    xlabel(ax2, '# Cars at second location', 'Color', theme.Text, 'FontSize', 10);
    ylabel(ax2, '# Cars at first location', 'Color', theme.Text, 'FontSize', 10);
    zlabel(ax2, 'Value V(s)', 'Color', theme.Text, 'FontSize', 10);
    title(ax2, sprintf('%s: State Value V(s)', opts.Title), 'Color', theme.Text, 'FontSize', 11, 'FontWeight', 'bold');
    
    legend(ax2, 'Location', 'northeast', 'TextColor', theme.Text, 'Color', [theme.Background 0.8], 'EdgeColor', 'none');
    hold(ax2, 'off');
end
