function plotParametricSurface(fx, fy, fz, u_range, v_range, grid_density)
% PLOTPARAMETRICSURFACE Plots a parametric surface in two styles side-by-side.
%
%   Inputs:
%       fx, fy, fz   : Function handles for x(u,v), y(u,v), z(u,v)
%       u_range      : Vector [u_min, u_max]
%       v_range      : Vector [v_min, v_max]
%       grid_density : (Optional) Mesh density (default: 60)

    %Set default density if not provided
    if nargin < 6
        grid_density = 60;
    end

    %Calculate Surface Data (High Density for smoothness)
    u = linspace(u_range(1), u_range(2), grid_density);
    v = linspace(v_range(1), v_range(2), grid_density);
    [U, V] = meshgrid(u, v);

    X = fx(U, V);
    Y = fy(U, V);
    Z = fz(U, V);

    %Setup Figure (Dark Mode)
    figure('Color', 'k', 'Position', [100, 100, 1200, 600]);
    t = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(t, 'Parametric Surface Visualization & Grid Analysis', 'Color', 'w', 'FontSize', 14);

    %Standard Plotting
    ax1 = nexttile;
    surf(X, Y, Z, 'EdgeColor', 'interp', 'FaceAlpha', 0.9);
    
    % Styling for Plot 1
    colormap(ax1, 'jet');     % High contrast for the shape
    shading interp;
    light('Position', [1 1 1], 'Style', 'infinite');
    %lighting gouraud;
    material shiny;
    % lighting phong;
    
    %Axis Setup
    setupDarkAxis(ax1, 'Standard Surface');
    ax2 = nexttile;
    hold on;
    %Base Surface (Faint/Transparent)
    surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.2, 'FaceColor', 'white');

    %Highlight Grid Curves ---
    num_lines = 8; % Number of grid lines to highlight
    
    %Plot u = constant (Varying v) -> CYAN LINES
    u_fixed_vals = linspace(u_range(1), u_range(2), num_lines);
    v_dense = linspace(v_range(1), v_range(2), 100); % Smooth lines for curves
    
    for k = 1:length(u_fixed_vals)
        u_val = u_fixed_vals(k);
        % Create vectors: u is fixed, v varies
        U_line = u_val * ones(size(v_dense));
        V_line = v_dense;
        plot3(fx(U_line, V_line), fy(U_line, V_line), fz(U_line, V_line), ...
            'Color', 'c', 'LineWidth', 1.5);
    end
    % B. Plot v = constant (Varying u) -> MAGENTA LINES
    v_fixed_vals = linspace(v_range(1), v_range(2), num_lines);
    u_dense = linspace(u_range(1), u_range(2), 100);
    
    for k = 1:length(v_fixed_vals)
        v_val = v_fixed_vals(k);
        % Create vectors: v is fixed, u varies
        U_line = u_dense;
        V_line = v_val * ones(size(u_dense));
        plot3(fx(U_line, V_line), fy(U_line, V_line), fz(U_line, V_line), ...
            'Color', 'm', 'LineWidth', 1.5);
    end
    % Styling for Plot 2
    colormap(ax2, gray);    % Gray base so colored lines pop
    shading interp;
    
    % Legend for Grid Lines
    h1 = plot3(nan, nan, nan, 'Color', 'c', 'LineWidth', 2);
    h2 = plot3(nan, nan, nan, 'Color', 'm', 'LineWidth', 2);
    legend([h1, h2], {'u = const (v-curves)', 'v = const (u-curves)'}, ...
           'TextColor', 'w', 'Color', 'none', 'Location', 'best');

    % Axis Setup
    setupDarkAxis(ax2, 'Grid Curve Analysis');
   
    hold off;
    
    % Enable rotation for both
    rotate3d on;
end

function setupDarkAxis(ax, titleText)
    % Helper function to apply dark mode styling to an axis
    set(ax, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
    grid(ax, 'on');
    axis(ax, 'equal');
    view(ax, 3);
    
    xlabel(ax, 'X', 'Color', 'w', 'FontWeight', 'bold');
    ylabel(ax, 'Y', 'Color', 'w', 'FontWeight', 'bold');
    zlabel(ax, 'Z', 'Color', 'w', 'FontWeight', 'bold');
    title(ax, titleText, 'Color', 'w', 'FontSize', 12);
    
    ax.GridColor = 'w';
    ax.GridAlpha = 0.3;
end