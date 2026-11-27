function plotParametricSurfaceWithGrid(fx, fy, fz, u_range, v_range)
% PLOTPARAMETRICSURFACEWITHGRID Plots a surface and highlights grid curves.
%   Distinguishes between u-constant and v-constant curves for identification.

    % 1. Setup Domains
    grid_density = 60; % High density for the smooth surface background
    u = linspace(u_range(1), u_range(2), grid_density);
    v = linspace(v_range(1), v_range(2), grid_density);
    [U, V] = meshgrid(u, v);

    % 2. Calculate Surface Coordinates
    X = fx(U, V);
    Y = fy(U, V);
    Z = fz(U, V);

    % 3. Setup Dark Mode Figure
    figure('Color', 'k');
    hold on;

    % 4. Plot the Base Surface (Faint/Transparent)
    % We make it very transparent (Alpha 0.3) so the grid lines pop out
    surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'FaceColor', 'interp');

    % ---------------------------------------------------------
    % 5. HIGHLIGHT GRID CURVES
    % ---------------------------------------------------------
    num_lines = 8; % How many grid lines to highlight in each direction

    % A. Plot u = constant curves (Varying v) -> CYAN LINES
    % These lines show the direction of v
    u_fixed_vals = linspace(u_range(1), u_range(2), num_lines);
    v_dense = linspace(v_range(1), v_range(2), 100); % Smooth lines

    for k = 1:length(u_fixed_vals)
        u_val = u_fixed_vals(k);
        % Create vectors where u is fixed number, v varies
        U_line = u_val * ones(size(v_dense));
        V_line = v_dense;

        % Plot the line
        plot3(fx(U_line, V_line), fy(U_line, V_line), fz(U_line, V_line), ...
            'Color', 'c', 'LineWidth', 2); % Cyan
    end

    % B. Plot v = constant curves (Varying u) -> MAGENTA LINES
    % These lines show the direction of u
    v_fixed_vals = linspace(v_range(1), v_range(2), num_lines);
    u_dense = linspace(u_range(1), u_range(2), 100);

    for k = 1:length(v_fixed_vals)
        v_val = v_fixed_vals(k);
        % Create vectors where v is fixed number, u varies
        U_line = u_dense;
        V_line = v_val * ones(size(u_dense));

        % Plot the line
        plot3(fx(U_line, V_line), fy(U_line, V_line), fz(U_line, V_line), ...
            'Color', 'm', 'LineWidth', 2); % Magenta
    end

    % ---------------------------------------------------------
    % 6. Aesthetics & Legend
    % ---------------------------------------------------------
    colormap(gray); % Use gray for surface so colors stand out
    shading interp;
    light('Position',[1 1 1],'Style','infinite');
    lighting gouraud;
    view(3); axis equal; grid on;

    % Axis Styling
    ax = gca;
    ax.Color = 'k';
    ax.XColor = 'w'; ax.YColor = 'w'; ax.ZColor = 'w';

    % Labels and Legend
    xlabel('X', 'Color', 'w'); ylabel('Y', 'Color', 'w'); zlabel('Z', 'Color', 'w');
    
    % Create dummy lines for legend to ensure correct labeling
    h1 = plot3(nan, nan, nan, 'Color', 'c', 'LineWidth', 2);
    h2 = plot3(nan, nan, nan, 'Color', 'm', 'LineWidth', 2);
    legend([h1, h2], {'u = constant (v-curves)', 'v = constant (u-curves)'}, ...
           'TextColor', 'w', 'Color', 'none', 'Location', 'best');
           
    rotate3d on;
    hold off;
end