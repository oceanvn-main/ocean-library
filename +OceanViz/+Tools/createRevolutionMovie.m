function createRevolutionMovie(f1, f2, min_val, max_val, axis_type, k, axis_str, filename, volume_str)
    % createRevolutionMovie - Creates a movie of a region rotated
    %                         about ANY specified axis, optimized for dark mode.
    %
    % Inputs:
    %   f1         - Function handle for the "outer" curve (y_top or x_right)
    %   f2         - Function handle for the "inner" curve (y_bot or x_left)
    %   min_val    - Min integration boundary (c)
    %   max_val    - Max integration boundary (d)
    %   axis_type  - 'vertical' (for x=k) or 'horizontal' (for y=k)
    %   k          - The coordinate of the axis of rotation
    %   axis_str   - String of the axis of rotation (e.g., 'x=3', 'y=-2')
    %   filename   - The name of the output movie file
    %   volume_str - String representation of the calculated volume
    
    % --- Setup Video Writer ---
    writeObj = VideoWriter(filename);
    writeObj.FrameRate = 20;
    open(writeObj);
    
    % --- Setup Figure and Axes ---
    fig = figure;
    fig.Position = [100, 100, 800, 600];
    
    % --- DARK MODE ADJUSTMENTS: Figure and Axes background and text colors ---
    fig.Color = [0.1 0.1 0.1]; % Dark gray figure background
    ax = gca; % Get current axes handle
    ax.Color = [0.15 0.15 0.15]; % Slightly lighter dark gray axes background
    ax.XColor = [0.8 0.8 0.8]; % Light gray axis labels and ticks
    ax.YColor = [0.8 0.8 0.8];
    ax.ZColor = [0.8 0.8 0.8];
    ax.GridColor = [0.4 0.4 0.4]; % Lighter grid lines
    ax.MinorGridColor = [0.2 0.2 0.2]; % Even lighter minor grid
    ax.GridAlpha = 0.5; % Make grid slightly transparent
    ax.MinorGridAlpha = 0.3;
    ax.Title.Color = [0.9 0.9 0.9]; % White title text
    ax.FontSize = 10; % Slightly larger font size for readability
    ax.FontWeight = 'bold';
    % ------------------------------------------------------------------------
    
    n_frames = 90;
    theta_steps = linspace(0, 2*pi, n_frames);
    
    % --- Animation Loop ---
    for i = 1:n_frames
        theta_current = theta_steps(i);
        cla(ax); % Clear specific axes for new frame
        hold(ax, 'on'); % Hold on for the specific axes
        
        if strcmp(axis_type, 'vertical')
            % --- VERTICAL AXIS (x=k) LOGIC ---
            f_top = f1; f_bot = f2;
            xmin = min_val; xmax = max_val;
            x_range = linspace(xmin, xmax, 100);
            
            % Determine axis limits
            y_max_val = f_top(x_range);
            y_min_val = f_bot(x_range);
            if isscalar(y_max_val), y_max_val = y_max_val .* ones(size(x_range)); end
            if isscalar(y_min_val), y_min_val = y_min_val .* ones(size(x_range)); end
            y_max = max(y_max_val);
            y_min = min(y_min_val);

            max_radius = max(abs(x_range - k));
            axis_lim_x = [min(xmin, k) - max_radius, max(xmax, k) + max_radius];
            axis_lim_y = [min(0, y_min), max(y_max, 0)];
            if axis_lim_y(1) == axis_lim_y(2), axis_lim_y(1) = axis_lim_y(1) - 1; end
            axis_lim_z = [-max_radius, max_radius];
            
            % 1. Plot surfaces
            theta_vec = linspace(0, theta_current, 40);
            [x_mesh, t_mesh] = meshgrid(x_range, theta_vec);
            
            r_mesh = x_mesh - k;
            X = k + r_mesh .* cos(t_mesh);
            Z = r_mesh .* sin(t_mesh);

            Y_top_val = f_top(x_mesh);
            if isscalar(Y_top_val)
                Y_top = Y_top_val .* ones(size(x_mesh));
            else
                Y_top = Y_top_val;
            end
            % Surface colors: Cyan for top, light blue for bottom
            surf(ax, X, Y_top, Z, 'FaceColor', 'cyan', 'EdgeColor', 'none', 'FaceAlpha', 0.7);
            
            Y_bot_val = f_bot(x_mesh);
            if isscalar(Y_bot_val)
                Y_bot = Y_bot_val .* ones(size(x_mesh));
            else
                Y_bot = Y_bot_val;
            end
            surf(ax, X, Y_bot, Z, 'FaceColor', [0.6 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.6); % Slightly darker light blue
            
            % 2. Plot 2D region (using brighter variants of red/blue if desired)
            r_reg = x_range - k;
            X_reg = k + r_reg .* cos(theta_current);
            Z_reg = r_reg .* sin(theta_current);
            
            Y_reg_top = f_top(x_range);
            if isscalar(Y_reg_top)
                Y_reg_top = Y_reg_top .* ones(size(x_range));
            end
            plot3(ax, X_reg, Y_reg_top, Z_reg, 'Color', [1 0.4 0.4], 'LineWidth', 3); % Brighter red
            
            Y_reg_bot = f_bot(x_range);
            if isscalar(Y_reg_bot)
                Y_reg_bot = Y_reg_bot .* ones(size(x_range));
            end
            plot3(ax, X_reg, Y_reg_bot, Z_reg, 'Color', [0.4 0.4 1], 'LineWidth', 3); % Brighter blue
            
            % Plot connecting edges (white/light gray for dark mode)
            r_min = xmin - k;
            X_edge_min = k + r_min * cos(theta_current);
            Z_edge_min = r_min * sin(theta_current);
            plot3(ax, [X_edge_min X_edge_min], [f_bot(xmin), f_top(xmin)], [Z_edge_min Z_edge_min], 'Color', [0.9 0.9 0.9], 'LineWidth', 2); % Light gray
            
            r_max = xmax - k;
            X_edge_max = k + r_max * cos(theta_current);
            Z_edge_max = r_max * sin(theta_current);
            plot3(ax, [X_edge_max X_edge_max], [f_bot(xmax), f_top(xmax)], [Z_edge_max Z_edge_max], 'Color', [0.9 0.9 0.9], 'LineWidth', 2); % Light gray

            % 3. Plot axis of rotation (brighter green)
            plot3(ax, [k k], axis_lim_y, [0 0], 'Color', [0.2 1 0.2], 'LineStyle', '--', 'LineWidth', 2); % Brighter green
            
            % 4. Set Figure Properties
            axis(ax, 'equal');
            axis(ax, [axis_lim_x, axis_lim_y, axis_lim_z]);
            
        elseif strcmp(axis_type, 'horizontal')
            % --- HORIZONTAL AXIS (y=k) LOGIC ---
            f_right = f1; f_left = f2;
            ymin = min_val; ymax = max_val;
            y_range = linspace(ymin, ymax, 100);
            
            % Determine axis limits
            x_max_val = f_right(y_range);
            x_min_val = f_left(y_range);
            if isscalar(x_max_val), x_max_val = x_max_val .* ones(size(y_range)); end
            if isscalar(x_min_val), x_min_val = x_min_val .* ones(size(y_range)); end
            x_max = max(x_max_val);
            x_min = min(x_min_val);
            
            max_radius = max(abs(y_range - k));
            axis_lim_x = [min(x_min, 0), max(x_max, 0)];
            if axis_lim_x(1) == axis_lim_x(2), axis_lim_x(1) = axis_lim_x(1) - 1; end
            axis_lim_y = [min(ymin, k) - max_radius, max(ymax, k) + max_radius];
            axis_lim_z = [-max_radius, max_radius];
            
            % 1. Plot surfaces
            theta_vec = linspace(0, theta_current, 40);
            [y_mesh, t_mesh] = meshgrid(y_range, theta_vec);
            
            r_mesh = y_mesh - k;
            Y = k + r_mesh .* cos(t_mesh);
            Z = r_mesh .* sin(t_mesh);

            X_right_val = f_right(y_mesh);
            if isscalar(X_right_val), X_right = X_right_val .* ones(size(y_mesh)); else, X_right = X_right_val; end
            surf(ax, X_right, Y, Z, 'FaceColor', 'cyan', 'EdgeColor', 'none', 'FaceAlpha', .7);
            
            X_left_val = f_left(y_mesh);
            if isscalar(X_left_val), X_left = X_left_val .* ones(size(y_mesh)); else, X_left = X_left_val; end
            surf(ax, X_left, Y, Z, 'FaceColor', [0.6 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.6); % Slightly darker light blue
            
            % 2. Plot 2D region
            r_reg = y_range - k;
            Y_reg = k + r_reg .* cos(theta_current);
            Z_reg = r_reg .* sin(theta_current);

            X_reg_right = f_right(y_range);
            if isscalar(X_reg_right), X_reg_right = X_reg_right .* ones(size(y_range)); end
            plot3(ax, X_reg_right, Y_reg, Z_reg, 'Color', [1 0.4 0.4], 'LineWidth', 3); % Brighter red
            
            X_reg_left = f_left(y_range);
            if isscalar(X_reg_left), X_reg_left = X_reg_left .* ones(size(y_range)); end
            plot3(ax, X_reg_left, Y_reg, Z_reg, 'Color', [0.4 0.4 1], 'LineWidth', 3); % Brighter blue
            
            % Plot connecting edges
            r_min = ymin - k;
            Y_edge_min = k + r_min * cos(theta_current);
            Z_edge_min = r_min * sin(theta_current);
            plot3(ax, [f_left(ymin), f_right(ymin)], [Y_edge_min Y_edge_min], [Z_edge_min Z_edge_min], 'Color', [0.9 0.9 0.9], 'LineWidth', 2); % Light gray
            
            r_max = ymax - k;
            Y_edge_max = k + r_max * cos(theta_current);
            Z_edge_max = r_max * sin(theta_current);
            plot3(ax, [f_left(ymax), f_right(ymax)], [Y_edge_max Y_edge_max], [Z_edge_max Z_edge_max], 'Color', [0.9 0.9 0.9], 'LineWidth', 2); % Light gray
            
            % 3. Plot axis of rotation
            plot3(ax, axis_lim_x, [k k], [0 0], 'Color', [0.2 1 0.2], 'LineStyle', '--', 'LineWidth', 2); % Brighter green
            
            % 4. Set Figure Properties
            axis(ax, 'equal');
            axis(ax, [axis_lim_x, axis_lim_y, axis_lim_z]);
        end
        
        % --- Common Figure Properties ---
        xlabel(ax, 'X-axis'); ylabel(ax, 'Y-axis'); zlabel(ax, 'Z-axis'); % Specify axes for labels
        title_text = sprintf('Solid of Revolution (Axis: %s) | Volume = %s\nAngle = %.0f°', ...
                             axis_str, volume_str, rad2deg(theta_current));
        title(ax, title_text); % Specify axes for title
        view(ax, 120, 25); % Specify axes for view
        grid(ax, 'on'); % Specify axes for grid
        
        % Light source (adjust to ensure good visibility with dark background)
        % Using the axis handle for light properties
        camlight(ax, 'headlight'); % Position light relative to camera
        lighting(ax, 'gouraud');
        
        % --- Capture Frame ---
        drawnow;
        frame = getframe(fig);
        writeVideo(writeObj, frame);
    end
    
    close(writeObj);
    % close(fig); % Commented out as in user's request
    disp(['Movie saved as ' filename]);
end