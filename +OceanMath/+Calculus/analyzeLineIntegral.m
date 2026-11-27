function analyze_line_integral(F_func, r_func, t_range, varargin)
% ANALYZE_LINE_INTEGRAL Visualizes vector line integrals in 2D or 3D.
%
%   Inputs:
%       F_func  : Function handle for the Vector Field.
%                 Must accept (x,y) or (x,y,z) and return [u,v] or [u,v,w].
%       r_func  : Function handle for the Parametric Path.
%                 Must accept (t) and return [x; y] or [x; y; z].
%       t_range : [start, end] vector for parameter t.
%       options : (Optional) Name-Value pairs for 'Resolution', 'Dark', etc.
%
%   Examples:
%       % 2D Problem
%       F = @(x,y) [x.*y.^2; -x.^2];
%       r = @(t) [t.^3; t.^2];
%       analyze_line_integral(F, r, [0 1]);
%
%       % 3D Problem
%       F = @(x,y,z) [x+y.^2; x.*z; y+z];
%       r = @(t) [t.^2; t.^3; -2*t];
%       analyze_line_integral(F, r, [0 2]);
    
    %% 1. PARSE SETTINGS
    p = inputParser;
    addParameter(p, 'Resolution', 500);
    addParameter(p, 'GridDensity', 15);
    parse(p, varargin{:});
    res = p.Results.Resolution;
    
    %% 2. NUMERICAL ENGINE
    % A. Generate Parameter t
    t = linspace(t_range(1), t_range(2), res);
    dt = t(2) - t(1);

    %Evaluate path (r)
    %r_vals should be a matrix where rows are x,y,(z), and cols are time
    r_vals = r_func(t);
    [dims, ~] = size(r_vals);

    %Calculate Derivatives dr/dt (Numerically)
    % We use gradient for robustness so users don't have to derive manually
    dr_vals = gradient(r_vals, dt);

    %Evaluate Field F  along the path
    %We accept inputs as row, so we convert r_vals to cell for input
    if dims == 2
        x = r_vals(1,:); y = r_vals(2,:);
        F_vals = F_func(x, y); % Returns [Fx; Fy]
    elseif dims == 3
        x = r_vals(1,:); y = r_vals(2,:); z = r_vals(3,:);
        F_vals = F_func(x, y, z); % Returns [Fx; Fy; Fz]
    else
        error('Function only supports 2D or 3D paths.');
    end

    %Calculate Dot Product & Integral F⋅dr=Fx​dx+Fy​dy(+Fz​dz)
    % Dot product of columns: sum(A .* B)
    dot_product = sum(F_vals .* dr_vals, 1);
    integral_val = trapz(t, dot_product);

    %Visualization dispatcher
    if dims == 2
        plot_2d_case(x, y, dot_product, integral_val, F_func, p.Results.GridDensity);
    else
        plot_3d_case(x, y, z, t, F_vals, dot_product, integral_val, F_func);
    end
end

%Visualization

function plot_2d_case(x, y, dot_prod, integral_val, F_func, dense)
    setup_figure('2D Vector Field Analysis (Curtain View)');
    
    % 1. Create Grid
    margin = 0.1 * max(max(x)-min(x), max(y)-min(y));
    [X, Y] = meshgrid(linspace(min(x)-margin, max(x)+margin, dense), ...
                      linspace(min(y)-margin, max(y)+margin, dense));
    
    % 2. EXTRACT FIELD COMPONENTS (FIXED)
    FV = F_func(X,Y); 
    
    % Determine how the user's function stacked the output (Vertically vs Horizontally)
    sz = size(X);
    rows = sz(1);
    
    % Check if stacked vertically (most common for @(x,y) [u; v])
    if size(FV, 1) == 2 * rows
        U = FV(1:rows, :);
        V = FV(rows+1:end, :);
    else
        % Fallback for other structures, strictly reshaping
        % This assumes FV is a list of elements that matches numel(X)*2
        U = reshape(FV(1:numel(X)), sz);
        V = reshape(FV(numel(X)+1:end), sz);
    end

    hold on; grid on; axis equal; view(30, 30);
    
    % 3. Plot Vector Field on Floor (z=0)
    quiver3(X, Y, zeros(size(X)), U, V, zeros(size(X)), 0.6, ...
        'Color', [0 0.8 0.8], 'LineWidth', 0.5, 'DisplayName', 'Vector Field');
    
    % 4. Plot The Curtain
    Z_surf = [zeros(size(dot_prod)); dot_prod];
    X_surf = [x; x];
    Y_surf = [y; y];
    
    surf(X_surf, Y_surf, Z_surf, 'EdgeColor', 'none', ...
        'FaceAlpha', 0.8, 'DisplayName', 'Integral Curtain');
    colormap('cool');
    
    % 5. Decoration
    plot3(x, y, zeros(size(x)), 'y', 'LineWidth', 3, 'DisplayName', 'Path C');
    patch([min(X(:)) max(X(:)) max(X(:)) min(X(:))], ...
          [min(Y(:)) min(Y(:)) max(Y(:)) max(Y(:))], ...
          [0 0 0 0], [0.3 0.3 0.3], 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
          'HandleVisibility', 'off');
      
    title({'\bf\color{white}2D Line Integral Visualization', ...
           sprintf('\\color{yellow}Total Signed Volume \\approx %.4f', integral_val)}, ...
           'Interpreter', 'tex');
    zlabel('Dot Product Value');
    legend('Location', 'best', 'TextColor', 'w', 'EdgeColor', 'none', 'Color', 'none');
end

function plot_3d_case(x, y, z, t, F_vals_path, dot_prod, integral_val, F_func)
    setup_figure('3D Vector Field Analysis');
    ax1 = subplot(121);
    style_axis(ax1); hold on; grid on; axis equal; view(-30, 20);

    %Background Grid
    margin = 0.5;
    [X,Y,Z_grid] = meshgrid(...
        linspace(min(x)-margin, max(x)+margin, 5), ...
        linspace(min(y)-margin, max(y)+margin, 5), ...
        linspace(min(z)-margin, max(z)+margin, 5));

    %Extract Field components
    FV = F_func(X, Y, Z_grid);
    sz = size(X);
    rows = sz(1);

    % Robust splitting
    %Stack case
    if size(FV, 1) == 3 * rows
        U = FV(1:rows, :, :);
        V = FV(rows+1:2*rows, :, :);
        W = FV(2*rows+1:end, :, :);
    else %Horizontal list case
        num = numel(X);
        U = reshape(FV(1:num), sz);
        V = reshape(FV(num+1:2*num), sz);
        W = reshape(FV(2*num+1:end), sz);
    end
    mag = sqrt(U.^2 + V.^2 + W.^2);
    quiver3(X,Y,Z_grid, U./mag, V./mag, W./mag, 0.5, ...
        'Color', [0.2 0.2 0.2], 'DisplayName', 'Field Direction'); %Gray
    %Path Colored (Trick) - Surface with no width
    col_work = [dot_prod; dot_prod];
    surf([x; x], [y; y], [z; z], col_work, 'FaceColor', 'none', 'EdgeColor', 'interp', 'LineWidth', 6, 'DisplayName', 'Path (Color = Work)');
    colormap(ax1, 'jet');
    limit = max(abs(dot_prod));
    if limit == 0, limit = 1; end
    clim(ax1, [-limit, limit]);

    %Color bar
    cb = colorbar(ax1);
    cb.Color = 'w';
    cb.Label.String = 'Work Rate (F \cdot dr)';

    %Force Vectors
    idx = 1:40:length(t); %Sample some
    quiver3(x(idx), y(idx), z(idx), F_vals_path(1,idx), F_vals_path(2,idx), F_vals_path(3,idx), ...
        0.5, 'Color', [1 0 1], 'LineWidth', 1.5, 'DisplayName', 'Force F');
    
    title('\bf\color{white}Geometry (Color = Work Rate)', 'Interpreter', 'tex');
    xlabel('x'); ylabel('y'); zlabel('z');

    %Integral
  
    % --- PANEL 2: INTEGRAL ---
    ax2 = subplot(1, 2, 2);
    style_axis(ax2); hold on; grid on;
    
    % Split data into Positive (Thrust) and Negative (Drag) components
    pos_area = max(dot_prod, 0);
    neg_area = min(dot_prod, 0);
    
    % 1. Plot Positive Area (Cyan)
    area(t, pos_area, 'FaceColor', [0 1 1], 'FaceAlpha', 0.5, ...
        'EdgeColor', 'none', 'DisplayName', 'Positive Work');
    
    % 2. Plot Negative Area (Magenta)
    area(t, neg_area, 'FaceColor', [1 0 1], 'FaceAlpha', 0.5, ...
        'EdgeColor', 'none', 'DisplayName', 'Negative Work');
    
    % 3. Draw the white outline on top for crispness
    plot(t, dot_prod, 'w', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    
    % 4. Zero line
    yline(0, 'w--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    
    title({'\bf\color{white}Unrolled Integral', ...
           sprintf('\\color{yellow}Total Area \\approx %.4f', integral_val)}, ...
           'Interpreter', 'tex');
    xlabel('t'); ylabel('Work Rate (F \cdot dr)');
    
    % Add a legend to explain the colors
    legend('Location', 'northwest', 'TextColor', 'w', 'EdgeColor', 'none', 'Color', 'none');
    
    sgtitle('\bf\color{white}3D Line Integral Analysis', 'Interpreter', 'tex');

end

function setup_figure(name)
    figure('Name', name, 'Color', [0.1 0.1 0.1], ...
           'Position', [100 100 1200 600], 'NumberTitle', 'off');
end

function style_axis(ax)
    set(ax, 'Color', 'k', 'XColor', [0.7 0.7 0.7], ...
        'YColor', [0.7 0.7 0.7], 'ZColor', [0.7 0.7 0.7], ...
        'GridColor', [0.3 0.3 0.3]);
end