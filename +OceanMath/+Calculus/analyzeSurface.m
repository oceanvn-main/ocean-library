function analyze_surface(f_sym, xy_bounds)
% ANALYZE_SURFACE Finds critical points and plots surface in Dark Mode.
%   v2.0 Fix: Uses symbolic 'solve' with constraints to ensure ALL roots are found.

    % 1. Setup Variables
    vars = symvar(f_sym);
    
    % Logic to ensure we always have x and y variables
    if isempty(vars)
        x = sym('x'); y = sym('y');
    elseif length(vars) == 1
        v1 = vars(1);
        if char(v1) == 'y', y = v1; x = sym('x'); 
        else, x = v1; y = sym('y'); end
    else
        x = vars(1); y = vars(2);
    end
    
    if nargin < 2, xy_bounds = [-3 3 -3 3]; end
    
    fprintf('---------------------------------------------------\n');
    fprintf('Analyzing: %s\n', char(f_sym));

    % 2. Derivatives
    grad_f = gradient(f_sym, [x, y]);
    hessian_f = hessian(f_sym, [x, y]);
    D_sym = det(hessian_f);
    fxx = hessian_f(1,1);
    
    % 3. Robust Solve
    % We pass the inequalities directly to 'solve'. This works for both
    % polynomials (Ex 5) and Trig functions (Ex 20) to find all specific roots.
    constraints = [x >= xy_bounds(1), x <= xy_bounds(2), ...
                   y >= xy_bounds(3), y <= xy_bounds(4)];
               
    try
        % 'Real', true ensures we don't get imaginary roots
        S = solve([grad_f(1)==0, grad_f(2)==0, constraints], [x, y], 'Real', true);
        
        % Handle structure output
        if isempty(S)
             cx = []; cy = [];
        else
             cx = double(S.(char(x))); 
             cy = double(S.(char(y)));
        end
        
    catch
        % Fallback for edge cases (like infinite solutions in Ex 17)
        fprintf('   Warning: Symbolic solve struggled. Trying numerical fallback...\n');
        cx = []; cy = [];
    end

    % 4. Classify and Store
    points_min = []; points_max = []; points_saddle = [];
    f_handle = matlabFunction(f_sym, 'Vars', [x, y]);
    
    if isempty(cx)
        fprintf('   No discrete critical points found in this region.\n');
    else
        fprintf('   Found %d critical point(s):\n', length(cx));
        for i = 1:length(cx)
            xi = cx(i); yi = cy(i);
            zi = f_handle(xi, yi);
            
            % Evaluate Second Derivative Test
            D_val = double(subs(D_sym, {x,y}, {xi,yi}));
            fxx_val = double(subs(fxx, {x,y}, {xi,yi}));
            
            % Classification Logic
            type = 'Inconclusive/Flat';
            if D_val > 1e-4 && fxx_val > 0
                type = 'Local Min';
                points_min = [points_min; xi, yi, zi];
            elseif D_val > 1e-4 && fxx_val < 0
                type = 'Local Max';
                points_max = [points_max; xi, yi, zi];
            elseif D_val < -1e-4
                type = 'Saddle';
                points_saddle = [points_saddle; xi, yi, zi];
            else
                points_saddle = [points_saddle; xi, yi, zi]; % Mark inconclusive as saddle color
            end
            fprintf('   (%6.2f, %6.2f) -> %s\n', xi, yi, type);
        end
    end

    % 5. Visualization (Dark Mode)
    figure('Color', [0.1 0.1 0.1], 'Position', [100, 100, 900, 600]);
    
    N = 60;
    x_vec = linspace(xy_bounds(1), xy_bounds(2), N);
    y_vec = linspace(xy_bounds(3), xy_bounds(4), N);
    [X, Y] = meshgrid(x_vec, y_vec);
    
    % Safe Z evaluation
    Z = arrayfun(@(xv, yv) double(subs(f_sym, {x,y}, {xv,yv})), X, Y);

    % Surface Plot
    surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.85); hold on;
    
    % Plot Markers
    ms = 30;
    if ~isempty(points_min), plot3(points_min(:,1), points_min(:,2), points_min(:,3), '.', 'Color', [1 0.2 0.2], 'MarkerSize', ms, 'DisplayName', 'Min'); end
    if ~isempty(points_max), plot3(points_max(:,1), points_max(:,2), points_max(:,3), '.', 'Color', [0.2 1 0.2], 'MarkerSize', ms, 'DisplayName', 'Max'); end
    if ~isempty(points_saddle), plot3(points_saddle(:,1), points_saddle(:,2), points_saddle(:,3), '.', 'Color', [1 0.2 1], 'MarkerSize', ms, 'DisplayName', 'Saddle'); end
    
    % Styling
    title(['Surface: ' char(f_sym)], 'Color', 'w', 'FontSize', 12, 'Interpreter', 'none');
    xlabel(char(x), 'Color', 'w'); ylabel(char(y), 'Color', 'w'); zlabel('Z', 'Color', 'w');
    set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w', 'GridAlpha', 0.2);
    grid on; shading interp; lighting phong; camlight left; view(3); colormap(parula);
    legend('show', 'Location', 'best', 'TextColor', 'w', 'Color', 'none', 'EdgeColor', 'none');
    hold off;
end