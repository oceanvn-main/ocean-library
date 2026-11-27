function V_symbolic = solveAndAnimate(str1, str2, axis_str, filename, bound_min_str, bound_max_str)
    % solveAndAnimate - Solves and animates the volume of revolution
    %                   around ANY vertical (x=k) or horizontal (y=k) axis.
    %
    % Inputs:
    %   str1          - String definition of the first curve.
    %                   - If axis is vertical (e.g., 'x=1'), this is y_top (e.g., '4*x-x^2')
    %                   - If axis is horizontal (e.g., 'y=2'), this is x_right (e.g., '2')
    %   str2          - String definition of the second curve.
    %                   - If axis is vertical, this is y_bottom (e.g., '3')
    %                   - If axis is horizontal, this is x_left (e.g., '2*y^2')
    %   axis_str      - String of the axis of rotation (e.g., 'x=3', 'y=-2', 'x', 'y')
    %   filename      - Name of the output .avi file.
    %   bound_min_str - (Optional) Min integration boundary (e.g., '0')
    %   bound_max_str - (Optional) Max integration boundary (e.g., '2')
    
    % --- 0. Setup Symbolic Environment ---
    if ~license('test', 'Symbolic_Toolbox')
        error('This function requires the Symbolic Math Toolbox.');
    end
    
    fprintf('--- Solving Volume of Revolution about %s ---\n', axis_str);
    
    % --- 1. Determine Rotation Type (Vertical or Horizontal) ---
    if contains(axis_str, 'x')
        % Vertical Axis Rotation (e.g., 'x=3' or 'x')
        % We integrate w.r.t. x. str1=y_top, str2=y_bot.
        axis_type = 'vertical';
        syms x; var = x;
        ind_var_label = 'x';
        dep_var_label = 'y';
        
        y_top_sym = str2sym(str1);
        y_bot_sym = str2sym(str2);
        func1_sym = y_top_sym;
        func2_sym = y_bot_sym;
        
        % Parse axis k
        k_str = strrep(axis_str, 'x=', '');
        if strcmp(k_str, 'x'), k = 0; else, k = double(str2sym(k_str)); end
        
        r_sym = abs(x - k);
        h_sym = y_top_sym - y_bot_sym;
        
        fprintf('   Mode: Vertical Axis Rotation (Shells integrate dx)\n');
        fprintf('   y_top(x) = %s\n', char(y_top_sym));
        fprintf('   y_bot(x) = %s\n', char(y_bot_sym));

    elseif contains(axis_str, 'y')
        % Horizontal Axis Rotation (e.g., 'y=-2' or 'y')
        % We integrate w.r.t. y. str1=x_right, str2=x_left.
        axis_type = 'horizontal';
        syms y; var = y;
        ind_var_label = 'y';
        dep_var_label = 'x';
        
        x_right_sym = str2sym(str1);
        x_left_sym = str2sym(str2);
        func1_sym = x_right_sym;
        func2_sym = x_left_sym;

        % Parse axis k
        k_str = strrep(axis_str, 'y=', '');
        if strcmp(k_str, 'y'), k = 0; else, k = double(str2sym(k_str)); end
        
        r_sym = abs(y - k);
        h_sym = x_right_sym - x_left_sym;
        
        fprintf('   Mode: Horizontal Axis Rotation (Shells integrate dy)\n');
        fprintf('   x_right(y) = %s\n', char(x_right_sym));
        fprintf('   x_left(y)  = %s\n', char(x_left_sym));
    else
        error('Invalid axis_str. Must contain ''x'' or ''y''.');
    end
    
    % --- 2. Find Limits of Integration (c, d) ---
    if nargin < 6
        fprintf('\n1. Finding intersection points (limits of integration)...\n');
        sols = solve(func1_sym == func2_sym, var);
        intersects = double(sols);
        real_intersects = intersects(imag(intersects) == 0);
        
        if length(real_intersects) < 2
            error('Could not find two real intersection points. Try providing boundaries as arguments.');
        end
        c = min(real_intersects);
        d = max(real_intersects);
        fprintf('   Intersections found at %s = %f and %s = %f\n', ind_var_label, c, ind_var_label, d);
    else
        fprintf('\n1. Using provided boundaries...\n');
        try 
            c = double(str2sym(bound_min_str));
            d = double(str2sym(bound_max_str));
        catch
            error('Could not parse bound_min_str or bound_max_str.');
        end
        fprintf('   Limits set to %s = %f (c) and %s = %f (d)\n', ind_var_label, c, ind_var_label, d);
    end
    
    % --- 3. Verify Boundaries (Top/Bottom or Right/Left) ---
    test_val = (c + d) / 2;
    val1 = double(subs(func1_sym, var, test_val));
    val2 = double(subs(func2_sym, var, test_val));
    
    if val1 < val2
        fprintf('\n   Warning: str1 and str2 appear to be swapped. Correcting...\n');
        [func1_sym, func2_sym] = deal(func2_sym, func1_sym);
        h_sym = -h_sym; % Correct the height
    end
    
    % --- 4. Set up and Calculate Volume Integral ---
    fprintf('\n2. Calculating volume using Cylindrical Shells...\n');
    integrand = 2 * pi * r_sym * h_sym;
    
    fprintf('   Axis of Rotation: %s = %f\n', ind_var_label, k);
    fprintf('   Radius r(%s) = |%s - k| = %s\n', ind_var_label, ind_var_label, char(r_sym));
    fprintf('   Height h(%s) = %s\n', ind_var_label, char(h_sym));
    fprintf('   V = int( 2*pi * (%s) * (%s), %s, %f, %f )\n', char(r_sym), char(h_sym), ind_var_label, c, d);
    
    V_symbolic = int(integrand, var, c, d);
    V_numeric = double(V_symbolic);
    
    fprintf('\n   Symbolic Volume V = %s\n', char(V_symbolic));
    fprintf('   Numeric Volume  V = %f\n', V_numeric);
    
    % --- 5. Generate Movie ---
    fprintf('\n3. Generating animation...\n');
    
    f1 = matlabFunction(func1_sym, 'Vars', var);
    f2 = matlabFunction(func2_sym, 'Vars', var);
    
    symbolic_str = char(V_symbolic);
    if length(symbolic_str) > 50 % Set a threshold for "too long"
        % If it's too long, just use the numeric value (to 5 decimal places)
        volume_display_str = sprintf('V \x2248 %.5f', V_numeric); % \x2248 is the "approx" symbol
    else
        % Otherwise, show the nice symbolic form and the numeric form
        volume_display_str = sprintf('V = %s \x2248 %.5f', symbolic_str, V_numeric);
    end
    % --- END: Readability Fix ---
    
    createRevolutionMovie(f1, f2, c, d, axis_type, k, axis_str, filename, volume_display_str);
    fprintf('--- Solution Complete ---\n');
    
end