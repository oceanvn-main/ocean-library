function explore_curves()
    % EXPLORE_CURVES - Interactive GUI for Problem 47
    % Equations: x = t^2, y = t^3 - c*t
    % Features: Dark mode, Real-time slider tuning.

    % --- 1. Data Setup ---
    t = linspace(-3, 3, 2000); % Parameter t
    initial_c = 2;             % Starting value for c

    % --- 2. Create Dark Mode Figure ---
    fig = figure('Name', 'Problem 47: Interactive Explorer', ...
                 'NumberTitle', 'off', ...
                 'Color', [0.15 0.15 0.15], ... % Dark Grey Background
                 'Position', [100, 100, 800, 600]);

    % --- 3. Setup Axes ---
    ax = axes('Parent', fig, ...
              'Color', [0 0 0], ...        % Black Plot Background
              'XColor', [1 1 1], ...       % White X-axis
              'YColor', [1 1 1], ...       % White Y-axis
              'GridColor', [1 1 1], ...
              'GridAlpha', 0.3, ...
              'LineWidth', 1.2, ...
              'FontSize', 10, ...
              'Position', [0.1, 0.25, 0.8, 0.65]); % Leave space at bottom for slider
    hold(ax, 'on'); grid(ax, 'on');
    axis(ax, 'equal');
    
    % Set fixed limits so the view doesn't jump around
    xlim(ax, [-1, 9]); 
    ylim(ax, [-10, 10]);
    
    % Add title and labels
    title(ax, 'Parametric Curve: x = t^2, y = t^3 - c \cdot t', ...
          'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel(ax, 'x', 'Color', 'w');
    ylabel(ax, 'y', 'Color', 'w');

    % --- 4. Initialize Plot Objects ---
    % The main curve
    hCurve = plot(ax, NaN, NaN, 'Color', [0 1 1], 'LineWidth', 2); 
    
    % A marker for the loop intersection point (c, 0)
    hPoint = plot(ax, NaN, NaN, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

    % --- 5. UI Controls (Slider & Text) ---
    
    % Label to display current C value
    lblValue = uicontrol('Parent', fig, ...
                         'Style', 'text', ...
                         'String', sprintf('c = %.2f', initial_c), ...
                         'Units', 'normalized', ...
                         'Position', [0.3, 0.12, 0.4, 0.05], ...
                         'BackgroundColor', [0.15 0.15 0.15], ...
                         'ForegroundColor', 'w', ...
                         'FontSize', 14);

    % The Slider
    slider = uicontrol('Parent', fig, ...
                       'Style', 'slider', ...
                       'Min', -3, ...
                       'Max', 8, ...
                       'Value', initial_c, ...
                       'Units', 'normalized', ...
                       'Position', [0.2, 0.05, 0.6, 0.05], ...
                       'BackgroundColor', [0.3 0.3 0.3]);

    % --- 6. Link Slider to Update Function ---
    % Use a listener to update continuously while dragging
    addlistener(slider, 'Value', 'PostSet', @(src, event) updateGraph());

    % Trigger the first update manually
    updateGraph();

    % --- 7. The Update Function ---
    function updateGraph()
        % Get current value from slider
        c = get(slider, 'Value');
        
        % Update text label
        desc = getShapeDescription(c);
        set(lblValue, 'String', sprintf('c = %.2f  %s', c, desc));
        
        % Recalculate coordinates
        x = t.^2;
        y = t.^3 - c .* t;
        
        % Update the curve data (efficiently, without re-plotting)
        set(hCurve, 'XData', x, 'YData', y);
        
        % Update the Loop Point
        if c > 0
            % If c > 0, the curve crosses itself at x = c, y = 0
            set(hPoint, 'XData', c, 'YData', 0, 'Visible', 'on');
            
            % Change color to Magenta when loop exists
            set(hCurve, 'Color', [1 0 1]); 
        else
            % Hide point if no loop
            set(hPoint, 'Visible', 'off');
            
            % Change color to Cyan when no loop
            set(hCurve, 'Color', [0 1 1]); 
        end
    end

    % Helper to describe the shape
    function str = getShapeDescription(val)
        if abs(val) < 0.1
            str = '(Cusp at Origin)';
        elseif val < 0
            str = '(Smooth)';
        else
            str = '(Loop)';
        end
    end
end