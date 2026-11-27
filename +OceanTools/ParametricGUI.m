function parametric_gui(funX, funY, t_range, c_limits, title_str, videoFilename)
%PARAMETRIC_GUI_V3 Professional Explorer with Video Recording
%   Modified version: Preserves V2 layout/features but adds video export.
%   Inputs:
%       funX, funY, t_range, c_limits, title_str (Standard)
%       videoFilename (Optional) - String (e.g., 'output.mp4'). 
%                                  If provided, auto-records a sweep on startup.

    % --- 0. Handle Optional Input ---
    if nargin < 6
        videoFilename = []; % Default: No recording, just interactive mode
    end

    % --- 1. State Management ---
    c_min = c_limits(1);
    c_max = c_limits(2);
    c_val = c_limits(3);
    is_playing = false;
    animation_direction = 1; % 1 for up, -1 for down
    
    % --- 2. Setup Modern Dark Figure ---
    fig = figure('Name', 'Parametric Lab Pro', ...
                 'NumberTitle', 'off', ...
                 'Color', [0.12 0.12 0.12], ... % Deep Charcoal
                 'Position', [100, 100, 1000, 600], ...
                 'Resize', 'on');
    
    % --- 3. Layout: Plot Area (Left) & Control Panel (Right) ---
    % Plot Panel
    panelPlot = uipanel('Parent', fig, ...
                        'Position', [0 0 0.75 1], ...
                        'BackgroundColor', [0.12 0.12 0.12], ...
                        'BorderType', 'none');
                    
    % Control Panel (Side Bar)
    panelControls = uipanel('Parent', fig, ...
                            'Position', [0.75 0 0.25 1], ...
                            'BackgroundColor', [0.18 0.18 0.18], ... % Slightly lighter
                            'BorderType', 'none');
    
    % --- 4. Plot Axes Setup ---
    ax = axes('Parent', panelPlot, ...
              'Color', [0.08 0.08 0.08], ... % Almost black
              'XColor', [0.8 0.8 0.8], ...
              'YColor', [0.8 0.8 0.8], ...
              'GridColor', [1 1 1], ...
              'GridAlpha', 0.15, ...
              'LineWidth', 1.2, ...
              'FontSize', 10, ...
              'Position', [0.1, 0.1, 0.85, 0.8]); 
    hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');
    
    % Initialize Plot
    x_init = funX(t_range, c_val);
    y_init = funY(t_range, c_val);
    hCurve = plot(ax, x_init, y_init, 'Color', [0 1 1], 'LineWidth', 2);
    
    % Reference Lines
    xline(ax, 0, 'Color', [0.3 0.3 0.3], 'LineWidth', 1);
    yline(ax, 0, 'Color', [0.3 0.3 0.3], 'LineWidth', 1);
    
    % Title & Labels
    title(ax, title_str, 'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
    xlabel(ax, 'x(t)', 'Color', [0.7 0.7 0.7]); 
    ylabel(ax, 'y(t)', 'Color', [0.7 0.7 0.7]);
    
    % Auto-scale Axes
    update_axes_limits();
    
    % --- 5. UI Controls (Side Panel) ---
    
    % Title
    uicontrol('Parent', panelControls, 'Style', 'text', 'String', 'CONTROLS', ...
              'Units', 'normalized', 'Position', [0.05 0.92 0.9 0.05], ...
              'BackgroundColor', [0.18 0.18 0.18], 'ForegroundColor', 'w', ...
              'FontSize', 14, 'FontWeight', 'bold');
          
    % -- Parameter Value Section --
    uicontrol('Parent', panelControls, 'Style', 'text', 'String', 'Current Value (c)', ...
              'Units', 'normalized', 'Position', [0.05 0.85 0.9 0.03], ...
              'BackgroundColor', [0.18 0.18 0.18], 'ForegroundColor', [0.8 0.8 0.8], ...
              'FontSize', 10, 'HorizontalAlignment', 'left');
          
    % Edit Box for Precision Input
    editValue = uicontrol('Parent', panelControls, 'Style', 'edit', ...
                          'String', num2str(c_val), ...
                          'Units', 'normalized', 'Position', [0.05 0.80 0.9 0.05], ...
                          'BackgroundColor', [0.25 0.25 0.25], 'ForegroundColor', 'w', ...
                          'FontSize', 12, 'Callback', @onEditValue);
    
    % Slider
    slider = uicontrol('Parent', panelControls, 'Style', 'slider', ...
                       'Min', c_min, 'Max', c_max, 'Value', c_val, ...
                       'Units', 'normalized', 'Position', [0.05 0.74 0.9 0.04], ...
                       'BackgroundColor', [0.18 0.18 0.18]);
    addlistener(slider, 'Value', 'PostSet', @onSliderMove);
    
    % -- Range Settings Section --
    uicontrol('Parent', panelControls, 'Style', 'text', 'String', 'Parameter Range', ...
              'Units', 'normalized', 'Position', [0.05 0.65 0.9 0.03], ...
              'BackgroundColor', [0.18 0.18 0.18], 'ForegroundColor', [0.8 0.8 0.8], ...
              'FontSize', 10, 'HorizontalAlignment', 'left');
          
    % Min/Max Inputs
    uicontrol('Parent', panelControls, 'Style', 'text', 'String', 'Min:', ...
              'Units', 'normalized', 'Position', [0.05 0.60 0.15 0.03], ...
              'BackgroundColor', [0.18 0.18 0.18], 'ForegroundColor', 'w');
    editMin = uicontrol('Parent', panelControls, 'Style', 'edit', 'String', num2str(c_min), ...
                        'Units', 'normalized', 'Position', [0.2 0.59 0.25 0.05], ...
                        'BackgroundColor', [0.25 0.25 0.25], 'ForegroundColor', 'w', ...
                        'Callback', @onUpdateRange);
                    
    uicontrol('Parent', panelControls, 'Style', 'text', 'String', 'Max:', ...
              'Units', 'normalized', 'Position', [0.5 0.60 0.15 0.03], ...
              'BackgroundColor', [0.18 0.18 0.18], 'ForegroundColor', 'w');
    editMax = uicontrol('Parent', panelControls, 'Style', 'edit', 'String', num2str(c_max), ...
                        'Units', 'normalized', 'Position', [0.65 0.59 0.25 0.05], ...
                        'BackgroundColor', [0.25 0.25 0.25], 'ForegroundColor', 'w', ...
                        'Callback', @onUpdateRange);
                    
    % -- Animation Section --
    btnPlay = uicontrol('Parent', panelControls, 'Style', 'pushbutton', ...
                        'String', '▶ Play Animation', ...
                        'Units', 'normalized', 'Position', [0.05 0.45 0.9 0.08], ...
                        'BackgroundColor', [0 0.6 0.5], 'ForegroundColor', 'w', ...
                        'FontSize', 12, 'FontWeight', 'bold', ...
                        'Callback', @toggleAnimation);
    uicontrol('Parent', panelControls, 'Style', 'text', 'String', 'Click Play to oscillate parameter c', ...
              'Units', 'normalized', 'Position', [0.05 0.40 0.9 0.04], ...
              'BackgroundColor', [0.18 0.18 0.18], 'ForegroundColor', [0.5 0.5 0.5], ...
              'FontSize', 8);

    % [NEW] Status Bar for Recording
    lblStatus = uicontrol('Parent', panelControls, 'Style', 'text', 'String', 'Ready', ...
                          'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.05], ...
                          'BackgroundColor', [0.18 0.18 0.18], 'ForegroundColor', 'y', ...
                          'FontSize', 10, 'FontAngle', 'italic');
    
    % --- 6. Animation Timer ---
    animTimer = timer('ExecutionMode', 'fixedRate', 'Period', 0.05, ...
                      'TimerFcn', @animateStep);
    
    % --- 7. AUTO-RECORD LOGIC ---
    if ~isempty(videoFilename)
        record_video_sequence(videoFilename);
    end

    % --- 8. Callback Functions ---
    function updateGraph(new_c)
        c_val = new_c;
        
        % Update UI Elements
        set(slider, 'Value', c_val);
        set(editValue, 'String', sprintf('%.4f', c_val));
        
        % Update Plot Data
        x_new = funX(t_range, c_val);
        y_new = funY(t_range, c_val);
        set(hCurve, 'XData', x_new, 'YData', y_new);
        
        % Dynamic Color Logic (Optional visual flair)
        if c_val < 0
            set(hCurve, 'Color', [0 1 1]); % Cyan
        else
            set(hCurve, 'Color', [1 0.2 0.6]); % Pink/Magenta
        end
    end

    function onSliderMove(~, ~)
        val = get(slider, 'Value');
        updateGraph(val);
    end

    function onEditValue(src, ~)
        val = str2double(get(src, 'String'));
        % Clamp value to limits
        if val < c_min, val = c_min; end
        if val > c_max, val = c_max; end
        updateGraph(val);
    end

    function onUpdateRange(~, ~)
        % Get new limits
        new_min = str2double(get(editMin, 'String'));
        new_max = str2double(get(editMax, 'String'));
        
        if new_min >= new_max
            % Revert if invalid
            set(editMin, 'String', num2str(c_min));
            set(editMax, 'String', num2str(c_max));
            return;
        end
        
        c_min = new_min;
        c_max = new_max;
        
        % Update Slider Limits
        set(slider, 'Min', c_min, 'Max', c_max);
        
        % Update Axes to accommodate new range potential
        update_axes_limits();
    end

    function toggleAnimation(src, ~)
        if is_playing
            stop(animTimer);
            set(src, 'String', '▶ Play Animation', 'BackgroundColor', [0 0.6 0.5]);
            is_playing = false;
        else
            start(animTimer);
            set(src, 'String', '■ Stop Animation', 'BackgroundColor', [0.8 0.2 0.2]);
            is_playing = true;
        end
    end

    function animateStep(~, ~)
        step = (c_max - c_min) / 100; % 1% movement per frame
        new_val = c_val + (step * animation_direction);
        
        % Bounce logic
        if new_val >= c_max
            new_val = c_max;
            animation_direction = -1;
        elseif new_val <= c_min
            new_val = c_min;
            animation_direction = 1;
        end
        
        updateGraph(new_val);
    end

    function update_axes_limits()
        % Estimate bounds based on min/max c
        x_all = [funX(t_range, c_min), funX(t_range, c_max)];
        y_all = [funY(t_range, c_min), funY(t_range, c_max)];
        margin = max(range(x_all), range(y_all)) * 0.1;
        
        xlim(ax, [min(x_all)-margin, max(x_all)+margin]);
        ylim(ax, [min(y_all)-margin, max(y_all)+margin]);
    end

    % [NEW] Nested Function to Handle Video Recording
    function record_video_sequence(fname)
        try
            set(lblStatus, 'String', 'RECORDING...', 'ForegroundColor', 'r');
            pause(0.5); % Brief pause to ensure UI is ready
            
            % Setup Writer
            v = VideoWriter(fname, 'MPEG-4');
            v.FrameRate = 30;
            open(v);
            
            % Generate smooth steps from Min to Max
            c_steps = linspace(c_min, c_max, 150);
            
            for k = 1:length(c_steps)
                updateGraph(c_steps(k));
                drawnow;
                
                % Capture Frame
                frame = getframe(fig);
                writeVideo(v, frame);
            end
            
            close(v);
            set(lblStatus, 'String', ['Saved: ' fname], 'ForegroundColor', 'g');
            fprintf('Video saved successfully: %s\n', fname);
            
        catch ME
            set(lblStatus, 'String', 'Error Saving', 'ForegroundColor', 'r');
            fprintf('Error recording video: %s\n', ME.message);
        end
    end

    % Cleanup on close
    set(fig, 'CloseRequestFcn', @(~,~) cleanup());
    function cleanup()
        stop(animTimer);
        delete(animTimer);
        delete(fig);
    end
end