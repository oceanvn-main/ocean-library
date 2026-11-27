function setup_dark_axes(ax)
    % Apply dark theme to axes
    set(ax, 'Color', [0.15 0.15 0.18], ...   % Dark Gray Background
            'XColor', [0.9 0.9 0.9], ...     % Light Text
            'YColor', [0.9 0.9 0.9], ...
            'GridColor', [1 1 1], ...        % White Grid
            'GridAlpha', 0.1, ...
            'LineWidth', 1.2, ...
            'FontSize', 11);
    hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'tight');
end