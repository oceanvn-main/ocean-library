classdef Exporter
    % EXPORTER Handles saving figures consistently.
    
    % === MUST BE STATIC ===
    methods (Static) 
        function save(fig, name)
            % SAVE Exports the figure to PNG (and optionally PDF)
            arguments
                fig (1,1) matlab.ui.Figure
                name (1,1) string
            end
            
            % 1. Clean up the Name
            if endsWith(name, ".png", 'IgnoreCase', true)
                name = extractBefore(name, strlength(name)-3);
            end
            
            % 2. Setup Export Settings
            set(fig, 'PaperPositionMode', 'auto'); 
            
            % 3. Save
            fprintf('Saving: %s.png... ', name);
            try
                exportgraphics(fig, name + ".png", 'Resolution', 300, 'BackgroundColor', 'current');
                fprintf('Done.\n');
            catch
                saveas(fig, name + ".png");
                fprintf('Done (Legacy Save).\n');
            end
        end
    end
end