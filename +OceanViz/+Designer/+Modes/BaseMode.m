classdef (Abstract) BaseMode < OceanViz.Designer.Modes.IDesignerMode
    % BASEMODE  Abstract base inheriting from IDesignerMode to provide default behavioral stubs.
    % 
    %   Prevents boilerplate by offering empty enter/exit handlers and a shared
    %   getLineSuffix() utility for the Status bar. Subclasses only override 
    %   what they strictly need.
    %
    % See also: OceanViz.Designer.Modes.IDesignerMode

    methods
        function enter(~, ~)
            % Base enter logic (No-op)
        end

        function exit(~, ~)
            % Base exit logic (No-op)
        end

        function onStationClick(~, ~, ~)
            % Base ignores station clicks by default
        end

        function suffix = getLineSuffix(~, designer)
            % Utility for appending the current Transit Line to status bars
            if ~isempty(designer.ActiveLineCode_)
                suffix = sprintf('  [Line: %s]', designer.ActiveLineCode_);
            else
                suffix = '';
            end
        end
    end
end
