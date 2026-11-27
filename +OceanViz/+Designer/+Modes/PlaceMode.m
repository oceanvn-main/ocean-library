classdef PlaceMode < OceanViz.Designer.Modes.BaseMode
    % PLACEMODE  Station placement interaction mode.
    %
    %   Left-clicking on empty canvas prompts for a station name and
    %   creates a new StationNode at the click position.
    %   Clicking an existing station simply selects it.
    %
    % See also: OceanViz.Designer.Modes.IDesignerMode

    methods
        function onAxesClick(obj, handler, clickPos, ~) %#ok<INUSL>
            td = handler.Designer_;

            name = sprintf('S%d', td.StationCount_ + 1);
            name = inputdlg('Station name:', 'New Station', 1, {name});
            if isempty(name), return; end

            cmd = OceanViz.Designer.Commands.AddStationCommand(td, name{1}, clickPos);
            if ~isempty(td.TimelineController_)
                td.TimelineController_.execute(cmd);
            else
                cmd.execute();
            end
            td.updateStatus_();
        end

        function onStationClick(~, ~, ~)
            % Place mode ignores station clicks — selection is handled
            % by the InteractionHandler before delegation.
        end

        function msg = getStatusText(obj, designer)
            suffix = obj.getLineSuffix(designer);
            msg = ['Mode: PLACE  |  Click on canvas to add a station', suffix];
        end

        function name = getName(~)
            name = 'place';
        end
    end
end
