classdef DeleteMode < OceanViz.Designer.Modes.BaseMode
    % DELETEMODE  Station and edge deletion interaction mode.
    %
    %   Clicking near a station deletes the station.
    %   Clicking near a curve (but not a station) deletes the edge.
    %
    % See also: OceanViz.Designer.Modes.IDesignerMode

    methods
        function onAxesClick(~, handler, clickPos, ~)
            td = handler.Designer_;

            % Try station first, then edge
            nearId = handler.findNearestStation_(clickPos);
            if nearId > 0
                td.deleteStation_(nearId);
            else
                handler.deleteNearestEdge_(clickPos);
            end
        end

        function onStationClick(~, handler, nodeId)
            handler.Designer_.deleteStation_(nodeId);
        end

        function msg = getStatusText(~, ~)
            msg = 'Mode: DELETE  |  Click a station or near a curve to remove';
        end

        function name = getName(~)
            name = 'delete';
        end
    end
end
