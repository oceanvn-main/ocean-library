classdef (Abstract) IDesignerMode < handle
    % IDESIGNERMODE  Abstract interface for TrackDesigner interaction modes.
    %
    %   Every interaction mode (Place, Connect, Edit, Delete, Insert)
    %   implements this interface. The InteractionHandler delegates all
    %   mode-specific logic to the currently active IDesignerMode object,
    %   eliminating switch/case blocks entirely (State Pattern).
    %
    % See also: OceanViz.Designer.Modes.BaseMode,
    %           OceanViz.Designer.Interaction.InteractionHandler

    methods (Abstract)
        % ONAXESCLICK  Handle a left-click on the canvas.
        %   handler  — InteractionHandler instance (provides utilities)
        %   clickPos — 3D intersection point on the working plane
        %   rawPoint — raw 2×3 CurrentPoint from the Axes (for laser beam)
        onAxesClick(obj, handler, clickPos, rawPoint)

        % ONSTATIONCLICK  Handle a direct click on a station marker.
        %   handler — InteractionHandler instance
        %   nodeId  — integer ID of the clicked station
        onStationClick(obj, handler, nodeId)

        % ENTER  Called when this mode becomes active.
        enter(obj, handler)

        % EXIT  Called when this mode is about to be replaced.
        exit(obj, handler)

        % GETSTATUSTEXT  Return the status bar message string.
        %   designer — TrackDesigner instance (for ActiveLineCode_, etc.)
        msg = getStatusText(obj, designer)

        % GETNAME  Return the mode's toolbar key ('place','connect', etc.)
        name = getName(obj)
    end
end
