classdef ConnectMode < OceanViz.Designer.Modes.BaseMode
    % CONNECTMODE  Interaction mode for linking two stations with a curve.
    %
    %   Click the first station to select it as the source (highlighted yellow).
    %   Click a second station to automatically draw the transit curve between them.
    %   Clicking empty space cancels the selection.
    %
    % See also: OceanViz.Designer.Modes.IDesignerMode

    properties (Access = private)
        ConnectSource_ (1,1) double = 0  % ID of the first station clicked
    end

    methods
        function onAxesClick(obj, handler, clickPos, ~)
            td = handler.Designer_;

            % Check if we clicked remotely close to any station
            nearId = handler.findNearestStation_(clickPos);
            
            if nearId == 0
                % Clicked entirely empty space: cancel pending connection
                if obj.ConnectSource_ > 0
                    srcH = td.Renderer_.StationHandles_{obj.ConnectSource_};
                    set(srcH, 'MarkerEdgeColor', td.Renderer_.StationColor * 0.6, 'LineWidth', 1.5);
                    obj.ConnectSource_ = 0;
                    td.updateStatus_();
                end
                return;
            end

            obj.processStationClick_(td, nearId);
        end

        function onStationClick(obj, handler, nodeId)
            td = handler.Designer_;
            obj.processStationClick_(td, nodeId);
        end

        function exit(obj, handler)
            % Clean up any lingering highlight before we swap out of Connect Mode
            td = handler.Designer_;
            if obj.ConnectSource_ > 0 && ~isempty(td.Renderer_.StationHandles_) ...
                    && obj.ConnectSource_ <= numel(td.Renderer_.StationHandles_) ...
                    && ~isempty(td.Renderer_.StationHandles_{obj.ConnectSource_}) ...
                    && isvalid(td.Renderer_.StationHandles_{obj.ConnectSource_})
                srcH = td.Renderer_.StationHandles_{obj.ConnectSource_};
                set(srcH, 'MarkerEdgeColor', td.Renderer_.StationColor * 0.6, 'LineWidth', 1.5);
            end
            obj.ConnectSource_ = 0;
        end

        function msg = getStatusText(obj, designer)
            suffix = obj.getLineSuffix(designer);
            if obj.ConnectSource_ == 0
                msg = ['Mode: CONNECT  |  Click the FIRST station', suffix];
            else
                srcName = designer.Transit_.getStation(obj.ConnectSource_).Name;
                msg = sprintf('Mode: CONNECT  |  Click the SECOND station (from "%s")%s', srcName, suffix);
            end
        end

        function name = getName(~)
            name = 'connect';
        end
    end

    methods (Access = private)
        function processStationClick_(obj, td, nodeId)
            if obj.ConnectSource_ == 0
                obj.ConnectSource_ = nodeId;
                % Highlight selected station
                h = td.Renderer_.StationHandles_{nodeId};
                set(h, 'MarkerEdgeColor', [1 1 0], 'LineWidth', 2.5);
                td.updateStatus_();
            else
                if nodeId ~= obj.ConnectSource_
                    td.connectStations(obj.ConnectSource_, nodeId);
                end
                % Unhighlight source station
                srcH = td.Renderer_.StationHandles_{obj.ConnectSource_};
                set(srcH, 'MarkerEdgeColor', td.Renderer_.StationColor * 0.6, 'LineWidth', 1.5);
                obj.ConnectSource_ = 0;
                td.updateStatus_();
            end
        end
    end
end
