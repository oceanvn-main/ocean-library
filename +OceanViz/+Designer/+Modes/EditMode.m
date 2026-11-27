classdef EditMode < OceanViz.Designer.Modes.BaseMode
    % EDITMODE  Interaction mode for dragging control points on transit curves.
    %
    %   Clicking near a curve spawns DraggablePoint markers along its path.
    %   Dragging these markers calls the spline update logic continuously.
    %   Clicking empty space despawns the markers to free up UI thread.
    %
    % See also: OceanViz.Designer.Modes.IDesignerMode

    methods
        function onAxesClick(obj, handler, clickPos, rawPoint)
            td = handler.Designer_;
            
            % Wake up the listeners targeted to the curve clicked
            xl = get(td.Axes_, 'XLim');
            yl = get(td.Axes_, 'YLim');
            axRange = max(xl(2) - xl(1), yl(2) - yl(1));
            threshold = axRange * 0.08;

            [edgeIdx, ~] = td.Renderer_.findNearestCurve(rawPoint, threshold);
            if edgeIdx > 0
                td.ActiveEditCurve_ = edgeIdx;
                td.Renderer_.destroyAllControlPoints(); % clear old ones
                spline = td.Transit_.getSpline(edgeIdx);
                if ~isempty(spline)
                    [grabCb, moveCb, relCb] = handler.makeCallbacks_();
                    td.Renderer_.drawControlPoints(edgeIdx, spline, grabCb, moveCb, relCb);
                end
                % Update draw order indicator
                order = td.Renderer_.getCurveDrawOrder(edgeIdx);
                set(td.PropPanel_.drawOrderLabel, 'String', ...
                    sprintf('Active Curve: Edge %d  |  Order: %d', edgeIdx, order));
            else
                td.ActiveEditCurve_ = 0;
                td.Renderer_.destroyAllControlPoints(); % free memory
                set(td.PropPanel_.drawOrderLabel, 'String', 'Active Curve: (none)');
                
                % If clicked completely empty space, unselect station too
                nearStationId = handler.findNearestStation_(clickPos);
                if nearStationId == 0
                    td.selectStation_(0);
                end
            end
            td.updateStatus_();
        end

        function onStationClick(~, ~, ~)
            % Properties panel is automatically populated by InteractionHandler
            % before this delegation. EditMode ignores direct station clicks
            % so that the control points aren't unnecessarily cleared.
        end

        function exit(obj, handler)
            td = handler.Designer_;
            td.ActiveEditCurve_ = 0;
            if ~isempty(td.Renderer_)
                td.Renderer_.destroyAllControlPoints();
            end
            if isfield(td.PropPanel_, 'drawOrderLabel') && isvalid(td.PropPanel_.drawOrderLabel)
                set(td.PropPanel_.drawOrderLabel, 'String', 'Active Curve: (none)');
            end
        end

        function msg = getStatusText(~, ~)
            msg = 'Mode: EDIT  |  Left-Drag: Move X/Y.  Shift-Drag: Move Z (Height).';
        end

        function name = getName(~)
            name = 'edit';
        end
    end
end
