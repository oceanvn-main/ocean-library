classdef InteractionHandler < handle
    % INTERACTIONHANDLER  Controller — handles mouse/keyboard input for TrackDesigner.
    %
    %   All mode-specific logic is delegated to the active IDesignerMode
    %   object (State Pattern). This class retains only:
    %     - Shared infrastructure (zoom, pan, keyboard shortcuts)
    %     - 3D ray-plane intersection math (shared by all modes)
    %     - Utility methods (findNearestStation_, deleteNearestEdge_, etc.)
    %     - DraggablePoint callback targets
    %
    % See also: OceanViz.Designer.Modes.IDesignerMode,
    %           OceanViz.Designer.TrackDesigner

    properties
        Designer_
    end

    methods

        function obj = InteractionHandler(designer)
            obj.Designer_ = designer;
        end

        % ----------------------------------------------------------------
        %  MODE MANAGEMENT (State Pattern)
        % ----------------------------------------------------------------

        function setMode_(obj, modeName)
            td = obj.Designer_;

            % Exit the current mode (cleanup)
            if ~isempty(td.ActiveMode_)
                td.ActiveMode_.exit(obj);
            end

            % Always destroy control points on any mode switch
            if ~isempty(td.Renderer_)
                td.Renderer_.destroyAllControlPoints();
            end

            % Create the new mode object
            td.ActiveMode_ = obj.createMode_(modeName);
            td.ActiveMode_.enter(obj);

            % Highlight the active toolbar button
            obj.highlightButton_(modeName);
            td.updateStatus_();
        end

        % ----------------------------------------------------------------
        %  ZOOM & PAN (shared infrastructure)
        % ----------------------------------------------------------------

        function onZoomCheck_(obj)
            td = obj.Designer_;
            if isempty(td.Renderer_) || ~isvalid(td.Axes_)
                return;
            end
            xl = xlim(td.Axes_);
            zoomRange = xl(2) - xl(1);
            isZoomedIn = (zoomRange < td.Renderer_.ZoomThreshold);

            if td.Renderer_.ZoomedIn_ ~= isZoomedIn
                td.Renderer_.ZoomedIn_ = isZoomedIn;
                if ~isempty(td.PropertyController_)
                    td.PropertyController_.onFocusModeChanged();
                end
            end
        end

        function onScrollZoom_(obj, evt)
            td = obj.Designer_;
            cp = get(td.Axes_, 'CurrentPoint');
            mx = cp(1,1); my = cp(1,2);

            xl = get(td.Axes_, 'XLim');
            yl = get(td.Axes_, 'YLim');

            factor = 1.2;
            if evt.VerticalScrollCount > 0
                s = factor; %Zoom out
            else
                s = 1 / factor; %Zoom in
            end

            newXL = mx + (xl - mx) * s;
            newYL = my + (yl - my) * s;
            xlim(td.Axes_, newXL);
            ylim(td.Axes_, newYL);

            % Trigger LOD decluttering check after zoom
            obj.onZoomCheck_();
        end

        function onPanStart_(obj)
            td = obj.Designer_;
            selType = get(td.Figure_, 'SelectionType');
            if ~strcmp(selType, 'alt'), return; end

            cp = get(td.Axes_, 'CurrentPoint');
            td.PanOrigin_ = [cp(1,1), cp(1,2)];
            td.IsPanning_ = true;
            set(td.Figure_, 'WindowButtonMotionFcn', @(~,~) obj.onPanMove_());
        end

        function onPanMove_(obj)
            td = obj.Designer_;
            if ~td.IsPanning_, return; end
            cp = get(td.Axes_, 'CurrentPoint'); %Notice this use td.Axes_ which returns exact coordinates(3D), differnt from td.Figure_ which returns 2D location of glass monitor
            dx = td.PanOrigin_(1) - cp(1,1);
            dy = td.PanOrigin_(2) - cp(1,2);

            xl = get(td.Axes_, 'XLim');
            yl = get(td.Axes_, 'YLim');
            xlim(td.Axes_, xl + dx);
            ylim(td.Axes_, yl + dy);
            
            drawnow limitrate;
        end

        function onPanEnd_(obj)
            td = obj.Designer_;
            td.IsPanning_ = false;
            set(td.Figure_, 'WindowButtonMotionFcn', '');
        end

        % ----------------------------------------------------------------
        %  KEYBOARD SHORTCUTS
        % ----------------------------------------------------------------

        function onKeyPress_(obj, evt)
            td = obj.Designer_;
            % Ctrl+Z = Undo
            if strcmp(evt.Key, 'z') && ~isempty(evt.Modifier) && ...
                    any(strcmp(evt.Modifier, 'control'))
                if ~isempty(td.TimelineController_)
                    td.TimelineController_.undo();
                end
                return;
            end
            % Ctrl+Y = Redo
            if strcmp(evt.Key, 'y') && ~isempty(evt.Modifier) && ...
                    any(strcmp(evt.Modifier, 'control'))
                td.TimelineController_.redo();
                return;
            end
        end

        % ----------------------------------------------------------------
        %  AXES CLICK (3D Ray-Plane Intersection + Mode Delegation)
        % ----------------------------------------------------------------

        function onAxesClick_(obj)
            td = obj.Designer_;
            selType = get(td.Figure_, 'SelectionType');
            if ~strcmp(selType, 'normal'), return; end

            cp = get(td.Axes_, 'CurrentPoint');

            % --- 3D Ray-Plane Intersection Math (shared by ALL modes) ---
            pFront = cp(1, :);
            pBack  = cp(2, :);
            dx = pBack(1) - pFront(1);
            dy = pBack(2) - pFront(2);
            dz = pBack(3) - pFront(3);

            [az, el] = view(td.Axes_);

            if abs(el) >= 89.9
                if abs(dz) > 1e-6
                    t = (td.WorkingPos_(3) - pFront(3)) / dz;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = pFront; clickPos(3) = td.WorkingPos_(3);
                end
            elseif abs(el) <= 0.1 && abs(az) <= 0.1
                if abs(dy) > 1e-6
                    t = (td.WorkingPos_(2) - pFront(2)) / dy;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = pFront; clickPos(2) = td.WorkingPos_(2);
                end
            elseif abs(el) <= 0.1 && (abs(az - 90) <= 0.1 || abs(az + 90) <= 0.1)
                if abs(dx) > 1e-6
                    t = (td.WorkingPos_(1) - pFront(1)) / dx;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = pFront; clickPos(1) = td.WorkingPos_(1);
                end
            else
                if abs(dz) > 1e-6
                    t = (td.WorkingPos_(3) - pFront(3)) / dz;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = pFront; clickPos(3) = td.WorkingPos_(3);
                end
            end

            if isempty(clickPos), return; end

            % Delegate to the active mode (State Pattern)
            td.ActiveMode_.onAxesClick(obj, clickPos, cp);
        end

        % ----------------------------------------------------------------
        %  STATION CLICK (Direct marker click + Mode Delegation)
        % ----------------------------------------------------------------

        function onStationClick_(obj, nodeId)
            td = obj.Designer_;

            % Always select the station for the properties panel
            td.selectStation_(nodeId);

            % Delegate to the active mode (State Pattern)
            td.ActiveMode_.onStationClick(obj, nodeId);
        end

        % ----------------------------------------------------------------
        %  DRAGGABLE POINT CALLBACKS (shared by EditMode & InsertMode)
        % ----------------------------------------------------------------

        function onControlPointGrabbed_(obj, edgeIdx, cpIdx, startPos) %#ok<INUSD>
            obj.Designer_.WorkingPos_ = startPos;
        end

        function onControlPointMoved_(obj, edgeIdx, cpIdx, newPos)
            td = obj.Designer_;
            spline = td.Transit_.getSpline(edgeIdx);
            spline.setControlPoint(cpIdx, newPos);
            td.Renderer_.updateCurve(edgeIdx, spline, 'Samples', 50);
        end

        function onControlPointReleased_(obj, edgeIdx, cpIdx, newPos)
            td = obj.Designer_;
            startPos = td.WorkingPos_;
            if isempty(startPos), startPos = newPos; end
            
            cmd = OceanViz.Designer.Commands.MoveControlPointCommand(td, edgeIdx, cpIdx, startPos, newPos);
            if ~isempty(td.TimelineController_)
                td.TimelineController_.execute(cmd);
            else
                cmd.execute();
            end
        end

        % ----------------------------------------------------------------
        %  UTILITY METHODS (used by Mode classes)
        % ----------------------------------------------------------------

        function nearId = findNearestStation_(obj, clickPos)
            td = obj.Designer_;
            nearId = 0;

            xl = get(td.Axes_, 'XLim');
            yl = get(td.Axes_, 'YLim');
            axRange = max(xl(2) - xl(1), yl(2) - yl(1));
            threshold = max(0.1, axRange * 0.10); %10 percent of the screen width

            if td.StationCount_ == 0, return; end

            % O(log N) spatial lookup via QuadTree
            if ~isempty(td.SpatialIdx_)
                nearId = td.SpatialIdx_.findNearest(clickPos(1:2), threshold);
            else
                % Fallback: vectorized O(N) if QuadTree not built yet
                positions = td.Transit_.getStationPositions();
                dists = sqrt(sum((positions(:,1:2) - clickPos(1:2)).^2, 2));
                dists(dists >= threshold) = inf;
                [minDist, nearId] = min(dists);
                if minDist == inf, nearId = 0; end
            end
        end

        function deleteNearestEdge_(obj, clickPos) %#ok<INUSD>
            td = obj.Designer_;

            xl = get(td.Axes_, 'XLim');
            yl = get(td.Axes_, 'YLim');
            axRange = max(xl(2) - xl(1), yl(2) - yl(1));
            threshold = axRange * 0.08;

            cp = get(td.Axes_, 'CurrentPoint');
            [edgeIdx, ~] = td.Renderer_.findNearestCurve(cp, threshold);
            if edgeIdx == 0, return; end

            e = td.Transit_.getRouteEdge(edgeIdx);
            if isempty(e), return; end
            % Confirm deletion and wipe undo stack
            ansStr = questdlg('Delete this route edge? (Cannot be undone)', 'Confirm', 'Yes', 'No', 'No');
            if ~strcmp(ansStr, 'Yes'), return; end
            
            if ~isempty(td.TimelineController_)
                td.TimelineController_.clear();
            end

            td.Transit_.removeRoute(e.Source, e.Target);
            td.Renderer_.deleteCurve(edgeIdx);
            td.updateStatus_();
        end

        function [grabCb, moveCb, relCb] = makeCallbacks_(obj)
            % MAKECALLBACKS_  Build DraggablePoint callback triplet.
            grabCb = @(eIdx, cpIdx, pos) obj.onControlPointGrabbed_(eIdx, cpIdx, pos);
            moveCb = @(eIdx, cpIdx, pos) obj.onControlPointMoved_(eIdx, cpIdx, pos);
            relCb  = @(eIdx, cpIdx, pos) obj.onControlPointReleased_(eIdx, cpIdx, pos);
        end

    end

    % ====================================================================
    %  PRIVATE HELPERS
    % ====================================================================

    methods (Access = private)

        function mode = createMode_(~, modeName)
            % CREATEMODE_  Factory method mapping string name to Mode object.
            switch modeName
                case 'place',   mode = OceanViz.Designer.Modes.PlaceMode();
                case 'connect', mode = OceanViz.Designer.Modes.ConnectMode();
                case 'edit',    mode = OceanViz.Designer.Modes.EditMode();
                case 'delete',  mode = OceanViz.Designer.Modes.DeleteMode();
                case 'insert',  mode = OceanViz.Designer.Modes.InsertMode();
                otherwise,      mode = OceanViz.Designer.Modes.PlaceMode();
            end
        end

        function highlightButton_(obj, modeName)
            % HIGHLIGHTBUTTON_  Visually toggle the active toolbar button.
            td = obj.Designer_;
            allBtns = {'place', 'connect', 'edit', 'delete', 'insert'};
            activeColor   = [0.3 0.5 0.8];
            inactiveColor = [0.2 0.2 0.25];
            for i = 1:numel(allBtns)
                btn = td.ToolButtons_.(allBtns{i});
                if strcmp(allBtns{i}, modeName)
                    set(btn, 'BackgroundColor', activeColor, ...
                        'ForegroundColor', [1 1 1]);
                else
                    set(btn, 'BackgroundColor', inactiveColor, ...
                        'ForegroundColor', [0.8 0.9 1.0]);
                end
            end
        end



    end
end
