classdef InteractionHandler < handle
    % INTERACTIONHANDLER  Handles mouse/keyboard input and mode logic for TrackDesigner.
    %Controller
    
    properties
        Designer_
    end
    
    methods
        function obj = InteractionHandler(designer)
            obj.Designer_ = designer;
        end
        
        function setMode_(obj, mode)
            td = obj.Designer_;
            td.Mode_ = mode;
            td.ConnectSource_ = 0;

            % Highlight active button
            allBtns = {'place', 'connect', 'edit', 'delete', 'insert'};
            activeColor = [0.3 0.5 0.8];
            inactiveColor = [0.2 0.2 0.25];
            for i = 1:numel(allBtns)
                btn = td.ToolButtons_.(allBtns{i});
                if strcmp(allBtns{i}, mode)
                    set(btn, 'BackgroundColor', activeColor, ...
                        'ForegroundColor', [1 1 1]);
                else
                    set(btn, 'BackgroundColor', inactiveColor, ...
                        'ForegroundColor', [0.8 0.9 1.0]);
                end
            end

            % Control points are completely hidden by default to save 90% UI thread overhead
            % We only spawn the exact 4 listeners for a specific curve when clicked directly in edit mode
            % Always clear them when switching modes, even to edit, until a curve is explicitly clicked
            if ~isempty(td.Renderer_)
                td.Renderer_.destroyAllControlPoints();
            end
            if ~strcmp(mode, 'edit')
                td.ActiveEditCurve_ = 0;
            end

            td.updateStatus_();
        end

        function onZoomCheck_(obj)
            td = obj.Designer_;
            % ONZOOMCHECK_ Evaluates whether the camera passes the zoom declutter threshold.
            if isempty(td.Renderer_) || ~isvalid(td.Axes_)
                return;
            end
            xl = xlim(td.Axes_);
            zoomRange = xl(2) - xl(1);
            isZoomedIn = (zoomRange < td.Renderer_.ZoomThreshold);
            
            if td.Renderer_.ZoomedIn_ ~= isZoomedIn
                td.Renderer_.ZoomedIn_ = isZoomedIn;
                td.onFocusModeChanged_(); 
            end
        end

        function onScrollZoom_(obj, evt)
            td = obj.Designer_;
            % Zoom toward cursor position
            cp = get(td.Axes_, 'CurrentPoint');
            mx = cp(1,1); my = cp(1,2);

            xl = get(td.Axes_, 'XLim');
            yl = get(td.Axes_, 'YLim');

            factor = 1.2;  % 20% per scroll step
            if evt.VerticalScrollCount > 0
                % Scroll down = zoom out
                s = factor;
            else
                % Scroll up = zoom in
                s = 1 / factor;
            end

            % Zoom centered on cursor position
            newXL = mx + (xl - mx) * s;
            newYL = my + (yl - my) * s;
            xlim(td.Axes_, newXL);
            ylim(td.Axes_, newYL);
        end

        function onPanEnd_(obj)
            td = obj.Designer_;
            td.IsPanning_ = false;
            set(td.Figure_, 'WindowButtonMotionFcn', '');
        end

        function onKeyPress_(obj, evt)
            td = obj.Designer_;
            % Ctrl+Z = Undo
            if strcmp(evt.Key, 'z') && ~isempty(evt.Modifier) && ...
                    any(strcmp(evt.Modifier, 'control'))
                td.onUndo_();
            end
        end

        function nearId = findNearestStation_(obj, clickPos)
            td = obj.Designer_;
            % Find the station nearest to the click position.
            % Adaptive threshold: 10% of the larger axes dimension.
            nearId = 0;
            minDist = inf;

            xl = get(td.Axes_, 'XLim');
            yl = get(td.Axes_, 'YLim');
            axRange = max(xl(2) - xl(1), yl(2) - yl(1));
            threshold = max(0.1, axRange * 0.10);  % at least 0.1, or 10% of range

            for i = 1:td.StationCount_
                pos = td.Transit_.getStation(i).Position;
                d = sqrt(sum((pos(1:2) - clickPos(1:2)).^2));
                if d < minDist && d < threshold
                    minDist = d;
                    nearId = i;
                end
            end
        end

        function onStationClick_(obj, nodeId)
            td = obj.Designer_;
            % Called when a station marker is clicked directly.
            
            % Always select the station to populate the properties panel
            td.selectStation_(nodeId);
            
            disp(['[Debug] Reached switch mode: ' td.Mode_]);
            switch td.Mode_
                case 'connect'
                    if td.ConnectSource_ == 0
                        td.ConnectSource_ = nodeId;
                        % Highlight selected station
                        h = td.Renderer_.StationHandles_{nodeId};
                        set(h, 'MarkerEdgeColor', [1 1 0], 'LineWidth', 2.5);
                        td.updateStatus_();
                    else
                        if nodeId ~= td.ConnectSource_
                            td.connectStations(td.ConnectSource_, nodeId);
                        end
                        % Unhighlight source station
                        srcH = td.Renderer_.StationHandles_{td.ConnectSource_};
                        set(srcH, 'MarkerEdgeColor', td.Renderer_.StationColor * 0.6, ...
                            'LineWidth', 1.5);
                        td.ConnectSource_ = 0;
                        td.updateStatus_();
                    end
                case 'place'
                    % Do nothing — place mode only responds to empty canvas
                case 'edit'
                    % Properties panel is already populated above
                case 'delete'
                    td.deleteStation_(nodeId);
            end
        end

        function onControlPointGrabbed_(obj, edgeIdx, cpIdx, startPos)
            % Capture memento before drag begins
            obj.Designer_.UndoManager_.pushState();
        end

        function onControlPointMoved_(obj, edgeIdx, cpIdx, newPos)
            td = obj.Designer_;
            % Update the local spline and redraw (Zero integration / physics logic, Pure UI)
            spline = td.Transit_.getSpline(edgeIdx);
            spline.setControlPoint(cpIdx, newPos);

            % Fast redraw curve using internal LOD proxy logic (Samples = 50)
            td.Renderer_.updateCurve(edgeIdx, spline, 'Samples', 50);
        end

        function onControlPointReleased_(obj, edgeIdx, cpIdx, newPos)
            td = obj.Designer_;
            % Update state (uses O(N) numerical integration for physical weight mapping!)
            % We defer this to "DROP" to completely eliminate dragging UI lag
            spline = td.Transit_.getSpline(edgeIdx);
            spline.setControlPoint(cpIdx, newPos);
            td.Transit_.updateRouteSpline(edgeIdx, spline);
            % Final HD presentation render
            td.Renderer_.updateCurve(edgeIdx, spline, 'Samples', 200);
        end

        function onAxesClick_(obj)
            td = obj.Designer_;
            % Ignore right-clicks (reserved for pan)
            selType = get(td.Figure_, 'SelectionType');
            if ~strcmp(selType, 'normal'), return; end

            cp = get(td.Axes_, 'CurrentPoint');
            
            % --- 3D Ray-Plane Intersection Math (Orthographic Option B) ---
            % We adapt instantly to whatever camera view the designer has chosen.
            pFront = cp(1, :);
            pBack  = cp(2, :);
            dx = pBack(1) - pFront(1);
            dy = pBack(2) - pFront(2);
            dz = pBack(3) - pFront(3);
            
            [az, el] = view(td.Axes_);
            
            if abs(el) >= 89.9 % TOP view -> Looking down Z-axis
                if abs(dz) > 1e-6
                    t = (td.WorkingPos_(3) - pFront(3)) / dz;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = pFront; clickPos(3) = td.WorkingPos_(3);
                end
            elseif abs(el) <= 0.1 && abs(az) <= 0.1 % FRONT view -> Looking down Y-axis
                if abs(dy) > 1e-6
                    t = (td.WorkingPos_(2) - pFront(2)) / dy;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = pFront; clickPos(2) = td.WorkingPos_(2);
                end
            elseif abs(el) <= 0.1 && (abs(az - 90) <= 0.1 || abs(az + 90) <= 0.1) % SIDE view -> Looking down X-axis
                if abs(dx) > 1e-6
                    t = (td.WorkingPos_(1) - pFront(1)) / dx;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = pFront; clickPos(1) = td.WorkingPos_(1);
                end
            else % Free Isometric 3D (Hybrid fallback to Top logic)
                if abs(dz) > 1e-6
                    t = (td.WorkingPos_(3) - pFront(3)) / dz;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = pFront; clickPos(3) = td.WorkingPos_(3);
                end
            end
            
            if isempty(clickPos)
                % The mouse perfectly aligned with the target plane, making math explode (Infinity).
                % This rarely triggers unless they somehow clicked from precisely edge-on in isometric.
                return;
            end

                        switch td.Mode_
                case 'place'
                    
                    name = sprintf('S%d', td.StationCount_ + 1);
                    name = inputdlg('Station name:', 'New Station', 1, {name});
                    if isempty(name), return; end
                    
                    td.pushUndo_(); % Record state right before placing station
                    td.addStation(name{1}, clickPos);
                    td.updateStatus_();

                case 'connect'
                    % Find nearest station to click
                    nearId = td.findNearestStation_(clickPos);
                    if nearId == 0
                        % Clicked empty space: cancel pending connection
                        if td.ConnectSource_ > 0
                            srcH = td.Renderer_.StationHandles_{td.ConnectSource_};
                            set(srcH, 'MarkerEdgeColor', td.Renderer_.StationColor * 0.6, ...
                                'LineWidth', 1.5);
                            td.ConnectSource_ = 0;
                            td.updateStatus_();
                        end
                        return;
                    end

                    if td.ConnectSource_ == 0
                        td.ConnectSource_ = nearId;
                        td.updateStatus_();
                    else
                        if nearId ~= td.ConnectSource_
                            td.connectStations(td.ConnectSource_, nearId);
                        end
                        td.ConnectSource_ = 0;
                        td.updateStatus_();
                    end

                case 'edit'
                    % Wake up the 4 listeners perfectly targeted to the curve clicked
                    xl = get(td.Axes_, 'XLim');
                    yl = get(td.Axes_, 'YLim');
                    axRange = max(xl(2) - xl(1), yl(2) - yl(1));
                    threshold = axRange * 0.08;

                    [edgeIdx, minDist] = td.Renderer_.findNearestCurve(cp, threshold); disp(['finding ' num2str(edgeIdx)]);
                    if edgeIdx > 0
                        td.ActiveEditCurve_ = edgeIdx;
                        td.Renderer_.destroyAllControlPoints(); % clear old ones
                        spline = td.Transit_.getSpline(edgeIdx);
                        if ~isempty(spline)
                            td.Renderer_.drawControlPoints(edgeIdx, spline, ...
                                @(eIdx, cpIdx, startPos) obj.onControlPointGrabbed_(eIdx, cpIdx, startPos), ...
                                @(eIdx, cpIdx, newPos) obj.onControlPointMoved_(eIdx, cpIdx, newPos), ...
                                @(eIdx, cpIdx, newPos) obj.onControlPointReleased_(eIdx, cpIdx, newPos));
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
                        nearStationId = td.findNearestStation_(clickPos);
                        if nearStationId == 0
                            td.selectStation_(0);
                        end
                    end
                    td.updateStatus_();

                case 'delete'
                    % Try station first, then edge
                    nearId = td.findNearestStation_(clickPos);
                    if nearId > 0
                        td.deleteStation_(nearId);
                    else
                        td.deleteNearestEdge_(clickPos);
                    end

                case 'insert'
                    td.insertStationOnCurve_(clickPos);
            end
        end

        function onPanStart_(obj)
            td = obj.Designer_;
            selType = get(td.Figure_, 'SelectionType');
            if ~strcmp(selType, 'alt'), return; end  % right-click only

            cp = get(td.Axes_, 'CurrentPoint');
            td.PanOrigin_ = [cp(1,1), cp(1,2)];
            td.IsPanning_ = true;
            set(td.Figure_, 'WindowButtonMotionFcn', @(~,~) td.onPanMove_());
        end

        function onPanMove_(obj)
            td = obj.Designer_;
            if ~td.IsPanning_, return; end
            cp = get(td.Axes_, 'CurrentPoint');
            dx = td.PanOrigin_(1) - cp(1,1);
            dy = td.PanOrigin_(2) - cp(1,2);

            xl = get(td.Axes_, 'XLim');
            yl = get(td.Axes_, 'YLim');
            xlim(td.Axes_, xl + dx);
            ylim(td.Axes_, yl + dy);
        end

        function deleteNearestEdge_(obj, clickPos)
            td = obj.Designer_;
            % DELETENEARESTEDGE_  Find and remove the nearest edge to click.
            td.pushUndo_();
            
            xl = get(td.Axes_, 'XLim');
            yl = get(td.Axes_, 'YLim');
            axRange = max(xl(2) - xl(1), yl(2) - yl(1));
            threshold = axRange * 0.08;

            cp = get(td.Axes_, 'CurrentPoint');
            [edgeIdx, ~] = td.Renderer_.findNearestCurve(cp, threshold);
            if edgeIdx == 0, return; end

            % Find source/target from the graph edge
            e = td.Transit_.getRouteEdge(edgeIdx);
            if isempty(e), return; end

            % Remove from backend
            td.Transit_.removeRoute(e.Source, e.Target);

            % Remove from renderer
            td.Renderer_.deleteCurve(edgeIdx);
            td.updateStatus_();
        end

        function insertStationOnCurve_(obj, clickPos)
            td = obj.Designer_;
            % INSERTSTATIONONCURVE_  Place a new station on the nearest curve.
            %   Splits the edge into two sub-edges.
            td.pushUndo_();
            
            xl = get(td.Axes_, 'XLim');
            yl = get(td.Axes_, 'YLim');
            axRange = max(xl(2) - xl(1), yl(2) - yl(1));
            threshold = axRange * 0.08;

            cp = get(td.Axes_, 'CurrentPoint');
            [edgeIdx, ~] = td.Renderer_.findNearestCurve(cp, threshold);
            if edgeIdx == 0, return; end

            % Get the edge source/target
            e = td.Transit_.getRouteEdge(edgeIdx);
            if isempty(e), return; end
            srcId = e.Source;
            tgtId = e.Target;

            % Get the spline and project click ray onto it precisely
            spline = td.Transit_.getSpline(edgeIdx);
            tVals = spline.parameterize(200);
            pts = spline.evaluate(tVals);
            
            pFront = cp(1, :);
            vDir = cp(2, :) - pFront;
            vDir = vDir / norm(vDir);
            W = pts - pFront;
            projs = W(:,1)*vDir(1) + W(:,2)*vDir(2) + W(:,3)*vDir(3);
            W_perp = W - projs * vDir;
            dists = sqrt(sum(W_perp.^2, 2));
            
            [~, nearIdx] = min(dists);
            insertPos = pts(nearIdx, :);

            % Create the new station
            td.StationCount_ = td.StationCount_ + 1;
            name = sprintf('S%d', td.StationCount_);
            newId = td.Transit_.addStation(name, insertPos);

            % Determine edge color (from old edge or active line)
            if edgeIdx <= numel(td.Renderer_.CurveColors_) && ...
                    ~isempty(td.Renderer_.CurveColors_{edgeIdx})
                edgeColor = td.Renderer_.CurveColors_{edgeIdx};
            else
                edgeColor = [0.4, 0.8, 1.0];
            end

            % Remove old edge (backend + visual)
            td.Transit_.removeRoute(srcId, tgtId);
            td.Renderer_.deleteCurve(edgeIdx);

            % Create two sub-curves that follow the ORIGINAL curve's shape
            tSplit = tVals(nearIdx);  % parameter at split point
            isCatmull = isa(spline, 'OceanMath.Geometry.CatmullRomSpline');

            if isCatmull
                % For interpolating splines, sample points on the curve
                % to use as the new strictly on-curve control points
                t1 = linspace(0, tSplit, 4)';
                cp1 = spline.evaluate(t1);
                spl1 = OceanMath.Geometry.CatmullRomSpline(cp1);
                
                t2 = linspace(tSplit, 1, 4)';
                cp2 = spline.evaluate(t2);
                spl2 = OceanMath.Geometry.CatmullRomSpline(cp2);
            else
                % Helper to fit cubic Bezier control points to 4 exactly evaluated points
                % at t = [0, 1/3, 2/3, 1]
                fitCubic = @(Q) [ ...
                    Q(1,:); ...
                    (-5*Q(1,:) + 18*Q(2,:) - 9*Q(3,:) + 2*Q(4,:)) / 6; ...
                    (2*Q(1,:) - 9*Q(2,:) + 18*Q(3,:) - 5*Q(4,:)) / 6; ...
                    Q(4,:) ];

                % First half: t = [0, tSplit] -> sample 4 points
                t1 = linspace(0, tSplit, 4)';
                q1 = spline.evaluate(t1);
                cp1 = fitCubic(q1);
                spl1 = OceanMath.Geometry.BezierCurve(cp1);

                % Second half: t = [tSplit, 1] -> sample 4 points
                t2 = linspace(tSplit, 1, 4)';
                q2 = spline.evaluate(t2);
                cp2 = fitCubic(q2);
                spl2 = OceanMath.Geometry.BezierCurve(cp2);
            end

            % Add routes to backend
            td.EdgeCount_ = td.EdgeCount_ + 1;
            e1Idx = td.EdgeCount_;
            td.Transit_.addRoute(srcId, newId, spl1);

            td.EdgeCount_ = td.EdgeCount_ + 1;
            e2Idx = td.EdgeCount_;
            td.Transit_.addRoute(newId, tgtId, spl2);

            % Inherit lines from the edge we just split
            lineCodes = td.Transit_.getLineNames();
            for lcIdx = 1:numel(lineCodes)
                lc = lineCodes{lcIdx};
                lObj = td.Transit_.getLine(lc);
                if lObj.hasStation(srcId) && lObj.hasStation(tgtId)
                    lObj.registerStation(newId);
                end
            end
            
            % Also ensure it's registered on active line if any
            if ~isempty(td.ActiveLineCode_) && any(strcmp(td.Transit_.getLineNames(), td.ActiveLineCode_))
                lineObj = td.Transit_.getLine(td.ActiveLineCode_);
                lineObj.registerStation(newId);
            end

            % Render new station + sub-curves
            if ~isempty(td.Renderer_)
                node = td.Transit_.getStation(newId);
                td.Renderer_.drawStation(node, @(nId) obj.onStationClick_(nId));
                td.Renderer_.drawCurve(e1Idx, spl1, 'Color', edgeColor);
                td.Renderer_.drawCurve(e2Idx, spl2, 'Color', edgeColor);
                td.Renderer_.drawControlPoints(e1Idx, spl1, ...
                    @(eIdx, cpIdx, startPos) obj.onControlPointGrabbed_(eIdx, cpIdx, startPos), ...
                    @(eIdx, cpIdx, newPos) obj.onControlPointMoved_(eIdx, cpIdx, newPos), ...
                    @(eIdx, cpIdx, newPos) obj.onControlPointReleased_(eIdx, cpIdx, newPos));
                td.Renderer_.drawControlPoints(e2Idx, spl2, ...
                    @(eIdx, cpIdx, startPos) obj.onControlPointGrabbed_(eIdx, cpIdx, startPos), ...
                    @(eIdx, cpIdx, newPos) obj.onControlPointMoved_(eIdx, cpIdx, newPos), ...
                    @(eIdx, cpIdx, newPos) obj.onControlPointReleased_(eIdx, cpIdx, newPos));
                td.Renderer_.bringStationsToFront();
            end
            
            % Update visuals to show interchange if applicable
            td.updateStationVisuals_(newId);
            td.updateStatus_();
        end

    end
end
