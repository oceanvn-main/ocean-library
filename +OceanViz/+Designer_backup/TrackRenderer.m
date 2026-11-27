classdef TrackRenderer < handle
    % TRACKRENDERER  Renders track state onto MATLAB axes.
    %
    % Draws stations as colored dots with labels, spline curves as smooth
    % lines, and control points as draggable markers. Uses OceanViz colors
    % when available.
    %
    % See also: OceanViz.Designer.TrackDesigner

    % ------------------------------------------------------------------ %
    properties (SetAccess = private, GetAccess = public)
        Axes_
        StationHandles_     % cell array of plot handles for station dots
        LabelHandles_       % cell array of text handles for station names
        CurveHandles_       % cell array of line handles for spline curves
        CurveHaloHandles_   % cell array of thick background halo handles
        ControlPointHandles_ % cell array of DraggablePoint arrays
        LabelOffsets_       % cell array of [dx, dy] per node
        CurveColors_        % cell array of [r,g,b] per edge
        CurveWidths_        % cell array of scalar widths per edge
        CurveDrawOrders_    % cell array of integer draw orders per edge
        LabelsVisible_  (1,1) logical = true
        StationsVisible_(1,1) logical = true
        
        IsFocus_        (1,1) logical = false
        FocusCode_      (1,:) char = ''
        FocusNetwork_   % OceanMath.Transit.TransitNetwork reference
    end

    properties
        LabelIsMajor_   % logical array mapping 1:N indicating topological priority
        ZoomedIn_       (1,1) logical = true
        StationColor    (1,3) double = [0.2, 0.6, 1.0]
        StationSize     (1,1) double = 16
        CurveColor      (1,3) double = [0.4, 0.8, 1.0]
        CurveLineWidth  (1,1) double = 4.2
        ControlColor    (1,3) double = [1, 0.4, 0]
        ControlSize     (1,1) double = 8
        LabelColor      (1,3) double = [0.1, 0.1, 0.1]
        CurveSamples    (1,1) double = 200

        % Canvas background color used for halo masking
        CanvasBackground (1,3) double = [1, 1, 1]
        HaloExtraWidth   (1,1) double = 4

        % Default label offset (relative to station position)
        DefaultLabelOffset (1,2) double = [0, 0.3]
        
        % Zoom limit threshold to trigger semantic decluttering maps
        ZoomThreshold (1,1) double = 800
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = TrackRenderer(ax)
            obj.Axes_ = ax;
            obj.StationHandles_ = {};
            obj.LabelHandles_ = {};
            obj.CurveHandles_ = {};
            obj.CurveHaloHandles_ = {};
            obj.ControlPointHandles_ = {};
            obj.LabelOffsets_ = {};
            obj.CurveColors_ = {};
            obj.CurveWidths_ = {};
            obj.CurveDrawOrders_ = {};
            obj.LabelIsMajor_ = logical([]);
        end

        function drawStation(obj, node, clickCb)
            % DRAWSTATION  Draw a single station dot + label.
            %   clickCb (optional) = @(nodeId) called when station is clicked
            id = node.Id;
            pos = node.Position;

            hold(obj.Axes_, 'on');

            z = 0; if numel(pos) >= 3, z = pos(3); end
            
            % Station marker — with click callback for connect mode
            if nargin >= 3 && ~isempty(clickCb)
                btnFcn = @(~,~) clickCb(id);
            else
                btnFcn = '';
            end
            h = plot3(obj.Axes_, pos(1), pos(2), z, 'o', ...
                'MarkerFaceColor', [1 1 1], ...
                'MarkerEdgeColor', [0.1 0.1 0.1], ...
                'MarkerSize', obj.StationSize * 0.35, ...
                'LineWidth', 1.0, ...
                'ButtonDownFcn', btnFcn);
            obj.StationHandles_{id} = h;

            % Store default label offset
            obj.LabelOffsets_{id} = obj.DefaultLabelOffset;
            offset = obj.LabelOffsets_{id};
            
            dx = offset(1);
            dy = offset(2);
            hAlign = 'center';
            if dx > 0.05, hAlign = 'left'; elseif dx < -0.05, hAlign = 'right'; end
            vAlign = 'middle';
            if dy > 0.05, vAlign = 'bottom'; elseif dy < -0.05, vAlign = 'top'; end

            % Label — HitTest off so it doesn't block clicks on the station
            t = text(obj.Axes_, pos(1) + dx, pos(2) + dy, z, node.Name, ...
                'Color', obj.LabelColor, ...
                'FontSize', 8, ...
                'FontWeight', 'normal', ...
                'HorizontalAlignment', hAlign, ...
                'VerticalAlignment', vAlign, ...
                'BackgroundColor', obj.CanvasBackground, ...
                'Margin', 0.1, ...
                'EdgeColor', 'none', ...
                'HitTest', 'off', ...
                'PickableParts', 'none');
            obj.LabelHandles_{id} = t;
        end

        function setLabelOffset(obj, nodeId, offset)
            % SETLABELOFFSET  Set the label offset [dx, dy] for a station.
            %
            %   renderer.setLabelOffset(1, [0.5, 0.2])  % right and slightly up
            %   renderer.setLabelOffset(1, [-0.3, 0])    % to the left
            arguments
                obj
                nodeId (1,1) double {mustBePositive, mustBeInteger}
                offset (1,2) double
            end
            obj.LabelOffsets_{nodeId} = offset;
            % Update the rendered label position
            if nodeId <= numel(obj.LabelHandles_) && ...
                    ~isempty(obj.LabelHandles_{nodeId}) && ...
                    isvalid(obj.LabelHandles_{nodeId})
                sH = obj.StationHandles_{nodeId};
                stationX = get(sH, 'XData');
                stationY = get(sH, 'YData');
                stationZ = get(sH, 'ZData');
                if isempty(stationZ), stationZ = 0; end
                
                dx = offset(1);
                dy = offset(2);
                hAlign = 'center';
                if dx > 0.05, hAlign = 'left'; elseif dx < -0.05, hAlign = 'right'; end
                vAlign = 'middle';
                if dy > 0.05, vAlign = 'bottom'; elseif dy < -0.05, vAlign = 'top'; end
                
                set(obj.LabelHandles_{nodeId}, ...
                    'Position', [stationX + dx, stationY + dy, stationZ], ...
                    'HorizontalAlignment', hAlign, ...
                    'VerticalAlignment', vAlign);
            end
        end

        function setAllLabelOffsets(obj, offset)
            % SETALLLABELOFFSETS  Apply the same offset to all labels.
            %
            %   renderer.setAllLabelOffsets([0.3, 0.15])  % consistent offset
            arguments
                obj
                offset (1,2) double
            end
            for i = 1:numel(obj.LabelHandles_)
                if ~isempty(obj.LabelHandles_{i}) && isvalid(obj.LabelHandles_{i})
                    obj.setLabelOffset(i, offset);
                end
            end
        end

        function updateLabelText(obj, nodeId, newName)
            % UPDATELABELTEXT  Change the displayed text of a station label.
            arguments
                obj
                nodeId  (1,1) double {mustBePositive, mustBeInteger}
                newName (1,:) char
            end
            if nodeId <= numel(obj.LabelHandles_) && ...
                    ~isempty(obj.LabelHandles_{nodeId}) && ...
                    isvalid(obj.LabelHandles_{nodeId})
                set(obj.LabelHandles_{nodeId}, 'String', newName);
            end
        end

        function updateStationStyle(obj, nodeId, styleType, edgeColorOverride)
            % UPDATESTATIONSTYLE  Change marker based on interchange/terminal classification.
            arguments
                obj
                nodeId (1,1) double {mustBePositive, mustBeInteger}
                styleType (1,:) char
                edgeColorOverride double = []
            end
            if nodeId <= numel(obj.StationHandles_) && ...
                    ~isempty(obj.StationHandles_{nodeId}) && ...
                    isvalid(obj.StationHandles_{nodeId})
                
                h = obj.StationHandles_{nodeId};
                
                t = [];
                if nodeId <= numel(obj.LabelHandles_)
                    t = obj.LabelHandles_{nodeId};
                end
                
                switch styleType
                    case 'interchange'
                        set(h, 'Marker', 'o', ...
                               'MarkerFaceColor', [1 1 1], ...      
                               'MarkerEdgeColor', [0 0 0], ... 
                               'LineWidth', 2.4, ...
                               'MarkerSize', obj.StationSize * 0.7);
                        if ~isempty(t) && isvalid(t), set(t, 'FontWeight', 'bold'); end
                    case 'terminal'
                        % Terminal stations: diamond (rhombus) marker
                        set(h, 'Marker', 'd', ...
                               'MarkerFaceColor', [1 1 1], ...
                               'MarkerEdgeColor', [0.15 0.15 0.15], ...
                               'LineWidth', 1.8, ...
                               'MarkerSize', obj.StationSize * 0.55);
                        if ~isempty(t) && isvalid(t), set(t, 'FontWeight', 'bold'); end
                    case 'standard'
                        % Hollow circle matching parent line color!
                        eCol = [0.3 0.3 0.3];
                        if ~isempty(edgeColorOverride)
                            eCol = edgeColorOverride;
                        end
                        set(h, 'Marker', 'o', ...
                               'MarkerFaceColor', [1 1 1], ...
                               'MarkerEdgeColor', eCol, ...
                               'LineWidth', 1.5, ...
                               'MarkerSize', obj.StationSize * 0.45);
                        if ~isempty(t) && isvalid(t), set(t, 'FontWeight', 'normal'); end
                end
            end
        end

         function drawCurve(obj, edgeIdx, spline, options)
            % DRAWCURVE  Draw a spline curve for an edge.
            %   Draws a background-colored halo underneath the color line
            %   to enable visual bridge/overpass effects at intersections.
            %   Draw order is handled by importJSON_ two-pass rendering.
            arguments
                obj
                edgeIdx (1,1) double
                spline
                options.Color (1,3) double = obj.CurveColor
                options.LineWidth (1,1) double = obj.CurveLineWidth
                options.DrawOrder (1,1) double = 0
            end
            tVals = spline.parameterize(obj.CurveSamples);
            pts = spline.evaluate(tVals);

            zVals = zeros(size(pts,1), 1);

            hold(obj.Axes_, 'on');
            
            % 1. Halo line (only for curves that need to visually pass over others)
            if options.DrawOrder > 0
                haloW = options.LineWidth + obj.HaloExtraWidth;
                hHalo = plot3(obj.Axes_, pts(:,1), pts(:,2), zVals, '-', ...
                    'Color', obj.CanvasBackground, ...
                    'LineWidth', haloW, ...
                    'HitTest', 'off', ...
                    'PickableParts', 'none');
                obj.CurveHaloHandles_{edgeIdx} = hHalo;
            end
            
            % 2. Color line
            h = plot3(obj.Axes_, pts(:,1), pts(:,2), zVals, '-', ...
                'Color', options.Color, ...
                'LineWidth', options.LineWidth, ...
                'HitTest', 'off', ...
                'PickableParts', 'none');
            obj.CurveHandles_{edgeIdx} = h;
            obj.CurveColors_{edgeIdx} = options.Color;
            obj.CurveWidths_{edgeIdx} = options.LineWidth;
            obj.CurveDrawOrders_{edgeIdx} = options.DrawOrder;
        end

        function updateCurve(obj, edgeIdx, spline)
            % UPDATECURVE  Redraw an existing curve after control point edit.
            %   Preserves the original edge color and width.
            tVals = spline.parameterize(obj.CurveSamples);
            pts = spline.evaluate(tVals);
            
            width = obj.CurveLineWidth;
            if edgeIdx <= numel(obj.CurveWidths_) && ~isempty(obj.CurveWidths_{edgeIdx})
                width = obj.CurveWidths_{edgeIdx};
            end

            zVals = zeros(size(pts,1), 1);
            if size(pts, 2) >= 3, zVals = pts(:,3); end

            % Update halo handle
            if edgeIdx <= numel(obj.CurveHaloHandles_) && ...
                    ~isempty(obj.CurveHaloHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHaloHandles_{edgeIdx})
                haloW = width + obj.HaloExtraWidth;
                set(obj.CurveHaloHandles_{edgeIdx}, ...
                    'XData', pts(:,1), 'YData', pts(:,2), 'ZData', zVals, ...
                    'LineWidth', haloW);
            end

            % Update color handle
            if edgeIdx <= numel(obj.CurveHandles_) && ...
                    ~isempty(obj.CurveHandles_{edgeIdx})
                set(obj.CurveHandles_{edgeIdx}, ...
                    'XData', pts(:,1), 'YData', pts(:,2), 'ZData', zVals, 'LineWidth', width);
            else
                % Restore saved color or use default
                if edgeIdx <= numel(obj.CurveColors_) && ~isempty(obj.CurveColors_{edgeIdx})
                    c = obj.CurveColors_{edgeIdx};
                else
                    c = obj.CurveColor;
                end
                obj.drawCurve(edgeIdx, spline, 'Color', c, 'LineWidth', width);
            end
        end

        function setCurveColor(obj, edgeIdx, color)
            % SETCURVECOLOR  Update the color of an existing curve.
            arguments
                obj
                edgeIdx (1,1) double
                color   (1,3) double
            end
            obj.CurveColors_{edgeIdx} = color;
            if edgeIdx <= numel(obj.CurveHandles_) && ...
                    ~isempty(obj.CurveHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHandles_{edgeIdx})
                set(obj.CurveHandles_{edgeIdx}, 'Color', color);
            end
        end

        function setCurveWidth(obj, edgeIdx, width)
            % SETCURVEWIDTH  Update the layered stroke width of a curve.
            arguments
                obj
                edgeIdx (1,1) double
                width   (1,1) double
            end
            obj.CurveWidths_{edgeIdx} = width;
            if edgeIdx <= numel(obj.CurveHandles_) && ...
                    ~isempty(obj.CurveHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHandles_{edgeIdx})
                set(obj.CurveHandles_{edgeIdx}, 'LineWidth', width);
                
                % Z-index layering based on thickness (thicker = bottom, thinner = top)
                % Wait, MATLAB draws the last element on top. So thinner lines should be re-plotted
                % to visually sit on top. We can accomplish this dynamically:
                uistack(obj.CurveHandles_{edgeIdx}, 'top');
            end
            obj.bringStationsToFront();
        end

        function toggleLabels(obj, visible)
            % TOGGLELABELS  Show or hide all station labels.
            arguments
                obj
                visible (1,1) logical
            end
            obj.LabelsVisible_ = visible;
            % Re-apply focus state to respect both master toggle and focus mode
            obj.applyFocus(obj.IsFocus_, obj.FocusCode_, obj.FocusNetwork_);
        end

        function toggleStations(obj, visible)
            % TOGGLESTATIONS  Show or hide all mathematical station markers visually.
            arguments
                obj
                visible (1,1) logical
            end
            obj.StationsVisible_ = visible;
            % Re-apply focus state to respect both master toggle and focus mode
            obj.applyFocus(obj.IsFocus_, obj.FocusCode_, obj.FocusNetwork_);
        end

        function applyFocus(obj, isFocus, activeCode, transitNetwork)
            % APPLYFOCUS  Hides everything except the active line if isFocus evaluates true.
            
            % Save state
            obj.IsFocus_ = isFocus;
            obj.FocusCode_ = activeCode;
            obj.FocusNetwork_ = transitNetwork;
            
            % Master visibility toggles
            sVis = 'on';
            if ~obj.StationsVisible_, sVis = 'off'; end
            lVis = 'on';
            if ~obj.LabelsVisible_, lVis = 'off'; end
            
            % If focus mode is off, or active code is invalid, instantly show all based on master toggle
            if ~obj.IsFocus_ || isempty(obj.FocusCode_) || isempty(obj.FocusNetwork_)
                for i = 1:numel(obj.CurveHandles_)
                    if ~isempty(obj.CurveHandles_{i}) && isvalid(obj.CurveHandles_{i})
                        set(obj.CurveHandles_{i}, 'Visible', 'on');
                    end
                    if i <= numel(obj.CurveHaloHandles_) && ~isempty(obj.CurveHaloHandles_{i}) && isvalid(obj.CurveHaloHandles_{i})
                        set(obj.CurveHaloHandles_{i}, 'Visible', 'on');
                    end
                end
                for i = 1:numel(obj.StationHandles_)
                    if ~isempty(obj.StationHandles_{i}) && isvalid(obj.StationHandles_{i})
                        set(obj.StationHandles_{i}, 'Visible', sVis);
                    end
                end
                for i = 1:numel(obj.LabelHandles_)
                    if ~isempty(obj.LabelHandles_{i}) && isvalid(obj.LabelHandles_{i})
                        actualLVis = lVis;
                        if strcmp(lVis, 'on') && ~obj.ZoomedIn_
                            if i > numel(obj.LabelIsMajor_) || ~obj.LabelIsMajor_(i)
                                actualLVis = 'off';
                            end
                        end
                        set(obj.LabelHandles_{i}, 'Visible', actualLVis);
                    end
                end
                return;
            end
            
            % FOCUS MODE ON
            lineObj = transitNetwork.getLine(activeCode);
            
            % 1. Filter Curves mathematically based on source/target membership
            for i = 1:numel(obj.CurveHandles_)
                if isempty(obj.CurveHandles_{i}) || ~isvalid(obj.CurveHandles_{i})
                    continue;
                end
                
                % Check if this edge is visually owned by the active line (Color ID mapping)
                e = transitNetwork.getRouteEdge(i);
                cVis = 'off';
                if ~isempty(e)
                    if i <= numel(obj.CurveColors_) && ~isempty(obj.CurveColors_{i})
                        % In the Designer, edge colors strictly map 1:1 to their creator line
                        % Tolerate extremely minor floating point differences from JSON serialization
                        c1 = obj.CurveColors_{i}(:)';
                        c2 = lineObj.Color(:)';
                        if norm(c1 - c2) < 1e-4
                            cVis = 'on';
                        end
                    end
                end
                
                set(obj.CurveHandles_{i}, 'Visible', cVis);
                if i <= numel(obj.CurveHaloHandles_) && ~isempty(obj.CurveHaloHandles_{i}) && isvalid(obj.CurveHaloHandles_{i})
                    set(obj.CurveHaloHandles_{i}, 'Visible', cVis);
                end
            end
            
            % 2. Filter Stations
            for i = 1:numel(obj.StationHandles_)
                
                isVisibleS = 'off';
                isVisibleL = 'off';
                
                if lineObj.hasStation(i)
                    isVisibleS = sVis;  % Show, obeying master toggle
                    isVisibleL = lVis;
                    if strcmp(isVisibleL, 'on') && ~obj.ZoomedIn_
                        if i > numel(obj.LabelIsMajor_) || ~obj.LabelIsMajor_(i)
                            isVisibleL = 'off';
                        end
                    end
                end
                
                if ~isempty(obj.StationHandles_{i}) && isvalid(obj.StationHandles_{i})
                    set(obj.StationHandles_{i}, 'Visible', isVisibleS);
                end
                if ~isempty(obj.LabelHandles_{i}) && isvalid(obj.LabelHandles_{i})
                    set(obj.LabelHandles_{i}, 'Visible', isVisibleL);
                end
            end
        end

        function setLabelFontSize(obj, sz)
            % SETLABELFONTSIZE  Set font size for all station labels.
            arguments
                obj
                sz (1,1) double {mustBePositive}
            end
            for i = 1:numel(obj.LabelHandles_)
                if ~isempty(obj.LabelHandles_{i}) && isvalid(obj.LabelHandles_{i})
                    set(obj.LabelHandles_{i}, 'FontSize', sz);
                end
            end
        end

        function bringStationsToFront(obj)
            % BRINGSTATIONSTOFRONT  Move station markers and labels to top z-order.
            % Ensures stations are always clickable even when placed over curves.
            
            % Collect all graphics handles into a single array for O(1) bulk stacking
            handlesToPush = gobjects(0);
            
            for i = 1:numel(obj.StationHandles_)
                if ~isempty(obj.StationHandles_{i}) && isvalid(obj.StationHandles_{i}) && ...
                        strcmp(get(obj.StationHandles_{i}, 'Visible'), 'on')
                    handlesToPush(end+1) = obj.StationHandles_{i}; %#ok<AGROW>
                end
            end
            
            for i = 1:numel(obj.LabelHandles_)
                if ~isempty(obj.LabelHandles_{i}) && isvalid(obj.LabelHandles_{i}) && ...
                        strcmp(get(obj.LabelHandles_{i}, 'Visible'), 'on')
                    handlesToPush(end+1) = obj.LabelHandles_{i}; %#ok<AGROW>
                end
            end
            
            if ~isempty(handlesToPush)
                uistack(handlesToPush, 'top');
            end
        end

        function deleteCurve(obj, edgeIdx)
            % DELETECURVE  Remove a curve and its control points visually.
            % Delete halo
            if edgeIdx <= numel(obj.CurveHaloHandles_) && ...
                    ~isempty(obj.CurveHaloHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHaloHandles_{edgeIdx})
                delete(obj.CurveHaloHandles_{edgeIdx});
                obj.CurveHaloHandles_{edgeIdx} = [];
            end
            % Delete color line
            if edgeIdx <= numel(obj.CurveHandles_) && ...
                    ~isempty(obj.CurveHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHandles_{edgeIdx})
                delete(obj.CurveHandles_{edgeIdx});
                obj.CurveHandles_{edgeIdx} = [];
            end
            if edgeIdx <= numel(obj.CurveColors_)
                obj.CurveColors_{edgeIdx} = [];
            end
            if edgeIdx <= numel(obj.CurveDrawOrders_)
                obj.CurveDrawOrders_{edgeIdx} = [];
            end
            % Remove control points
            if edgeIdx <= numel(obj.ControlPointHandles_) && ...
                    ~isempty(obj.ControlPointHandles_{edgeIdx})
                for j = 1:numel(obj.ControlPointHandles_{edgeIdx})
                    delete(obj.ControlPointHandles_{edgeIdx}{j});
                end
                obj.ControlPointHandles_{edgeIdx} = [];
            end
        end

        function deleteStation(obj, nodeId)
            % DELETESTATION  Visually remove a station marker and label.
            if nodeId <= numel(obj.StationHandles_) && ...
                    ~isempty(obj.StationHandles_{nodeId}) && ...
                    isvalid(obj.StationHandles_{nodeId})
                delete(obj.StationHandles_{nodeId});
                obj.StationHandles_{nodeId} = [];
            end
            if nodeId <= numel(obj.LabelHandles_) && ...
                    ~isempty(obj.LabelHandles_{nodeId}) && ...
                    isvalid(obj.LabelHandles_{nodeId})
                delete(obj.LabelHandles_{nodeId});
                obj.LabelHandles_{nodeId} = [];
            end
        end

        function [edgeIdx, minDist] = findNearestCurve(obj, ray, threshold)
            % FINDNEARESTCURVE  Exact 3D Ray-to-Curve intersection detection.
            %   ray: 2x3 matrix [pFront; pBack] from CurrentPoint.
            %   Returns edgeIdx = 0 if no vertices fall within threshold.
            arguments
                obj
                ray  (2,3) double
                threshold (1,1) double = inf
            end
            edgeIdx = 0;
            minDist = inf;
            
            pFront = ray(1, :);
            vDir = ray(2, :) - pFront;
            vLen = norm(vDir);
            if vLen > 1e-6
                vDir = vDir / vLen;
            end
            
            for i = 1:numel(obj.CurveHandles_)
                h = obj.CurveHandles_{i};
                if isempty(h) || ~isvalid(h), continue; end
                
                xd = get(h, 'XData');
                yd = get(h, 'YData');
                zd = get(h, 'ZData');
                if isempty(zd), zd = zeros(size(xd)); end
                
                % Compute perpendicular distance from each vertex to the infinite ray
                % D = || (V - P0) - dot(V - P0, Dir)*Dir ||
                nPts = numel(xd);
                V = [xd(:), yd(:), zd(:)];
                
                % W is vectors from ray origin to vertices
                W = V - pFront; % N x 3
                
                % projections = dot(W, vDir)
                projs = W(:,1)*vDir(1) + W(:,2)*vDir(2) + W(:,3)*vDir(3);
                
                % orthogonal components: W_perp = W - proj * vDir
                W_perp = W - projs * vDir;
                dists = sqrt(sum(W_perp.^2, 2));
                
                d = min(dists);
                if d < minDist && d < threshold
                    minDist = d;
                    edgeIdx = i;
                end
            end
        end

        function dpArray = drawControlPoints(obj, edgeIdx, spline, onMoveCb)
            % DRAWCONTROLPOINTS  Draw draggable markers for spline control points.
            %
            %   onMoveCb = @(cpIdx, newPos) — called when a point is dragged
            ctrlPts = spline.getControlPoints();
            nPts = size(ctrlPts, 1);
            dpArray = cell(1, nPts);

            for i = 1:nPts
                dp = OceanViz.Designer.DraggablePoint(obj.Axes_, ctrlPts(i,:), ...
                    'Color', obj.ControlColor, ...
                    'MarkerSize', obj.ControlSize, ...
                    'Tag', sprintf('edge%d_cp%d', edgeIdx, i));

                % Ensure they spawn completely visible!
                dp.setVisible(true);
                
                % Closure captures edgeIdx and control point index
                cpIdx = i;
                dp.OnMove = @(newPos) onMoveCb(edgeIdx, cpIdx, newPos);
                dpArray{i} = dp;
            end
            obj.ControlPointHandles_{edgeIdx} = dpArray;
        end

        function updateControlPoints(obj, edgeIdx, spline)
            % UPDATECONTROLPOINTS  Reposition control point markers.
            if edgeIdx > numel(obj.ControlPointHandles_) || ...
                    isempty(obj.ControlPointHandles_{edgeIdx})
                return;
            end
            ctrlPts = spline.getControlPoints();
            dpArray = obj.ControlPointHandles_{edgeIdx};
            for i = 1:numel(dpArray)
                if i <= size(ctrlPts, 1)
                    dpArray{i}.setPosition(ctrlPts(i,:));
                end
            end
        end

        function destroyAllControlPoints(obj)
            % DESTROYALLCONTROLPOINTS  Delete all control points from memory to save Handle Graphics overhead.
            for i = 1:numel(obj.ControlPointHandles_)
                cpArr = obj.ControlPointHandles_{i};
                if ~isempty(cpArr)
                    for j = 1:numel(cpArr)
                        delete(cpArr{j});
                    end
                    obj.ControlPointHandles_{i} = [];
                end
            end
        end

        function clearAll(obj)
            % CLEARALL  Remove all rendered elements.
            for i = 1:numel(obj.StationHandles_)
                if ~isempty(obj.StationHandles_{i}) && isvalid(obj.StationHandles_{i})
                    delete(obj.StationHandles_{i});
                end
            end
            for i = 1:numel(obj.LabelHandles_)
                if ~isempty(obj.LabelHandles_{i}) && isvalid(obj.LabelHandles_{i})
                    delete(obj.LabelHandles_{i});
                end
            end
            for i = 1:numel(obj.CurveHaloHandles_)
                if ~isempty(obj.CurveHaloHandles_{i}) && isvalid(obj.CurveHaloHandles_{i})
                    delete(obj.CurveHaloHandles_{i});
                end
            end
            for i = 1:numel(obj.CurveHandles_)
                if ~isempty(obj.CurveHandles_{i}) && isvalid(obj.CurveHandles_{i})
                    delete(obj.CurveHandles_{i});
                end
            end
            for i = 1:numel(obj.ControlPointHandles_)
                if ~isempty(obj.ControlPointHandles_{i})
                    for j = 1:numel(obj.ControlPointHandles_{i})
                        delete(obj.ControlPointHandles_{i}{j});
                    end
                end
            end
            obj.StationHandles_ = {};
            obj.LabelHandles_ = {};
            obj.CurveHandles_ = {};
            obj.CurveHaloHandles_ = {};
            obj.ControlPointHandles_ = {};
            obj.LabelOffsets_ = {};
            obj.CurveColors_ = {};
            obj.CurveDrawOrders_ = {};
            obj.LabelIsMajor_ = logical([]);
        end

        function ensureHalo(obj, edgeIdx)
            % ENSUREHALO  Create a halo for a curve that just got DrawOrder > 0.
            if edgeIdx > numel(obj.CurveHandles_) || isempty(obj.CurveHandles_{edgeIdx})
                return;
            end
            % Already has halo? skip
            if edgeIdx <= numel(obj.CurveHaloHandles_) && ...
                    ~isempty(obj.CurveHaloHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHaloHandles_{edgeIdx})
                return;
            end
            h = obj.CurveHandles_{edgeIdx};
            xd = get(h, 'XData'); yd = get(h, 'YData'); zd = get(h, 'ZData');
            width = obj.CurveLineWidth;
            if edgeIdx <= numel(obj.CurveWidths_) && ~isempty(obj.CurveWidths_{edgeIdx})
                width = obj.CurveWidths_{edgeIdx};
            end
            haloW = width + obj.HaloExtraWidth;
            hold(obj.Axes_, 'on');
            hHalo = plot3(obj.Axes_, xd, yd, zd, '-', ...
                'Color', obj.CanvasBackground, ...
                'LineWidth', haloW, ...
                'HitTest', 'off', 'PickableParts', 'none');
            obj.CurveHaloHandles_{edgeIdx} = hHalo;
        end

        function removeHalo(obj, edgeIdx)
            % REMOVEHALO  Delete a halo when DrawOrder drops to 0.
            if edgeIdx <= numel(obj.CurveHaloHandles_) && ...
                    ~isempty(obj.CurveHaloHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHaloHandles_{edgeIdx})
                delete(obj.CurveHaloHandles_{edgeIdx});
                obj.CurveHaloHandles_{edgeIdx} = [];
            end
        end

        function setCurveDrawOrder(obj, edgeIdx, order)
            % SETCURVEDRAWORDER  Set the visual draw order for a curve.
            arguments
                obj
                edgeIdx (1,1) double
                order   (1,1) double = 0
            end
            obj.CurveDrawOrders_{edgeIdx} = order;
        end

        function order = getCurveDrawOrder(obj, edgeIdx)
            % GETCURVEDRAWORDER  Get the draw order of a curve.
            if edgeIdx <= numel(obj.CurveDrawOrders_) && ~isempty(obj.CurveDrawOrders_{edgeIdx})
                order = obj.CurveDrawOrders_{edgeIdx};
            else
                order = 0;
            end
        end

        function reorderCurves(obj)
            % REORDERCURVES  Ensure high-priority curves render on top.
            %   Physically redraws drawOrder>0 curves so they are the last
            %   graphics objects placed on the axes (guaranteed on top).
            %   Then brings station markers to the very front.
            
            for i = 1:numel(obj.CurveDrawOrders_)
                if isempty(obj.CurveDrawOrders_{i}) || obj.CurveDrawOrders_{i} <= 0
                    continue;
                end
                if i > numel(obj.CurveHandles_) || isempty(obj.CurveHandles_{i}) || ...
                        ~isvalid(obj.CurveHandles_{i})
                    continue;
                end
                
                % Extract data from existing handles
                h = obj.CurveHandles_{i};
                xd = get(h, 'XData');
                yd = get(h, 'YData');
                zd = get(h, 'ZData');
                col = obj.CurveColors_{i};
                w = obj.CurveWidths_{i};
                
                % Delete old halo and line
                if i <= numel(obj.CurveHaloHandles_) && ...
                        ~isempty(obj.CurveHaloHandles_{i}) && ...
                        isvalid(obj.CurveHaloHandles_{i})
                    delete(obj.CurveHaloHandles_{i});
                end
                delete(h);
                
                % Redraw halo (underneath)
                hold(obj.Axes_, 'on');
                haloW = w + obj.HaloExtraWidth;
                hHalo = plot3(obj.Axes_, xd, yd, zd, '-', ...
                    'Color', obj.CanvasBackground, ...
                    'LineWidth', haloW, ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.CurveHaloHandles_{i} = hHalo;
                
                % Redraw color line on top
                hNew = plot3(obj.Axes_, xd, yd, zd, '-', ...
                    'Color', col, ...
                    'LineWidth', w, ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.CurveHandles_{i} = hNew;
            end
            
            % Stations always on top of everything
            obj.bringStationsToFront();
        end
    end
end
