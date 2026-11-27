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
        %Gaphics Pointers
        Axes_
        StationHandles_     % cell array of plot handles for station dots
        LabelHandles_       % cell array of text handles for station names
        CurveHandles_       % cell array of line handles for spline curves
        CurveHaloHandles_   % cell array of thick background halo handles
        ControlPointHandles_ % cell array of DraggablePoint arrays

        %Visuaal Overrides
        LabelOffsets_       % cell array of [dx, dy] per node
        CurveColors_        % cell array of [r,g,b] per edge
        CurveWidths_        % cell array of scalar widths per edge
        CurveDrawOrders_    % cell array of integer draw orders per edge

        %Interaction Toggles
        LabelsVisible_  (1,1) logical = true
        StationsVisible_(1,1) logical = true
        
        IsFocus_        (1,1) logical = false
        FocusCode_      (1,:) char = ''
        FocusNetwork_   % OceanMath.Transit.TransitNetwork reference
    end

    properties
        LabelIsMajor_   % logical array mapping 1:N indicating topological priority
        ZoomedIn_       (1,1) logical = true
        ForceAllLabels_ (1,1) logical = false
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
            arguments
                ax
            end
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

            %Normal station
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
            
            %Station Label Position
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
                'PickableParts', 'none', ...
                'Clipping', 'on');
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
            %Memory the new offset
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
                
                %Override
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
            tVals = spline.parameterize(obj.CurveSamples); %Sample the curve
            pts = spline.evaluate(tVals);

            zVals = zeros(size(pts,1), 1);

            hold(obj.Axes_, 'on');
            
            % 1. Halo line (only for curves that need to visually pass over others)
            if options.DrawOrder > 0
                haloW = options.LineWidth + obj.HaloExtraWidth;
                hHalo = plot3(obj.Axes_, pts(:,1), pts(:,2), zVals - 0.1, '-', ...
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

            %Stateful Memory Arrays. For Focus Mode
            obj.CurveHandles_{edgeIdx} = h;
            obj.CurveColors_{edgeIdx} = options.Color;
            obj.CurveWidths_{edgeIdx} = options.LineWidth;
            obj.CurveDrawOrders_{edgeIdx} = options.DrawOrder;
        end

        function updateCurve(obj, edgeIdx, spline, options)
            % UPDATECURVE  Redraw an existing curve after control point edit.
            %   Preserves the original edge color and width.
            arguments
                obj
                edgeIdx (1,1) double
                spline
                options.Samples (1,1) double = obj.CurveSamples
                options.ZOffset (1,1) double = 0
            end
            tVals = spline.parameterize(options.Samples); %Sample for speed
            pts = spline.evaluate(tVals);
            
            width = obj.CurveLineWidth;
            if edgeIdx <= numel(obj.CurveWidths_) && ~isempty(obj.CurveWidths_{edgeIdx})
                width = obj.CurveWidths_{edgeIdx};
            end

            zVals = zeros(size(pts,1), 1);
            if size(pts, 2) >= 3, zVals = pts(:,3); end
            zVals = zVals + options.ZOffset;

            % Update halo handle. Overwrite
            if edgeIdx <= numel(obj.CurveHaloHandles_) && ...
                    ~isempty(obj.CurveHaloHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHaloHandles_{edgeIdx})
                haloW = width + obj.HaloExtraWidth;
                set(obj.CurveHaloHandles_{edgeIdx}, ...
                    'XData', pts(:,1), 'YData', pts(:,2), 'ZData', zVals - 0.1, ...
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
            obj.CurveColors_{edgeIdx} = color; %Record
            if edgeIdx <= numel(obj.CurveHandles_) && ...
                    ~isempty(obj.CurveHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHandles_{edgeIdx})
                set(obj.CurveHandles_{edgeIdx}, 'Color', color);
            end
        end

        function setHaloVisibility(obj, edgeIdx, isVisible)
            % SETHALOVISIBILITY Hide/Show halo dynamically for nested shared paths
            if edgeIdx <= numel(obj.CurveHaloHandles_) && ...
                    ~isempty(obj.CurveHaloHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHaloHandles_{edgeIdx})
                if isVisible
                    set(obj.CurveHaloHandles_{edgeIdx}, 'Visible', 'on');
                else
                    set(obj.CurveHaloHandles_{edgeIdx}, 'Visible', 'off');
                end
            end
        end

        function setCurveVisibility(obj, edgeIdx, isVisible)
            % SETCURVEVISIBILITY Hide/Show core curve dynamically
            if edgeIdx <= numel(obj.CurveHandles_) && ...
                    ~isempty(obj.CurveHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHandles_{edgeIdx})
                if isVisible
                    set(obj.CurveHandles_{edgeIdx}, 'Visible', 'on');
                else
                    set(obj.CurveHandles_{edgeIdx}, 'Visible', 'off');
                end
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
                
                [az, el] = view(obj.Axes_);
                if az == 0 && el == 90 %2D. Azimuth (left/right orbit) = 0, Elevation (up/down orbit) = 90
                    uistack(obj.CurveHandles_{edgeIdx}, 'top');
                end
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

        function toggleAllLabels(obj, visible)
            % TOGGLEALLLABELS  Force all labels to be visible, regardless of topological zoom priority.
            arguments
                obj
                visible (1,1) logical
            end
            obj.ForceAllLabels_ = visible;
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

                %Dynamic Zoom Level of Details (LOD) Decluttering Engine
                for i = 1:numel(obj.StationHandles_)
                    %Inheriting the God Toggle
                    actualSVis = sVis;
                    actualLVis = lVis;
                    %The Micro-Zoom Intercept. 

                    %If the camera is heavy zoom out, hide minor stations and labels
                    if (strcmp(actualLVis, 'on') || strcmp(actualSVis, 'on')) && ~obj.ZoomedIn_ && ~obj.ForceAllLabels_
                        if i > numel(obj.LabelIsMajor_) || ~obj.LabelIsMajor_(i)
                            actualLVis = 'off';
                        end
                    end
                    %Keep the stations icon visible
                    if ~isempty(obj.StationHandles_{i}) && isvalid(obj.StationHandles_{i})
                        set(obj.StationHandles_{i}, 'Visible', actualSVis);
                    end
                end
                for i = 1:numel(obj.LabelHandles_)
                    actualLVis = lVis;
                    if strcmp(actualLVis, 'on') && ~obj.ZoomedIn_ && ~obj.ForceAllLabels_
                        %If not zoom in, hide minor labels
                        if i > numel(obj.LabelIsMajor_) || ~obj.LabelIsMajor_(i)
                            actualLVis = 'off';
                        end
                    end
                    if ~isempty(obj.LabelHandles_{i}) && isvalid(obj.LabelHandles_{i})
                        set(obj.LabelHandles_{i}, 'Visible', actualLVis);
                    end
                end
                return;
            end
            
            % FOCUS MODE ON
            %Extracting the true Mathematical line object
            lineObj = transitNetwork.getLine(activeCode);
            
            % 1. Filter Curves mathematically based on source/target membership
            for i = 1:numel(obj.CurveHandles_)
                if isempty(obj.CurveHandles_{i}) || ~isvalid(obj.CurveHandles_{i})
                    continue;
                end
                
                % Check if this edge is visually owned by the active line (Color ID mapping)
                e = transitNetwork.getRouteEdge(i);
                %Turn off all first
                cVis = 'off';
                if ~isempty(e)
                    if i <= numel(obj.CurveColors_) && ~isempty(obj.CurveColors_{i})
                        % In the Designer, edge colors strictly map 1:1 to their creator line
                        % Tolerate extremely minor floating point differences from JSON serialization
                        %Hex Color Hack instead of Adjacency Grapth Traversal
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
                    if strcmp(isVisibleL, 'on') && ~obj.ZoomedIn_ && ~obj.ForceAllLabels_
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
            
            % BUGFIX: If in 3D mode, uistack destroys the OpenGL Z-buffer depth sorting.
            % Rely on actual Z-coordinates in 3D; abort 2D uistack completely!
            [az, el] = view(obj.Axes_);
            if az ~= 0 || el ~= 90
                return;
            end
            
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
            
            %Bulk Hardware Refresh
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


        %RayCasting / Ray Picking
        function [edgeIdx, minDist] = findNearestCurve(obj, ray, threshold)
            % FINDNEARESTCURVE  Exact 3D Ray-to-Curve intersection detection.
            %   ray: 2x3 matrix [pFront; pBack] from CurrentPoint.
            %   Returns edgeIdx = 0 if no vertices fall within threshold.
            %
            %   Optimization: AABB early-rejection skips curves whose
            %   bounding boxes are completely outside the search region.
            arguments
                obj
                ray  (2,3) double
                threshold (1,1) double = inf
            end
            edgeIdx = 0;
            minDist = inf;
            pBack = ray(2, :);
            pFront = ray(1, :);
            vDir = pBack - pFront;
            vLen = norm(vDir);
            if vLen > 1e-6
                vDir = vDir / vLen;
            end

            % AABB pre-filter: only valid in top-down 2D (camera elevation = 90).
            % In 3D, projecting the ray to Z=0 gives a wrong XY hitPt when
            % the click is over a high-Z part of the curve, causing false rejection.
            % In 3D we skip the AABB and rely entirely on the perpendicular
            % ray-distance math below, which is correct at any camera angle.
            [~, el] = view(obj.Axes_);
            isTopDown = (abs(el) >= 89.9);
            if isTopDown
                if abs(vDir(3)) > 1e-6
                    tHit = (0 - pFront(3)) / vDir(3);
                    hitPt = pFront + tHit * (vDir * vLen);
                else
                    hitPt = pFront;
                end
                searchBox = [hitPt(1) - threshold, hitPt(2) - threshold, ...
                             hitPt(1) + threshold, hitPt(2) + threshold];
            end

            for i = 1:numel(obj.CurveHandles_)
                h = obj.CurveHandles_{i};
                if isempty(h) || ~isvalid(h), continue; end

                xd = get(h, 'XData');
                yd = get(h, 'YData');
                zd = get(h, 'ZData');
                if isempty(zd), zd = zeros(size(xd)); end

                % AABB early-rejection (2D top-down only — see comment above)
                if isTopDown
                    curveBox = [min(xd), min(yd), max(xd), max(yd)];
                    if curveBox(1) > searchBox(3) || curveBox(3) < searchBox(1) || ...
                       curveBox(2) > searchBox(4) || curveBox(4) < searchBox(2)
                        continue;  % Entire curve is outside search radius
                    end
                end
                
                % Compute perpendicular distance from each vertex to the infinite ray
                % D = || (V - P0) - dot(V - P0, Dir)*Dir ||
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

        function dpArray = drawControlPoints(obj, edgeIdx, spline, onGrabCb, onMoveCb, onReleaseCb)
            % DRAWCONTROLPOINTS  Draw draggable markers for spline control points.
            %   onGrabCb = @(edgeIdx, cpIdx, startPos) — called when drag begins
            %   onMoveCb = @(edgeIdx, cpIdx, newPos) — called when a point is dragged
            %   onReleaseCb = @(edgeIdx, cpIdx, newPos) — called when drag completes
            ctrlPts = spline.getControlPoints(); %Get control points, 4 for Catmull
            nPts = size(ctrlPts, 1);
            dpArray = cell(1, nPts);

            for i = 1:nPts
                dp = OceanViz.Designer.Interaction.DraggablePoint(obj.Axes_, ctrlPts(i,:), ...
                    'Color', obj.ControlColor, ...
                    'MarkerSize', obj.ControlSize, ...
                    'Tag', sprintf('edge%d_cp%d', edgeIdx, i));

                % Ensure they spawn completely visible!
                dp.setVisible(true);
                
                % Closure captures edgeIdx and control point index
                %Force creating of memory to avoid Late Binding Closure Bug
                cpIdx = i;
                if nargin >= 4 && ~isempty(onGrabCb)
                    dp.OnGrab = @(startPos) onGrabCb(edgeIdx, cpIdx, startPos);
                end
                if nargin >= 5 && ~isempty(onMoveCb)
                    dp.OnMove = @(newPos) onMoveCb(edgeIdx, cpIdx, newPos);
                end
                if nargin >= 6 && ~isempty(onReleaseCb)
                    dp.OnRelease = @(newPos) onReleaseCb(edgeIdx, cpIdx, newPos);
                end
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
            %Ask the Math engine for control points
            ctrlPts = spline.getControlPoints();
            %dp(Draggable Points) = OceanViz.Designer.Interaction.DraggablePoint
            dpArray = obj.ControlPointHandles_{edgeIdx};
            for i = 1:numel(dpArray)
                if i <= size(ctrlPts, 1)
                    %Write directly in the Register memory to avoid overhead
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
            % CLEARALL  Remove all rendered elements using O(1) vectorized blocks.
            
            h = [obj.StationHandles_{:}];
            if ~isempty(h), delete(h(isvalid(h))); end
            
            h = [obj.LabelHandles_{:}];
            if ~isempty(h), delete(h(isvalid(h))); end
            
            h = [obj.CurveHaloHandles_{:}];
            if ~isempty(h), delete(h(isvalid(h))); end
            
            h = [obj.CurveHandles_{:}];
            if ~isempty(h), delete(h(isvalid(h))); end

            for i = 1:numel(obj.ControlPointHandles_)
                if ~isempty(obj.ControlPointHandles_{i})
                    h = [obj.ControlPointHandles_{i}{:}];
                    if ~isempty(h), delete(h(isvalid(h))); end
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


        %Dynamic Overpass Engine
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

        function reorderCurveLocal(obj, edgeIdx, direction)
            % REORDERCURVELOCAL  O(1) Z-depth shifting for a single curve.
            %   Natively shifts the graphics object in the GPU buffer instead
            %   of computationally destroying and re-drawing the curve.
            if nargin < 3
                direction = 'top';
            end
            
            if edgeIdx > numel(obj.CurveHandles_) || isempty(obj.CurveHandles_{edgeIdx}) || ...
                    ~isvalid(obj.CurveHandles_{edgeIdx})
                return;
            end
            
            h = obj.CurveHandles_{edgeIdx};
            
            % Push track
            uistack(h, direction);
            
            % If it has a glowing halo, it must sit exactly underneath the track
            if edgeIdx <= numel(obj.CurveHaloHandles_) && ...
                    ~isempty(obj.CurveHaloHandles_{edgeIdx}) && ...
                    isvalid(obj.CurveHaloHandles_{edgeIdx})
                
                hHalo = obj.CurveHaloHandles_{edgeIdx};
                uistack(hHalo, direction); % push halo
                if strcmp(direction, 'top')
                    uistack(h, direction); % push track OVER the halo if coming to front
                else
                    uistack(hHalo, direction); % push halo UNDER the track if going to back
                end
            end
            
            % Cap the visual hierarchy: station dots must ALWAYS float above track pixels
            obj.bringStationsToFront();
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
