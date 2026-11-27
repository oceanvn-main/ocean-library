classdef TrackDesigner < handle
    % TRACKDESIGNER  Interactive GUI for designing transit tracks.
    %
    % Provides a figure-based canvas where the user can:
    %   1. Place stations by clicking (place mode)
    %   2. Connect stations with CatmullRom spline edges (connect mode)
    %   3. Drag control points to reshape curves in real-time (edit mode)
    %   4. Export the design as a NetworkGraph with spline arc-length weights
    %   5. Save and load designs as .mat files
    %
    % Usage:
    %   td = OceanViz.Designer.TrackDesigner();
    %   td.launch();
    %
    %   % Or programmatic use (no GUI):
    %   td = OceanViz.Designer.TrackDesigner();
    %   td.addStation('Orchard', [3, 7]);
    %   td.addStation('Dhoby Ghaut', [4, 6]);
    %   td.connectStations(1, 2);
    %   g = td.exportGraph();
    %
    % See also: OceanMath.Transit.TransitNetwork
    %           OceanMath.Graph.DigraphCus
    %           OceanMath.Geometry.CatmullRomSpline
    %           OceanViz.Designer.TrackRenderer
    %           OceanViz.Designer.DraggablePoint

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % Core data
        Transit_        % OceanMath.Transit.TransitNetwork
        StationCount_   (1,1) double = 0
        EdgeCount_      (1,1) double = 0

        % GUI handles
        Figure_
        Axes_
        Renderer_       % TrackRenderer instance
        StatusText_     % status bar text handle
        ToolButtons_    % struct with button handles
        
        % Interaction state
        Mode_           (1,:) char = 'place'
        ConnectSource_  (1,1) double = 0  % first station clicked in connect mode
        SelectedStation_ (1,1) double = 0 % currently selected station in edit mode

        LinePanel_      (1,1) struct
        PropPanel_      (1,1) struct

        ActiveLineCode_ (1,:) char = ''
        
        WorkingPos_      (1,3) double = [0, 0, 0] % Caches last edited position for 3D placement
        ActiveEditCurve_ (1,1) double = 0  % Only show listeners for the currently selected curve
        % Phase 3: Curve type
        CurveType_      (1,:) char = 'catmullrom'  % 'catmullrom' or 'bezier'
        FocusCheckbox_  % UI handle for Focus Mode

        % Pan/zoom state
        PanOrigin_      (1,2) double = [0 0]  % mouse start for panning
        IsPanning_      (1,1) logical = false

        % Undo stack (stores .mat design snapshots)
        UndoStack_      cell = {}
        LegendHandles_  cell = {}  % Handles for bottom legend annotations
        LegendPos1_     (1,2) double = [NaN NaN]  % Train Services box position
        LegendPos2_     (1,2) double = [NaN NaN]  % Station Types + Path Notes box position
        DraggingLegend_ (1,1) double = 0           % Which box is being dragged (0=none, 1 or 2)
        DragOffset_     (1,2) double = [0 0]       % Mouse offset from box origin

        % Weight display
        WeightHandles_  cell = {}    % text handles for edge weight labels
        ShowWeights_    (1,1) logical = false
        WeightUnit_     (1,:) char = 'km'  % 'km' or 'mi'

        % Map title (draggable)
        TitleHandle_               % text handle for map title
        TitlePos_       (1,2) double = [NaN NaN]
        MapTitle_       = {'Singapore Mass Rapid Transit', '& Light Rapid Transit System 2035'}
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = TrackDesigner()
            % TRACKDESIGNER  Construct (does NOT open GUI yet).
            obj.Transit_ = OceanMath.Transit.TransitNetwork();
            obj.LinePanel_ = struct('init_', true);
        end

        % ============================================================== %
        %  GUI Launch
        % ============================================================== %

        function launch(obj)
            % LAUNCH  Open the interactive designer window.
            obj.Figure_ = figure('Name', 'Track Designer', ...
                'NumberTitle', 'off', ...
                'Color', [0.12, 0.12, 0.14], ...
                'MenuBar', 'none', ...
                'ToolBar', 'none', ...
                'WindowState', 'maximized', ...
                'CloseRequestFcn', @(~,~) obj.onClose_());
                
            % Force MATLAB UI thread to handshake and initialize the JS renderer immediately
            drawnow;
            pause(0.1);

            % Main axes — maximized, limits set dynamically after data load
            obj.Axes_ = axes(obj.Figure_, ...
                'Position', [0.01, 0.01, 0.71, 0.925], ...
                'Color', [1, 1, 1], ...
                'XColor', [0.6, 0.6, 0.6], ...
                'YColor', [0.6, 0.6, 0.6], ...
                'GridColor', [0.85, 0.85, 0.85], ...
                'Box', 'on');
            set(obj.Axes_, 'LooseInset', [0 0 0 0]);
            hold(obj.Axes_, 'on');
            grid(obj.Axes_, 'on');

            % Renderer
            obj.Renderer_ = OceanViz.Designer.TrackRenderer(obj.Axes_);

            % Toolbar buttons
            obj.createToolbar_();

            % Property panel (right side)
            obj.createPropertyPanel_();

            % Status bar — placed at bottom of right panel
            obj.StatusText_ = uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Mode: PLACE  |  Click to add stations', ...
                'Units', 'normalized', ...
                'Position', [0.73, 0.01, 0.26, 0.035], ...
                'BackgroundColor', [0.15, 0.15, 0.18], ...
                'ForegroundColor', [0.8, 0.95, 1.0], ...
                'FontSize', 9, ...
                'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');

            % Mouse click callback on axes
            set(obj.Axes_, 'ButtonDownFcn', @(~,~) obj.onAxesClick_());

            % Scroll wheel zoom (zoom toward cursor)
            set(obj.Figure_, 'WindowScrollWheelFcn', @(~,evt) obj.onScrollZoom_(evt));

            % Right-click drag to pan
            set(obj.Figure_, 'WindowButtonDownFcn',   @(~,~) obj.onPanStart_());
            set(obj.Figure_, 'WindowButtonUpFcn',     @(~,~) obj.onPanEnd_());
            
            % Listen for zoom/pan scaling
            addlistener(obj.Axes_, 'XLim', 'PostSet', @(~,~) obj.onZoomCheck_());

            % Ctrl+Z for undo
            set(obj.Figure_, 'KeyPressFcn', @(~,evt) obj.onKeyPress_(evt));

            obj.updateStatus_();
        end

        % ============================================================== %
        %  Programmatic API (works with or without GUI)
        % ============================================================== %

        function id = addStation(obj, name, position, options)
            % ADDSTATION  Add a station to the track.
            %
            %   id = td.addStation('Orchard', [3, 7, 0])
            arguments
                obj
                name     (1,:) char
                position (1,:) double
                options.SuspendGraphics (1,1) logical = false
            end
            id = obj.Transit_.addStation(name, position);
            obj.StationCount_ = obj.StationCount_ + 1;
            
            % Update working pos cache
            pos3 = [position(1), position(2), 0];
            if length(position) >= 3, pos3(3) = position(3); end
            obj.WorkingPos_ = pos3;

            % Render if GUI is open
            if ~isempty(obj.Renderer_)
                node = obj.Transit_.getStation(id);
                obj.Renderer_.drawStation(node, @(nId) obj.onStationClick_(nId));
                
                if ~options.SuspendGraphics
                    % Bring station markers to front so they're always clickable
                    obj.Renderer_.bringStationsToFront();
                    % Keep axis limits stable — pad around all stations
                    obj.updateAxisLimits_();
                end
            end
        end

        function edgeIdx = connectStations(obj, id1, id2, options)
            % CONNECTSTATIONS  Create a spline edge between two stations.
            %
            %   td.connectStations(1, 2)
            %   td.connectStations(1, 2, 'Label', 'North-South Line')
            %   td.connectStations(1, 2, 'NumMidpoints', 3)
            arguments
                obj
                id1     (1,1) double {mustBePositive, mustBeInteger}
                id2     (1,1) double {mustBePositive, mustBeInteger}
                options.Label (1,:) char = ''
                options.NumMidpoints (1,1) double {mustBeNonnegative, mustBeInteger} = 1
            end
            
            obj.pushUndo_();

            pos1 = obj.Transit_.getStation(id1).Position;
            pos2 = obj.Transit_.getStation(id2).Position;

            % Generate intermediate control points (evenly spaced + slight offset)
            nMid = options.NumMidpoints;
            
            % Ensure pos are 1x3
            if length(pos1) < 3, pos1 = [pos1(1), pos1(2), 0]; end
            if length(pos2) < 3, pos2 = [pos2(1), pos2(2), 0]; end
            
            controlPts = zeros(nMid + 2, 3);
            controlPts(1, :)   = pos1;
            controlPts(end, :) = pos2;

            for k = 1:nMid
                frac = k / (nMid + 1);
                mid = pos1 + frac * (pos2 - pos1);
                % Add a small perpendicular offset for visual curvature
                dir = pos2 - pos1;
                perp = [-dir(2), dir(1), 0];
                perpNorm = sqrt(sum(perp.^2));
                if perpNorm > 1e-12
                    perp = perp / perpNorm;
                end
                offset = 0.3 * perpNorm * sin(pi * frac);
                controlPts(k+1, :) = mid + offset * perp;
            end

            % Build spline from control points
            switch obj.CurveType_
                case 'linear'
                    spline = OceanMath.Geometry.BezierCurve([pos1; pos2]);
                case 'bezier'
                    spline = OceanMath.Geometry.BezierCurve(controlPts);
                otherwise % 'catmullrom'
                    spline = OceanMath.Geometry.CatmullRomSpline(controlPts);
            end
            obj.EdgeCount_ = obj.EdgeCount_ + 1;
            edgeIdx = obj.EdgeCount_;

            % Determine edge color from active line
            if ~isempty(obj.ActiveLineCode_) && any(strcmp(obj.Transit_.getLineNames(), obj.ActiveLineCode_))
                lineObj = obj.Transit_.getLine(obj.ActiveLineCode_);
                edgeColor = lineObj.Color;
                % Register stations on the line
                lineObj.registerStation(id1);
                lineObj.registerStation(id2);
                obj.updateStationVisuals_(id1);
                obj.updateStationVisuals_(id2);
            else
                edgeColor = [0.4, 0.8, 1.0]; % default cyan
            end

            % Add to transit network (weight auto-computed from spline)
            obj.Transit_.addRoute(id1, id2, spline, ...
                'Label', options.Label);

            % Render if GUI is open
            if ~isempty(obj.Renderer_)
                obj.Renderer_.drawCurve(edgeIdx, spline, 'Color', edgeColor);
            end
        end

        function g = exportGraph(obj)
            % EXPORTGRAPH  Return the current NetworkGraph.
            g = obj.Transit_.getGraph();
        end

        function tn = exportTransitNetwork(obj)
            % EXPORTTRANSITNETWORK  Return the TransitNetwork wrapper.
            tn = obj.Transit_;
        end

        function s = getDesignStruct_(obj)
            % GETDESIGNSTRUCT_  Package the entire graph + UI state into a memory struct.
            s = struct();
            s.graphStruct = obj.Transit_.toStruct();
            
            % Save UI visual state that the abstract math backend intentionally drops
            uiState = struct();
            if ~isempty(obj.Renderer_)
                % Compact CurveColors and CurveWidths strictly matching the compacted Edges array
                compactColors = {};
                compactWidths = {};
                compactDrawOrders = {};
                for i = 1:obj.Transit_.getGraph().getEdgeCount()
                    if ~isempty(obj.Transit_.getGraph().getEdge(i))
                        % Curve Colors
                        if i <= numel(obj.Renderer_.CurveColors_) && ~isempty(obj.Renderer_.CurveColors_{i})
                            compactColors{end+1} = obj.Renderer_.CurveColors_{i}; %#ok<AGROW>
                        else
                            compactColors{end+1} = [0.4 0.8 1.0]; %#ok<AGROW>
                        end
                        % Curve Widths
                        if i <= numel(obj.Renderer_.CurveWidths_) && ~isempty(obj.Renderer_.CurveWidths_{i})
                            compactWidths{end+1} = obj.Renderer_.CurveWidths_{i}; %#ok<AGROW>
                        else
                            compactWidths{end+1} = obj.Renderer_.CurveLineWidth; %#ok<AGROW>
                        end
                        % Draw Orders
                        compactDrawOrders{end+1} = obj.Renderer_.getCurveDrawOrder(i); %#ok<AGROW>
                    end
                end
                uiState.CurveColors = compactColors;
                uiState.CurveWidths = compactWidths;
                uiState.CurveDrawOrders = compactDrawOrders;
                
                % Track visually deleted stations so they don't resurrect
                deletedStations = [];
                for i = 1:obj.StationCount_
                    if isempty(obj.Renderer_.StationHandles_) || i > numel(obj.Renderer_.StationHandles_) ...
                            || isempty(obj.Renderer_.StationHandles_{i}) || ~isvalid(obj.Renderer_.StationHandles_{i})
                        deletedStations(end+1) = i; %#ok<AGROW>
                    end
                end
                uiState.DeletedStations = deletedStations;
                
            else
                uiState.CurveColors = {};
                uiState.CurveWidths = {};
                uiState.CurveDrawOrders = {};
                uiState.LabelOffsets = {};
                uiState.DeletedStations = [];
            end
            s.uiState = uiState;
        end

        function saveDesign(obj, filepath)
            % SAVEDESIGN  Save the track design to a .mat file.
            arguments
                obj
                filepath (1,:) char
            end
            s = obj.getDesignStruct_();
            graphStruct = s.graphStruct;
            uiState = s.uiState;
            save(filepath, 'graphStruct', 'uiState');
        end

        function loadDesignStruct_(obj, dataStr)
            % LOADDESIGNSTRUCT_  Rebuild the entire graph + UI state from a memory struct.
            if ~isfield(dataStr, 'graphStruct')
                return;
            end
            
            % Clear current state
            obj.Transit_ = OceanMath.Transit.TransitNetwork();
            obj.StationCount_ = 0;
            obj.EdgeCount_ = 0;
            if ~isempty(obj.Renderer_)
                obj.Renderer_.clearAll();
            end

            % Rebuild nodes
            if ~isempty(dataStr.graphStruct.Nodes)
                obj.Transit_.reserveStations(numel(dataStr.graphStruct.Nodes));
            end
            for i = 1:numel(dataStr.graphStruct.Nodes)
                n = dataStr.graphStruct.Nodes{i};
                obj.addStation(n.Name, n.Position, 'SuspendGraphics', true);
                
                % If it was supposed to be dead, brutally delete the handles that addStation just created
                if isfield(dataStr, 'uiState') && isfield(dataStr.uiState, 'DeletedStations') && ...
                        ismember(i, dataStr.uiState.DeletedStations)
                    if ~isempty(obj.Renderer_)
                        obj.Renderer_.deleteStation(i);
                    end
                end
                
                % Restore label offset if available
                if isfield(dataStr, 'uiState') && isfield(dataStr.uiState, 'LabelOffsets') && ...
                        i <= numel(dataStr.uiState.LabelOffsets) && ~isempty(dataStr.uiState.LabelOffsets{i})
                    obj.Renderer_.setLabelOffset(i, dataStr.uiState.LabelOffsets{i});
                end
            end

            % Rebuild edges with splines
            for i = 1:numel(dataStr.graphStruct.Edges)
                e = dataStr.graphStruct.Edges{i};
                splineCtrl = dataStr.graphStruct.SplineData{i};
                if ~isempty(splineCtrl)
                    spline = OceanMath.Geometry.CatmullRomSpline(splineCtrl);
                    obj.EdgeCount_ = obj.EdgeCount_ + 1;
                    obj.Transit_.addRoute(e.Source, e.Target, spline, ...
                        'Label', e.Label);
                    if ~isempty(obj.Renderer_)
                        % Restore edge curve color
                        c = [0.4 0.8 1.0];
                        if isfield(dataStr, 'uiState') && isfield(dataStr.uiState, 'CurveColors') && ...
                                i <= numel(dataStr.uiState.CurveColors) && ~isempty(dataStr.uiState.CurveColors{i})
                            c = dataStr.uiState.CurveColors{i};
                        end
                        % Restore draw order
                        dOrder = 0;
                        if isfield(dataStr, 'uiState') && isfield(dataStr.uiState, 'CurveDrawOrders') && ...
                                i <= numel(dataStr.uiState.CurveDrawOrders) && ~isempty(dataStr.uiState.CurveDrawOrders{i})
                            dOrder = dataStr.uiState.CurveDrawOrders{i};
                        end
                        obj.Renderer_.drawCurve(obj.EdgeCount_, spline, 'Color', c, 'DrawOrder', dOrder);
                        obj.Renderer_.drawControlPoints(obj.EdgeCount_, spline, ...
                            @(eIdx, cpIdx, newPos) obj.onControlPointMoved_(eIdx, cpIdx, newPos));
                    end
                else
                    obj.EdgeCount_ = obj.EdgeCount_ + 1;
                    obj.Transit_.addSimpleRoute(e.Source, e.Target, e.Weight, ...
                        'Label', e.Label);
                end
            end
            
            % Rebuild transit lines
            if isfield(dataStr.graphStruct, 'LineData')
                for i = 1:numel(dataStr.graphStruct.LineData)
                    ld = dataStr.graphStruct.LineData{i};
                    obj.Transit_.addLine(ld.Code, ld.Name, ld.Color);
                    lineObj = obj.Transit_.getLine(ld.Code);
                    
                    for j = 1:numel(ld.StationIds)
                        lineObj.registerStation(ld.StationIds(j));
                    end
                    
                    if isfield(ld, 'Policy')
                        pKeys = fieldnames(ld.Policy);
                        for k = 1:numel(pKeys)
                            lineObj.setPolicy(pKeys{k}, ld.Policy.(pKeys{k}));
                        end
                    end
                end
                obj.refreshLineDropdown_();
            end
            
            % Restore interchange visuals globally (Fix for disappearing white circles)
            for i = 1:obj.StationCount_
                obj.updateStationVisuals_(i);
            end
            if ~isempty(obj.Renderer_)
                obj.Renderer_.reorderCurves();
            end
        end

        function loadDesign(obj, filepath)
            % LOADDESIGN  Load a track design from a .mat file.
            arguments
                obj
                filepath (1,:) char
            end
            dataStr = load(filepath);
            obj.loadDesignStruct_(dataStr);
        end

        function describe(obj)
            % DESCRIBE  Print summary.
            fprintf('TrackDesigner:\n');
            fprintf('  Stations: %d\n', obj.StationCount_);
            fprintf('  Edges:    %d\n', obj.EdgeCount_);
            fprintf('  Mode:     %s\n', obj.Mode_);
            obj.Transit_.describe();
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)

        % ---- Toolbar ---- %
        function createToolbar_(obj)
            % Compact toolbar — two rows of small buttons
            btnW = 0.055;
            btnH = 0.025;
            gap  = 0.003;
            startX = 0.01;
            
            yTop = 0.970;
            yBot = 0.940;

            colors = struct('bg', [0.2 0.2 0.25], 'fg', [0.8 0.9 1.0]);
            
            % ---- ROW 1: Editing Mode Tools ----
            x = startX;

            obj.ToolButtons_.place = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Place', ...
                'Units', 'normalized', ...
                'Position', [x, yTop, btnW, btnH], ...
                'BackgroundColor', [0.3 0.5 0.8], ...
                'ForegroundColor', [1 1 1], ...
                'Callback', @(~,~) obj.setMode_('place'));
            x = x + btnW + gap;

            obj.ToolButtons_.connect = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Connect', ...
                'Units', 'normalized', ...
                'Position', [x, yTop, btnW, btnH], ...
                'BackgroundColor', colors.bg, ...
                'ForegroundColor', colors.fg, ...
                'Callback', @(~,~) obj.setMode_('connect'));
            x = x + btnW + gap;

            obj.ToolButtons_.edit = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Edit', ...
                'Units', 'normalized', ...
                'Position', [x, yTop, btnW, btnH], ...
                'BackgroundColor', colors.bg, ...
                'ForegroundColor', colors.fg, ...
                'Callback', @(~,~) obj.setMode_('edit'));
            x = x + btnW + gap;

            obj.ToolButtons_.delete = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Delete', ...
                'Units', 'normalized', ...
                'Position', [x, yTop, btnW, btnH], ...
                'BackgroundColor', [0.7 0.2 0.2], ...
                'ForegroundColor', [1 1 1], ...
                'Callback', @(~,~) obj.setMode_('delete'));
            x = x + btnW + gap;

            obj.ToolButtons_.insert = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Insert', ...
                'Units', 'normalized', ...
                'Position', [x, yTop, btnW, btnH], ...
                'BackgroundColor', [0.5 0.3 0.7], ...
                'ForegroundColor', [1 1 1], ...
                'Callback', @(~,~) obj.setMode_('insert'));
            x = x + btnW + gap;

            obj.ToolButtons_.undo = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Undo', ...
                'Units', 'normalized', ...
                'Position', [x, yTop, btnW, btnH], ...
                'BackgroundColor', [0.6 0.5 0.2], ...
                'ForegroundColor', [1 1 1], ...
                'Callback', @(~,~) obj.onUndo_());
            x = x + btnW + gap;

            obj.ToolButtons_.export = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Export', ...
                'Units', 'normalized', ...
                'Position', [x, yTop, btnW, btnH], ...
                'BackgroundColor', [0.2 0.6 0.3], ...
                'ForegroundColor', [1 1 1], ...
                'Callback', @(~,~) obj.onExport_());
            x = x + btnW + gap;

            obj.ToolButtons_.new = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'New', ...
                'Units', 'normalized', ...
                'Position', [x, yTop, btnW, btnH], ...
                'BackgroundColor', [0.6 0.3 0.3], ...
                'ForegroundColor', [1 1 1], ...
                'Callback', @(~,~) obj.onNew_());
            x = x + btnW + gap;

            obj.ToolButtons_.save = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Save', ...
                'Units', 'normalized', ...
                'Position', [x, yTop, btnW, btnH], ...
                'BackgroundColor', colors.bg, ...
                'ForegroundColor', colors.fg, ...
                'Callback', @(~,~) obj.onSave_());
            x = x + btnW + gap;

            obj.ToolButtons_.load = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Load', ...
                'Units', 'normalized', ...
                'Position', [x, yTop, btnW, btnH], ...
                'BackgroundColor', colors.bg, ...
                'ForegroundColor', colors.fg, ...
                'Callback', @(~,~) obj.onLoad_());

            % ---- ROW 2: Camera & Visual Overlays ----
            x = startX;

            obj.ToolButtons_.labels = uicontrol(obj.Figure_, ...
                'Style', 'togglebutton', 'String', 'Labels', ...
                'Units', 'normalized', ...
                'Position', [x, yBot, btnW, btnH], ...
                'BackgroundColor', [0.3 0.5 0.8], ...
                'ForegroundColor', [1 1 1], ...
                'Value', 1, ...
                'Callback', @(src,~) obj.onToggleLabels_(get(src, 'Value')));
            x = x + btnW + gap;
            
            obj.ToolButtons_.stations = uicontrol(obj.Figure_, ...
                'Style', 'togglebutton', 'String', 'Stations', ...
                'Units', 'normalized', ...
                'Position', [x, yBot, btnW, btnH], ...
                'BackgroundColor', [0.3 0.6 0.8], ...
                'ForegroundColor', [1 1 1], ...
                'Value', 1, ...
                'Callback', @(src,~) obj.onToggleStations_(get(src, 'Value')));
            x = x + btnW + gap;
            
            obj.ToolButtons_.cam3d = uicontrol(obj.Figure_, ...
                'Style', 'togglebutton', 'String', '3D Cam', ...
                'Units', 'normalized', ...
                'Position', [x, yBot, btnW, btnH], ...
                'BackgroundColor', [0.2 0.8 0.4], ...
                'ForegroundColor', [1 1 1], ...
                'Callback', @(src,~) obj.toggle3DView_(src));
            x = x + btnW + gap;
            
            % The 3 Core placement views
            obj.ToolButtons_.camTop = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Top View', ...
                'Units', 'normalized', ...
                'Position', [x, yBot, btnW, btnH], ...
                'BackgroundColor', [0.2 0.2 0.25], ...
                'ForegroundColor', [1 0.8 0.4], ...
                'Callback', @(~,~) view(obj.Axes_, 2));
            x = x + btnW + gap;
            
            obj.ToolButtons_.camFront = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Front View', ...
                'Units', 'normalized', ...
                'Position', [x, yBot, btnW, btnH], ...
                'BackgroundColor', [0.2 0.2 0.25], ...
                'ForegroundColor', [1 0.8 0.4], ...
                'Callback', @(~,~) view(obj.Axes_, [0, 0]));
            x = x + btnW + gap;
            
            obj.ToolButtons_.camSide = uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Side View', ...
                'Units', 'normalized', ...
                'Position', [x, yBot, btnW, btnH], ...
                'BackgroundColor', [0.2 0.2 0.25], ...
                'ForegroundColor', [1 0.8 0.4], ...
                'Callback', @(~,~) view(obj.Axes_, [90, 0]));
            x = x + btnW + gap;

            % Weight display toggles
            obj.ToolButtons_.weights = uicontrol(obj.Figure_, ...
                'Style', 'togglebutton', 'String', 'Weights', ...
                'Units', 'normalized', ...
                'Position', [x, yBot, btnW, btnH], ...
                'BackgroundColor', [0.2 0.2 0.25], ...
                'ForegroundColor', [1 1 1], ...
                'Value', 0, ...
                'Callback', @(src,~) obj.onToggleWeights_(get(src, 'Value')));
            x = x + btnW + gap;

            obj.ToolButtons_.weightUnit = uicontrol(obj.Figure_, ...
                'Style', 'togglebutton', 'String', 'km', ...
                'Units', 'normalized', ...
                'Position', [x, yBot, btnW * 0.6, btnH], ...
                'BackgroundColor', [0.2 0.2 0.25], ...
                'ForegroundColor', [0.4 0.9 0.4], ...
                'Value', 0, ...
                'Callback', @(src,~) obj.onToggleWeightUnit_(src));
        end

        % ---- Mode Switching ---- %
        function setMode_(obj, mode)
            obj.Mode_ = mode;
            obj.ConnectSource_ = 0;

            % Highlight active button
            allBtns = {'place', 'connect', 'edit', 'delete', 'insert'};
            activeColor = [0.3 0.5 0.8];
            inactiveColor = [0.2 0.2 0.25];
            for i = 1:numel(allBtns)
                btn = obj.ToolButtons_.(allBtns{i});
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
            showCPs = strcmp(mode, 'edit');
            if ~showCPs
                obj.ActiveEditCurve_ = 0;
            end
            
            if ~isempty(obj.Renderer_) && ~showCPs
                obj.Renderer_.destroyAllControlPoints();
            end

            obj.updateStatus_();
        end

        function onToggleWeights_(obj, val)
            % Toggle edge weight label visibility
            obj.ShowWeights_ = logical(val);
            if obj.ShowWeights_
                obj.drawWeightLabels_();
                set(obj.ToolButtons_.weights, 'BackgroundColor', [0.3 0.5 0.8]);
            else
                obj.clearWeightLabels_();
                set(obj.ToolButtons_.weights, 'BackgroundColor', [0.2 0.2 0.25]);
            end
        end

        function onToggleWeightUnit_(obj, src)
            % Toggle between km and miles
            if get(src, 'Value') == 1
                obj.WeightUnit_ = 'mi';
                set(src, 'String', 'mi', 'ForegroundColor', [0.9 0.6 0.2]);
            else
                obj.WeightUnit_ = 'km';
                set(src, 'String', 'km', 'ForegroundColor', [0.4 0.9 0.4]);
            end
            % Redraw if weights are visible
            if obj.ShowWeights_
                obj.drawWeightLabels_();
            end
        end

        function drawWeightLabels_(obj)
            % DRAWWEIGHTLABELS_  Draw distance labels at midpoint of each edge curve.
            obj.clearWeightLabels_();
            E = obj.Transit_.getRouteCount();
            for i = 1:E
                edge = obj.Transit_.getRouteEdge(i);
                if isempty(edge), continue; end

                spline = obj.Transit_.getSpline(i);
                w = edge.Weight;

                % Convert if miles
                if strcmp(obj.WeightUnit_, 'mi')
                    w = w * 0.621371;
                end

                % Get midpoint position
                if ~isempty(spline) && ismethod(spline, 'evaluate')
                    midPt = spline.evaluate(0.5);
                    mx = midPt(1);
                    my = midPt(2);
                else
                    % Fallback: average of source and target positions
                    pSrc = obj.Transit_.getStation(edge.Source).Position;
                    pTgt = obj.Transit_.getStation(edge.Target).Position;
                    mx = (pSrc(1) + pTgt(1)) / 2;
                    my = (pSrc(2) + pTgt(2)) / 2;
                end

                % Format label — round to integer
                label = sprintf('%d', round(w));

                % Draw with background for readability
                hTxt = text(obj.Axes_, mx, my, label, ...
                    'FontSize', 6, ...
                    'FontWeight', 'normal', ...
                    'Color', [0.1 0.1 0.5], ...
                    'BackgroundColor', [1 1 1 0.85], ...
                    'EdgeColor', [0.7 0.7 0.8], ...
                    'Margin', 1, ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.WeightHandles_{end+1} = hTxt;
            end
        end

        function clearWeightLabels_(obj)
            % Remove all weight label handles
            for i = 1:numel(obj.WeightHandles_)
                if isvalid(obj.WeightHandles_{i})
                    delete(obj.WeightHandles_{i});
                end
            end
            obj.WeightHandles_ = {};
        end

        function updateStatus_(obj)
            % Build active line suffix
            if ~isempty(obj.ActiveLineCode_)
                lineSuffix = sprintf('  [Line: %s]', obj.ActiveLineCode_);
            else
                lineSuffix = '';
            end
            
            switch obj.Mode_
                case 'place'
                    msg = ['Mode: PLACE  |  Click on canvas to add a station', lineSuffix];
                case 'connect'
                    if obj.ConnectSource_ == 0
                        msg = ['Mode: CONNECT  |  Click the FIRST station', lineSuffix];
                    else
                        srcName = obj.Transit_.getStation(obj.ConnectSource_).Name;
                        msg = sprintf('Mode: CONNECT  |  Click the SECOND station (from "%s")%s', srcName, lineSuffix);
                    end
                case 'edit'
                    msg = 'Mode: EDIT  |  Left-Drag: Move X/Y.  Shift-Drag: Move Z (Height).';
                case 'delete'
                    msg = 'Mode: DELETE  |  Click a station or near a curve to remove';
                case 'insert'
                    msg = ['Mode: INSERT  |  Click near a curve to add a station on it', lineSuffix];
                otherwise
                    msg = ['Mode: ', upper(obj.Mode_)];
            end
            
            % Refresh layered thicknesses if any routes overlap
            obj.updateParallelStyles_();
            
            if ~isempty(obj.StatusText_) && isvalid(obj.StatusText_)
                set(obj.StatusText_, 'String', msg);
            end
        end

        % ---- Mouse Callbacks ---- %
        function onAxesClick_(obj)
            % Ignore right-clicks (reserved for pan)
            selType = get(obj.Figure_, 'SelectionType');
            if ~strcmp(selType, 'normal'), return; end

            cp = get(obj.Axes_, 'CurrentPoint');
            
            % --- 3D Ray-Plane Intersection Math (Orthographic Option B) ---
            % We adapt instantly to whatever camera view the designer has chosen.
            pFront = cp(1, :);
            pBack  = cp(2, :);
            dx = pBack(1) - pFront(1);
            dy = pBack(2) - pFront(2);
            dz = pBack(3) - pFront(3);
            
            [az, el] = view(obj.Axes_);
            
            if abs(el) >= 89.9 % TOP view -> Looking down Z-axis
                if abs(dz) > 1e-6
                    t = (obj.WorkingPos_(3) - pFront(3)) / dz;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = [];
                end
            elseif abs(el) <= 0.1 && abs(az) <= 0.1 % FRONT view -> Looking down Y-axis
                if abs(dy) > 1e-6
                    t = (obj.WorkingPos_(2) - pFront(2)) / dy;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = [];
                end
            elseif abs(el) <= 0.1 && (abs(az - 90) <= 0.1 || abs(az + 90) <= 0.1) % SIDE view -> Looking down X-axis
                if abs(dx) > 1e-6
                    t = (obj.WorkingPos_(1) - pFront(1)) / dx;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = [];
                end
            else % Free Isometric 3D (Hybrid fallback to Top logic)
                if abs(dz) > 1e-6
                    t = (obj.WorkingPos_(3) - pFront(3)) / dz;
                    clickPos = pFront + t * (pBack - pFront);
                else
                    clickPos = [];
                end
            end
            
            if isempty(clickPos)
                % The mouse perfectly aligned with the target plane, making math explode (Infinity).
                % This rarely triggers unless they somehow clicked from precisely edge-on in isometric.
                return;
            end

            switch obj.Mode_
                case 'place'
                    
                    name = sprintf('S%d', obj.StationCount_ + 1);
                    name = inputdlg('Station name:', 'New Station', 1, {name});
                    if isempty(name), return; end
                    
                    obj.pushUndo_(); % Record state right before placing station
                    obj.addStation(name{1}, clickPos);
                    obj.updateStatus_();

                case 'connect'
                    % Find nearest station to click
                    nearId = obj.findNearestStation_(clickPos);
                    if nearId == 0
                        % Clicked empty space: cancel pending connection
                        if obj.ConnectSource_ > 0
                            srcH = obj.Renderer_.StationHandles_{obj.ConnectSource_};
                            set(srcH, 'MarkerEdgeColor', obj.Renderer_.StationColor * 0.6, ...
                                'LineWidth', 1.5);
                            obj.ConnectSource_ = 0;
                            obj.updateStatus_();
                        end
                        return;
                    end

                    if obj.ConnectSource_ == 0
                        obj.ConnectSource_ = nearId;
                        obj.updateStatus_();
                    else
                        if nearId ~= obj.ConnectSource_
                            obj.connectStations(obj.ConnectSource_, nearId);
                        end
                        obj.ConnectSource_ = 0;
                        obj.updateStatus_();
                    end

                case 'edit'
                    % Wake up the 4 listeners perfectly targeted to the curve clicked
                    xl = get(obj.Axes_, 'XLim');
                    yl = get(obj.Axes_, 'YLim');
                    axRange = max(xl(2) - xl(1), yl(2) - yl(1));
                    threshold = axRange * 0.08;

                    [edgeIdx, ~] = obj.Renderer_.findNearestCurve(cp, threshold);
                    if edgeIdx > 0
                        obj.ActiveEditCurve_ = edgeIdx;
                        obj.Renderer_.destroyAllControlPoints(); % clear old ones
                        spline = obj.Transit_.getSpline(edgeIdx);
                        if ~isempty(spline)
                            obj.Renderer_.drawControlPoints(edgeIdx, spline, ...
                                @(eIdx, cpIdx, newPos) obj.onControlPointMoved_(eIdx, cpIdx, newPos));
                        end
                        % Update draw order indicator
                        order = obj.Renderer_.getCurveDrawOrder(edgeIdx);
                        set(obj.PropPanel_.drawOrderLabel, 'String', ...
                            sprintf('Active Curve: Edge %d  |  Order: %d', edgeIdx, order));
                    else
                        obj.ActiveEditCurve_ = 0;
                        obj.Renderer_.destroyAllControlPoints(); % free memory
                        set(obj.PropPanel_.drawOrderLabel, 'String', 'Active Curve: (none)');
                        
                        % If clicked completely empty space, unselect station too
                        nearStationId = obj.findNearestStation_(clickPos);
                        if nearStationId == 0
                            obj.selectStation_(0);
                        end
                    end
                    obj.updateStatus_();

                case 'delete'
                    % Try station first, then edge
                    nearId = obj.findNearestStation_(clickPos);
                    if nearId > 0
                        obj.deleteStation_(nearId);
                    else
                        obj.deleteNearestEdge_(clickPos);
                    end

                case 'insert'
                    obj.insertStationOnCurve_(clickPos);
            end
        end

        function onZoomCheck_(obj)
            % ONZOOMCHECK_ Evaluates whether the camera passes the zoom declutter threshold.
            if isempty(obj.Renderer_) || ~isvalid(obj.Axes_)
                return;
            end
            xl = xlim(obj.Axes_);
            zoomRange = xl(2) - xl(1);
            isZoomedIn = (zoomRange < obj.Renderer_.ZoomThreshold);
            
            if obj.Renderer_.ZoomedIn_ ~= isZoomedIn
                obj.Renderer_.ZoomedIn_ = isZoomedIn;
                obj.onFocusModeChanged_(); 
            end
        end

        % ---- Scroll Wheel Zoom ---- %
        function onScrollZoom_(obj, evt)
            % Zoom toward cursor position
            cp = get(obj.Axes_, 'CurrentPoint');
            mx = cp(1,1); my = cp(1,2);

            xl = get(obj.Axes_, 'XLim');
            yl = get(obj.Axes_, 'YLim');

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
            xlim(obj.Axes_, newXL);
            ylim(obj.Axes_, newYL);
        end

        % ---- Right-Click Pan ---- %
        function onPanStart_(obj)
            selType = get(obj.Figure_, 'SelectionType');
            if ~strcmp(selType, 'alt'), return; end  % right-click only

            cp = get(obj.Axes_, 'CurrentPoint');
            obj.PanOrigin_ = [cp(1,1), cp(1,2)];
            obj.IsPanning_ = true;
            set(obj.Figure_, 'WindowButtonMotionFcn', @(~,~) obj.onPanMove_());
        end

        function onPanMove_(obj)
            if ~obj.IsPanning_, return; end
            cp = get(obj.Axes_, 'CurrentPoint');
            dx = obj.PanOrigin_(1) - cp(1,1);
            dy = obj.PanOrigin_(2) - cp(1,2);

            xl = get(obj.Axes_, 'XLim');
            yl = get(obj.Axes_, 'YLim');
            xlim(obj.Axes_, xl + dx);
            ylim(obj.Axes_, yl + dy);
        end

        function onPanEnd_(obj)
            obj.IsPanning_ = false;
            set(obj.Figure_, 'WindowButtonMotionFcn', '');
        end

        % ---- Undo System ---- %
        function pushUndo_(obj)
            % Save current state strictly to RAM and push onto stack
            obj.UndoStack_{end+1} = obj.getDesignStruct_();
            
            % Limit stack to 20 entries to cap memory usage
            if numel(obj.UndoStack_) > 20
                obj.UndoStack_(1) = [];
            end
        end

        function onUndo_(obj)
            if isempty(obj.UndoStack_)
                return;
            end
            % Pop last snapshot
            dataStr = obj.UndoStack_{end};
            obj.UndoStack_(end) = [];

            % Reload strictly from RAM
            obj.loadDesignStruct_(dataStr);
            obj.updateStatus_();
        end

        function onKeyPress_(obj, evt)
            % Ctrl+Z = Undo
            if strcmp(evt.Key, 'z') && ~isempty(evt.Modifier) && ...
                    any(strcmp(evt.Modifier, 'control'))
                obj.onUndo_();
            end
        end

        function nearId = findNearestStation_(obj, clickPos)
            % Find the station nearest to the click position.
            % Adaptive threshold: 10% of the larger axes dimension.
            nearId = 0;
            minDist = inf;

            xl = get(obj.Axes_, 'XLim');
            yl = get(obj.Axes_, 'YLim');
            axRange = max(xl(2) - xl(1), yl(2) - yl(1));
            threshold = max(0.1, axRange * 0.10);  % at least 0.1, or 10% of range

            for i = 1:obj.StationCount_
                pos = obj.Transit_.getStation(i).Position;
                d = sqrt(sum((pos(1:2) - clickPos(1:2)).^2));
                if d < minDist && d < threshold
                    minDist = d;
                    nearId = i;
                end
            end
        end

        % ---- Station Click (direct from marker) ---- %
        function onStationClick_(obj, nodeId)
            % Called when a station marker is clicked directly.
            
            % Always select the station to populate the properties panel
            obj.selectStation_(nodeId);
            
            switch obj.Mode_
                case 'connect'
                    if obj.ConnectSource_ == 0
                        obj.ConnectSource_ = nodeId;
                        % Highlight selected station
                        h = obj.Renderer_.StationHandles_{nodeId};
                        set(h, 'MarkerEdgeColor', [1 1 0], 'LineWidth', 2.5);
                        obj.updateStatus_();
                    else
                        if nodeId ~= obj.ConnectSource_
                            obj.connectStations(obj.ConnectSource_, nodeId);
                        end
                        % Unhighlight source station
                        srcH = obj.Renderer_.StationHandles_{obj.ConnectSource_};
                        set(srcH, 'MarkerEdgeColor', obj.Renderer_.StationColor * 0.6, ...
                            'LineWidth', 1.5);
                        obj.ConnectSource_ = 0;
                        obj.updateStatus_();
                    end
                case 'place'
                    % Do nothing — place mode only responds to empty canvas
                case 'edit'
                    % Properties panel is already populated above
                case 'delete'
                    obj.deleteStation_(nodeId);
            end
        end


        % ---- Control Point Drag ---- %
        function onControlPointMoved_(obj, edgeIdx, cpIdx, newPos)
            % Update the spline's control point and redraw
            spline = obj.Transit_.getSpline(edgeIdx);
            spline.setControlPoint(cpIdx, newPos);

            % Redraw curve
            obj.Renderer_.updateCurve(edgeIdx, spline);

            % Update weight via transit network
            obj.Transit_.updateRouteSpline(edgeIdx, spline);
        end

        % ---- Button Callbacks ---- %
        function onExport_(obj)
            choice = questdlg('Export format:', 'Export', ...
                'Workspace', 'JSON File', 'Cancel', 'JSON File');
            switch choice
                case 'Workspace'
                    g = obj.exportGraph();
                    assignin('base', 'trackGraph', g);
                    g.describe();
                    msgbox('Graph exported to workspace as "trackGraph"', 'Export');
                case 'JSON File'
                    obj.exportJSON_();
            end
        end

        function exportJSON_(obj)
            [file, path] = uiputfile('*.json', 'Export Track as JSON');
            if file == 0, return; end
            filepath = fullfile(path, file);

            % Create dynamic ID mapping map cleanly
            idMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
            newIdCounter = 1;
            
            % Extract Global Meta Data
            data = struct();
            data.meta = struct();
            data.meta.defaultLabelOffset = obj.Renderer_.DefaultLabelOffset(:)';
            
            % Build stations array
            % We use cell arrays of structs so jsonencode reliably writes arrays of objects
            stations = cell(0);
            for i = 1:obj.StationCount_
                % Check if visually deleted
                h = obj.Renderer_.StationHandles_{i};
                if isempty(h) || ~isvalid(h)
                    continue;
                end
                
                st = obj.Transit_.getStation(i);
                if isempty(st), continue; end
                
                % Assign consecutive sequential ID
                idMap(st.Id) = newIdCounter;
                
                s = struct();
                s.id = newIdCounter;
                s.name = st.Name;
                
                % Optimize position: strip Z if it's strictly 0
                pos = st.Position;
                if length(pos) >= 3 && pos(3) == 0
                    pos = pos(1:2);
                end
                s.position = pos;
                
                % Only embed label offset if it's uniquely overridden for this exact station
                if i <= numel(obj.Renderer_.LabelOffsets_) && ~isempty(obj.Renderer_.LabelOffsets_{i})
                    overrideOffset = obj.Renderer_.LabelOffsets_{i}(:)';
                    if ~isequal(overrideOffset, data.meta.defaultLabelOffset)
                        s.labelOffset = overrideOffset;
                    end
                end
                
                stations{end+1} = s; %#ok<AGROW>
                newIdCounter = newIdCounter + 1;
            end

            % Build edges array with spline control points and exact colors
            edges = cell(0);
            for i = 1:obj.EdgeCount_
                e = obj.Transit_.getRouteEdge(i);
                if isempty(e), continue; end
                
                % Map Edge sources/targets strictly to the new consecutive sequence
                if ~idMap.isKey(e.Source) || ~idMap.isKey(e.Target)
                    continue; % Corrupt ghost edge found mathematically, skip it
                end
                
                ed = struct();
                ed.id = numel(edges) + 1;
                ed.source = idMap(e.Source);
                ed.target = idMap(e.Target);
                ed.weight = e.Weight;
                
                % Store explicit visual color of the curve
                if i <= numel(obj.Renderer_.CurveColors_) && ~isempty(obj.Renderer_.CurveColors_{i})
                    ed.color = obj.Renderer_.CurveColors_{i};
                else
                    ed.color = obj.Renderer_.CurveColor; % fallback cyan
                end

                spl = obj.Transit_.getSpline(i);
                if ~isempty(spl)
                    ed.controlPoints = spl.getControlPoints();
                    
                    % Simplify MATLAB class names for Python/agnostic consumers
                    cName = class(spl);
                    if contains(cName, 'CatmullRom')
                        ed.curveType = 'catmullrom';
                    elseif contains(cName, 'Bezier')
                        ed.curveType = 'bezier';
                    else
                        ed.curveType = cName;
                    end
                else
                    ed.controlPoints = [];
                    ed.curveType = 'none';
                end
                % Draw order for intersection rendering
                ed.drawOrder = obj.Renderer_.getCurveDrawOrder(i);
                edges{end+1} = ed; %#ok<AGROW>
            end

            % Build transit lines
            lines = cell(0);
            codes = obj.Transit_.getLineNames();
            for i = 1:numel(codes)
                ln = obj.Transit_.getLine(codes{i});
                
                li = struct();
                li.code = ln.Code;
                li.name = ln.Name;
                li.color = ln.Color;
                
                % Re-map route array through the dictionary
                oldRoute = ln.getRoute();
                newRoute = [];
                for rId = 1:numel(oldRoute)
                    if idMap.isKey(oldRoute(rId))
                        newRoute(end+1) = idMap(oldRoute(rId)); %#ok<AGROW>
                    end
                end
                li.route = newRoute;
                lines{end+1} = li; %#ok<AGROW>
            end

            % Assemble and write
            data.stations = stations;
            data.edges = edges;
            data.lines = lines;

            % Save legend and title positions
            data.legendPos1 = obj.LegendPos1_;
            data.legendPos2 = obj.LegendPos2_;
            data.titlePos = obj.TitlePos_;
            data.mapTitle = obj.MapTitle_;

            jsonStr = jsonencode(data, 'PrettyPrint', true);
            fid = fopen(filepath, 'w');
            fprintf(fid, '%s', jsonStr);
            fclose(fid);

            msgbox(sprintf('Exported to: %s', filepath), 'JSON Export');
        end

        function importJSON_(obj)
            [file, path] = uigetfile('*.json', 'Import Track from JSON');
            if file == 0, return; end
            filepath = fullfile(path, file);

            try
                text = fileread(filepath);
                data = jsondecode(text);
            catch ME
                uiwait(errordlg(sprintf('Failed to read JSON file:\n%s', ME.message), 'Import Error'));
                return;
            end
            
            % Normalize jsondecode output: struct arrays -> cell arrays
            % jsondecode can return either depending on field consistency
            toCell = @(x) arrayfun(@(v) v, x, 'UniformOutput', false);
            if isfield(data, 'stations') && isstruct(data.stations)
                data.stations = toCell(data.stations);
            end
            if isfield(data, 'edges') && isstruct(data.edges)
                data.edges = toCell(data.edges);
            end
            if isfield(data, 'lines') && isstruct(data.lines)
                data.lines = toCell(data.lines);
            end
            
            obj.pushUndo_(); % Save current before replacing
            
            % Clear current state
            obj.Transit_ = OceanMath.Transit.TransitNetwork();
            obj.StationCount_ = 0;
            obj.EdgeCount_ = 0;
            if ~isempty(obj.Renderer_)
                obj.Renderer_.clearAll();
            end
            
            idMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
            
            % Read Global Meta Data
            defaultOffset = [0, 0.3]; % Hard fallback
            if isfield(data, 'meta') && isfield(data.meta, 'defaultLabelOffset')
                defaultOffset = data.meta.defaultLabelOffset(:)';
            end

            % 1. Rebuild Stations
            if isfield(data, 'stations') && ~isempty(data.stations)
                obj.Transit_.reserveStations(numel(data.stations));
                for i = 1:numel(data.stations)
                    if iscell(data.stations)
                        s = data.stations{i};
                    else
                        s = data.stations(i);
                    end
                    obj.addStation(s.name, s.position(:)', 'SuspendGraphics', true);
                    newId = obj.StationCount_;
                    
                    if isfield(s, 'id')
                        idMap(s.id) = newId; % Map legacy explicit ID to strictly sequential ID
                    else
                        idMap(i) = newId; % Fallback
                    end
                    
                    if ~isempty(obj.Renderer_)
                        if isfield(s, 'labelOffset')
                            obj.Renderer_.setLabelOffset(newId, s.labelOffset(:)');
                        else
                            obj.Renderer_.setLabelOffset(newId, defaultOffset);
                        end
                    end
                end
            end
            
            % 2. Rebuild Edges
            if isfield(data, 'edges') && ~isempty(data.edges)
                for i = 1:numel(data.edges)
                    try
                        if iscell(data.edges)
                            e = data.edges{i};
                        else
                            e = data.edges(i);
                        end
                        if ~idMap.isKey(e.source) || ~idMap.isKey(e.target)
                            continue;
                        end
                        sId = idMap(e.source);
                        tId = idMap(e.target);
                        
                        obj.EdgeCount_ = obj.EdgeCount_ + 1;
                        
                        if isfield(e, 'controlPoints') && ~isempty(e.controlPoints)
                            cp = e.controlPoints;
                            if iscell(cp)
                                cp = cell2mat(reshape(cp, [], 1));
                            end
                            if isfield(e, 'curveType') && strcmpi(e.curveType, 'bezier')
                                spline = OceanMath.Geometry.BezierCurve(cp);
                            else
                                spline = OceanMath.Geometry.CatmullRomSpline(cp);
                            end
                            obj.Transit_.addRoute(sId, tId, spline);
                            if ~isempty(obj.Renderer_)
                                dOrder = 0;
                                if isfield(e, 'drawOrder'), dOrder = e.drawOrder; end
                                eCol = obj.Renderer_.CurveColor;
                                if isfield(e, 'color') && ~isempty(e.color), eCol = e.color(:)'; end
                                obj.Renderer_.drawCurve(obj.EdgeCount_, spline, 'Color', eCol, 'DrawOrder', dOrder);
                            end
                        else
                            obj.Transit_.addSimpleRoute(sId, tId, e.weight);
                        end
                    catch ME
                        fprintf('[importJSON] EDGE %d FAILED: %s\n', i, ME.message);
                    end
                end
            end
            
            % 3. Rebuild Lines
            if isfield(data, 'lines') && ~isempty(data.lines)
                for i = 1:numel(data.lines)
                    if iscell(data.lines)
                        li = data.lines{i};
                    else
                        li = data.lines(i);
                    end
                    obj.Transit_.addLine(li.code, li.name, li.color(:)');
                    lineObj = obj.Transit_.getLine(li.code);
                    if isfield(li, 'route') && ~isempty(li.route)
                        for j = 1:numel(li.route)
                            if idMap.isKey(li.route(j))
                                lineObj.registerStation(idMap(li.route(j)));
                            end
                        end
                    end
                end
                obj.Transit_.invalidateLineCache();
                obj.refreshLineDropdown_();
            end
            
            % Restore legend positions if saved
            if isfield(data, 'legendPos1') && numel(data.legendPos1) == 2
                obj.LegendPos1_ = data.legendPos1(:)';
            end
            if isfield(data, 'legendPos2') && numel(data.legendPos2) == 2
                obj.LegendPos2_ = data.legendPos2(:)';
            end
            if isfield(data, 'titlePos') && numel(data.titlePos) == 2
                obj.TitlePos_ = data.titlePos(:)';
            end
            if isfield(data, 'mapTitle') && ~isempty(data.mapTitle)
                obj.MapTitle_ = data.mapTitle;
            end

            % Restore interchange visuals globally and enforce axis limits
            for i = 1:obj.StationCount_
                obj.updateStationVisuals_(i);
            end
            % Update all overlaps and bounds perfectly
            obj.updateStatus_();
            if ~isempty(obj.Renderer_)
                obj.Renderer_.reorderCurves();
                obj.updateAxisLimits_();
                % Force 2D view and stretch-to-fill after all plot3 calls
                view(obj.Axes_, 2);
                axis(obj.Axes_, 'normal');
            end
            
            % Draw transit legend at bottom
            obj.drawLegend_();
        end

        function onSave_(obj)
            [file, path] = uiputfile('*.mat', 'Save Track Design');
            if file == 0, return; end
            obj.saveDesign(fullfile(path, file));
        end

        function onNew_(obj)
            % ONNEW_ Clear the workspace after confirmation.
            choice = questdlg('Clear entire workspace? Unsaved changes will be lost.', ...
                'Confirm New', 'Yes', 'Cancel', 'Cancel');
            if strcmp(choice, 'Yes')
                obj.pushUndo_();
                obj.Transit_ = OceanMath.Transit.TransitNetwork();
                obj.StationCount_ = 0;
                obj.EdgeCount_ = 0;
                obj.WorkingPos_ = [0 0 0];
                obj.SelectedStation_ = 0;
                obj.ActiveLineCode_ = '';
                obj.ActiveEditCurve_ = 0;
                if ~isempty(obj.Renderer_)
                    obj.Renderer_.clearAll();
                end
                obj.clearWeightLabels_();
                % Clear legend
                for k = 1:numel(obj.LegendHandles_)
                    if isvalid(obj.LegendHandles_{k}), delete(obj.LegendHandles_{k}); end
                end
                obj.LegendHandles_ = {};
                obj.LegendPos1_ = [NaN NaN];
                obj.LegendPos2_ = [NaN NaN];
                obj.TitlePos_ = [NaN NaN];
                if ~isempty(obj.TitleHandle_) && isvalid(obj.TitleHandle_)
                    delete(obj.TitleHandle_);
                end
                obj.ShowWeights_ = false;
                if isfield(obj.ToolButtons_, 'weights')
                    set(obj.ToolButtons_.weights, 'Value', 0, 'BackgroundColor', [0.2 0.2 0.25]);
                end
                obj.updateStatus_();
                obj.refreshLineDropdown_();
                set(obj.LinePanel_.colorIndicator, ...
                    'BackgroundColor', [0.15, 0.15, 0.18], 'String', '');
                set(obj.PropPanel_.drawOrderLabel, 'String', 'Active Curve: (none)');
                set(obj.PropPanel_.selectedLabel, 'String', 'No station selected');
            end
        end

        function onLoad_(obj)
            choice = questdlg('Load format:', 'Load Design', ...
                'Designer File (.mat)', 'JSON File (.json)', 'Cancel', 'Designer File (.mat)');
            switch choice
                case 'Designer File (.mat)'
                    [file, path] = uigetfile('*.mat', 'Load Track Design');
                    if file ~= 0
                        obj.loadDesign(fullfile(path, file));
                    end
                case 'JSON File (.json)'
                    obj.importJSON_();
            end
        end

        function onClose_(obj)
            delete(obj.Figure_);
        end

        function drawLegend_(obj)
            % DRAWLEGEND_  Draw two draggable legend boxes.
            %   Box 1: Train Services
            %   Box 2: Station Types + Path Notes
            %   Drag any title text to reposition its box.

            % Clear previous legend
            for i = 1:numel(obj.LegendHandles_)
                if isvalid(obj.LegendHandles_{i})
                    delete(obj.LegendHandles_{i});
                end
            end
            obj.LegendHandles_ = {};

            codes = obj.Transit_.getLineNames();
            if isempty(codes), return; end

            % Data bounds for sizing
            positions = obj.Transit_.getStationPositions();
            yMin = min(positions(:,2));
            yMax = max(positions(:,2));
            xMin = min(positions(:,1));
            xMax = max(positions(:,1));
            yRange = yMax - yMin;
            xRange = xMax - xMin;

            lineH = yRange * 0.025;
            lineBarW = xRange * 0.05;

            % Default positions if not yet placed
            if isnan(obj.LegendPos1_(1))
                obj.LegendPos1_ = [xMin, yMin + yRange * 0.35];
            end
            if isnan(obj.LegendPos2_(1))
                obj.LegendPos2_ = [xMin, yMin - yRange * 0.01];
            end
            if isnan(obj.TitlePos_(1))
                obj.TitlePos_ = [xMax - xRange * 0.01, yMax + yRange * 0.01];
            end

            % ============== MAP TITLE (draggable) ==============
            if ~isempty(obj.TitleHandle_) && isvalid(obj.TitleHandle_)
                delete(obj.TitleHandle_);
            end
            obj.TitleHandle_ = text(obj.Axes_, obj.TitlePos_(1), obj.TitlePos_(2), ...
                obj.MapTitle_, ...
                'FontSize', 14, 'FontWeight', 'bold', ...
                'Color', [0.05 0.05 0.05], ...
                'HorizontalAlignment', 'right', ...
                'VerticalAlignment', 'top', ...
                'HitTest', 'on', 'PickableParts', 'visible', ...
                'ButtonDownFcn', @(~,~) obj.startLegendDrag_(3));

            % ============== BOX 1: Train Services ==============
            bx = obj.LegendPos1_(1);
            by = obj.LegendPos1_(2);
            nLines = numel(codes);

            % Title (draggable)
            h = text(obj.Axes_, bx, by, 'Train Services', ...
                'FontSize', 13, 'FontWeight', 'bold', ...
                'Color', [0.1 0.1 0.1], ...
                'VerticalAlignment', 'top', ...
                'HitTest', 'on', 'PickableParts', 'visible', ...
                'ButtonDownFcn', @(~,~) obj.startLegendDrag_(1));
            obj.LegendHandles_{end+1} = h;

            entryY = by - lineH * 1.2;
            for i = 1:nLines
                lineObj = obj.Transit_.getLine(codes{i});
                ey = entryY - (i-1) * lineH;

                hBar = plot(obj.Axes_, [bx, bx + lineBarW], [ey, ey], '-', ...
                    'Color', lineObj.Color, 'LineWidth', 5, ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.LegendHandles_{end+1} = hBar;

                nSta = lineObj.getStationCount();
                label = sprintf('%s  %s (%d)', lineObj.Code, lineObj.Name, nSta);
                hTxt = text(obj.Axes_, bx + lineBarW * 1.2, ey, label, ...
                    'FontSize', 10, 'Color', [0.15 0.15 0.15], ...
                    'VerticalAlignment', 'middle', ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.LegendHandles_{end+1} = hTxt;
            end

            % ============== BOX 2: Station Types + Path Notes ==============
            bx2 = obj.LegendPos2_(1);
            by2 = obj.LegendPos2_(2);

            % Station Types title (draggable)
            h = text(obj.Axes_, bx2, by2, 'Station Types', ...
                'FontSize', 13, 'FontWeight', 'bold', ...
                'Color', [0.1 0.1 0.1], ...
                'VerticalAlignment', 'top', ...
                'HitTest', 'on', 'PickableParts', 'visible', ...
                'ButtonDownFcn', @(~,~) obj.startLegendDrag_(2));
            obj.LegendHandles_{end+1} = h;

            stEntryY = by2 - lineH * 1.2;
            types = {
                'o', [1 1 1], [0 0 0],        2.0, 'Interchange';
                'd', [1 1 1], [0.15 0.15 0.15], 1.5, 'Terminus';
                'o', [1 1 1], [0.4 0.4 0.4],   1.2, 'Standard'
            };
            for i = 1:size(types, 1)
                sy = stEntryY - (i-1) * lineH;
                hM = plot(obj.Axes_, bx2 + lineBarW * 0.4, sy, ...
                    types{i,1}, ...
                    'MarkerFaceColor', types{i,2}, ...
                    'MarkerEdgeColor', types{i,3}, ...
                    'LineWidth', types{i,4}, ...
                    'MarkerSize', 10, ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.LegendHandles_{end+1} = hM;
                hTxt = text(obj.Axes_, bx2 + lineBarW * 0.9, sy, types{i,5}, ...
                    'FontSize', 10, 'Color', [0.15 0.15 0.15], ...
                    'VerticalAlignment', 'middle', ...
                    'HitTest', 'off', 'PickableParts', 'none');
                obj.LegendHandles_{end+1} = hTxt;
            end

            % Path Notes
            notesY = stEntryY - 3 * lineH - lineH * 0.5;
            h = text(obj.Axes_, bx2, notesY, 'Path Notes', ...
                'FontSize', 13, 'FontWeight', 'bold', ...
                'Color', [0.1 0.1 0.1], ...
                'VerticalAlignment', 'top', ...
                'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = h;

            noteEntryY = notesY - lineH * 1.2;
            c1 = [0.85 0.2 0.2]; c2 = [0 0.35 0.75];

            % Shared Path
            spY = noteEntryY;
            hL1 = plot(obj.Axes_, [bx2, bx2 + lineBarW], [spY+lineH*0.08, spY+lineH*0.08], '-', ...
                'Color', c1, 'LineWidth', 3, 'HitTest', 'off', 'PickableParts', 'none');
            hL2 = plot(obj.Axes_, [bx2, bx2 + lineBarW], [spY-lineH*0.08, spY-lineH*0.08], '-', ...
                'Color', c2, 'LineWidth', 3, 'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = hL1;
            obj.LegendHandles_{end+1} = hL2;
            hTxt = text(obj.Axes_, bx2 + lineBarW * 1.2, spY, 'Shared Path', ...
                'FontSize', 10, 'Color', [0.15 0.15 0.15], ...
                'VerticalAlignment', 'middle', ...
                'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = hTxt;

            % Line Crossing
            crY = noteEntryY - lineH;
            hC1 = plot(obj.Axes_, [bx2, bx2 + lineBarW], [crY-lineH*0.12, crY+lineH*0.12], '-', ...
                'Color', c1, 'LineWidth', 3, 'HitTest', 'off', 'PickableParts', 'none');
            hC2 = plot(obj.Axes_, [bx2, bx2 + lineBarW], [crY+lineH*0.12, crY-lineH*0.12], '-', ...
                'Color', c2, 'LineWidth', 3, 'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = hC1;
            obj.LegendHandles_{end+1} = hC2;
            hTxt = text(obj.Axes_, bx2 + lineBarW * 1.2, crY, 'Line Crossing', ...
                'FontSize', 10, 'Color', [0.15 0.15 0.15], ...
                'VerticalAlignment', 'middle', ...
                'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = hTxt;

            % Distance label annotation
            distY = noteEntryY - 2 * lineH;
            hSample = text(obj.Axes_, bx2 + lineBarW * 0.25, distY, '5', ...
                'FontSize', 6, 'FontWeight', 'normal', ...
                'Color', [0.1 0.1 0.5], ...
                'BackgroundColor', [1 1 1 0.85], ...
                'EdgeColor', [0.7 0.7 0.8], ...
                'Margin', 1, ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = hSample;
            hTxt = text(obj.Axes_, bx2 + lineBarW * 0.9, distY, 'Distance (km/mi)', ...
                'FontSize', 10, 'Color', [0.15 0.15 0.15], ...
                'VerticalAlignment', 'middle', ...
                'HitTest', 'off', 'PickableParts', 'none');
            obj.LegendHandles_{end+1} = hTxt;
        end

        function startLegendDrag_(obj, boxId)
            % Begin dragging a legend box (1, 2, or 3=title)
            cp = get(obj.Axes_, 'CurrentPoint');
            mousePos = cp(1, 1:2);
            if boxId == 1
                obj.DragOffset_ = mousePos - obj.LegendPos1_;
            elseif boxId == 2
                obj.DragOffset_ = mousePos - obj.LegendPos2_;
            else
                obj.DragOffset_ = mousePos - obj.TitlePos_;
            end
            obj.DraggingLegend_ = boxId;
            set(obj.Figure_, 'WindowButtonMotionFcn', @(~,~) obj.onLegendDragMove_());
            set(obj.Figure_, 'WindowButtonUpFcn', @(~,~) obj.onLegendDragEnd_());
        end

        function onLegendDragMove_(obj)
            if obj.DraggingLegend_ == 0, return; end
            cp = get(obj.Axes_, 'CurrentPoint');
            newPos = cp(1, 1:2) - obj.DragOffset_;
            if obj.DraggingLegend_ == 1
                obj.LegendPos1_ = newPos;
            elseif obj.DraggingLegend_ == 2
                obj.LegendPos2_ = newPos;
            else
                obj.TitlePos_ = newPos;
            end
            obj.drawLegend_();
        end

        function onLegendDragEnd_(obj)
            obj.DraggingLegend_ = 0;
            set(obj.Figure_, 'WindowButtonMotionFcn', '');
            set(obj.Figure_, 'WindowButtonUpFcn', '');
        end

        function updateAxisLimits_(obj)
            % UPDATEAXISLIMITS_  Compute axis limits from data with margins.
            %   Y top:    max + 2% of Y range
            %   Y bottom: min - 10% of Y range (space for label tables)
            %   X:        min/max +/- 1% of X range
            if obj.StationCount_ < 2, return; end
            positions = obj.Transit_.getStationPositions();
            xMin = min(positions(:,1));  xMax = max(positions(:,1));
            yMin = min(positions(:,2));  yMax = max(positions(:,2));

            xRange = max(xMax - xMin, 1);
            yRange = max(yMax - yMin, 1);

            newXL = [xMin - xRange * 0.01,  xMax + xRange * 0.01];
            newYL = [yMin - yRange * 0.10,  yMax + yRange * 0.02];

            xlim(obj.Axes_, newXL);
            ylim(obj.Axes_, newYL);

            % Force 2D stretch-to-fill
            view(obj.Axes_, 2);
            axis(obj.Axes_, 'normal');
        end

        % ---- Property Panel ---- %
        function createPropertyPanel_(obj)
            % CREATEPROPERTYPANEL_  Build the right-side property editor.
            bgColor = [0.15, 0.15, 0.18];
            fgColor = [0.85, 0.90, 0.95];
            fieldBg = [0.20, 0.20, 0.25];
            panelX  = 0.72;
            panelW  = 0.26;
            rowH    = 0.033;
            gap     = 0.004;

            % ---- Line Management ---- %
            y = 0.94;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Transit Lines', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW, rowH], ...
                'BackgroundColor', [0.12 0.12 0.15], ...
                'ForegroundColor', [0.6 0.8 1.0], ...
                'FontSize', 12, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');

            % Line code
            y = y - rowH - gap*2;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Code:', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW*0.25, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                'FontSize', 9, 'HorizontalAlignment', 'left');

            obj.LinePanel_.codeField = uicontrol(obj.Figure_, ...
                'Style', 'edit', 'String', 'L1', ...
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.27, y, panelW*0.73, rowH], ...
                'BackgroundColor', fieldBg, 'ForegroundColor', fgColor, ...
                'FontSize', 9);

            % Line name
            y = y - rowH - gap;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Name:', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW*0.25, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                'FontSize', 9, 'HorizontalAlignment', 'left');

            obj.LinePanel_.nameField = uicontrol(obj.Figure_, ...
                'Style', 'edit', 'String', 'Line 1', ...
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.27, y, panelW*0.73, rowH], ...
                'BackgroundColor', fieldBg, 'ForegroundColor', fgColor, ...
                'FontSize', 9);

            % Line color (predefined palette)
            y = y - rowH - gap;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Color:', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW*0.25, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                'FontSize', 9, 'HorizontalAlignment', 'left');

            obj.LinePanel_.colorPicker = uicontrol(obj.Figure_, ...
                'Style', 'popupmenu', ...
                'String', {'Red (NSL)', 'Green (EWL)', 'Purple (NEL)', 'Orange (CCL)', 'Blue (DTL)', 'Brown (TEL)', 'Teal (JRL)', 'Lime (CRL)', 'Grey (LRT)', 'Cyan', 'Yellow', 'Magenta', 'Pink'}, ...
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.27, y, panelW*0.73, rowH], ...
                'BackgroundColor', fieldBg, 'ForegroundColor', fgColor, ...
                'FontSize', 9);

            % Create Line button
            y = y - rowH - gap*2;
            uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', ...
                'String', 'Create Line', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW, rowH], ...
                'BackgroundColor', [0.25 0.45 0.7], ...
                'ForegroundColor', [1 1 1], ...
                'FontSize', 9, ...
                'Callback', @(~,~) obj.onCreateLine_());

            % Active Line dropdown
            y = y - rowH - gap*2;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Active Line:', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW*0.40, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                'FontSize', 9, 'HorizontalAlignment', 'left');

            obj.LinePanel_.activeDropdown = uicontrol(obj.Figure_, ...
                'Style', 'popupmenu', ...
                'String', {'(none)'}, ...
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.42, y, panelW*0.58, rowH], ...
                'BackgroundColor', fieldBg, 'ForegroundColor', fgColor, ...
                'FontSize', 9, ...
                'Callback', @(src,~) obj.onSelectActiveLine_(src));

            % Active line color indicator
            y = y - rowH*0.6;
            obj.LinePanel_.colorIndicator = uicontrol(obj.Figure_, ...
                'Style', 'text', 'String', '', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW, rowH*0.5], ...
                'BackgroundColor', bgColor);

            % ---- Focus Mode ---- %
            y = y - rowH - gap*2;
            obj.FocusCheckbox_ = uicontrol(obj.Figure_, ...
                'Style', 'checkbox', ...
                'String', 'Focus Mode (Hide Others)', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW, rowH], ...
                'BackgroundColor', bgColor, ...
                'ForegroundColor', [1 0.8 0.2], ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'Callback', @(~,~) obj.onFocusModeChanged_());

            % ---- Separator ---- %
            y = y - gap*4;

            % ---- Curve Type ---- %
            y = y - rowH;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Curve Type:', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW*0.40, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                'FontSize', 9, 'HorizontalAlignment', 'left');

            obj.LinePanel_.curveTypeDropdown = uicontrol(obj.Figure_, ...
                'Style', 'popupmenu', ...
                'String', {'CatmullRom', 'Bezier', 'Linear'}, ...
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.42, y, panelW*0.58, rowH], ...
                'BackgroundColor', fieldBg, 'ForegroundColor', fgColor, ...
                'FontSize', 9, ...
                'Callback', @(src,~) obj.onSelectCurveType_(src));

            % ---- Separator ---- %
            y = y - gap*4;

            % ---- Station/Path Properties ---- %
            y = y - rowH;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Station/Path Properties', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW, rowH], ...
                'BackgroundColor', [0.12 0.12 0.15], ...
                'ForegroundColor', [0.6 0.8 1.0], ...
                'FontSize', 11, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');

            % --- Station Name ---
            y = y - rowH - gap*2;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Station Name:', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW*0.45, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                'FontSize', 9, 'HorizontalAlignment', 'left');

            obj.PropPanel_.nameField = uicontrol(obj.Figure_, ...
                'Style', 'edit', 'String', '', ...
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.47, y, panelW*0.53, rowH], ...
                'BackgroundColor', fieldBg, 'ForegroundColor', fgColor, ...
                'FontSize', 9, ...
                'Callback', @(~,~) obj.onApplyName_());

            % --- Elevation (Z) ---
            y = y - rowH - gap;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Elevation (Z):', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW*0.45, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                'FontSize', 9, 'HorizontalAlignment', 'left');

            obj.PropPanel_.elevationField = uicontrol(obj.Figure_, ...
                'Style', 'edit', 'String', '0', ...
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.47, y, panelW*0.53, rowH], ...
                'BackgroundColor', fieldBg, 'ForegroundColor', fgColor, ...
                'FontSize', 9, ...
                'Callback', @(~,~) obj.onApplyElevation_());

            % --- Label Offset X ---
            y = y - rowH - gap;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Label X Offset:', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW*0.55, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                'FontSize', 9, 'HorizontalAlignment', 'left');

            obj.PropPanel_.offsetX = uicontrol(obj.Figure_, ...
                'Style', 'edit', 'String', '0', ...
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.57, y, panelW*0.43, rowH], ...
                'BackgroundColor', fieldBg, 'ForegroundColor', fgColor, ...
                'FontSize', 9, ...
                'Callback', @(~,~) obj.onApplyOffset_());

            % --- Label Offset Y ---
            y = y - rowH - gap;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Label Y Offset:', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW*0.55, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                'FontSize', 9, 'HorizontalAlignment', 'left');

            obj.PropPanel_.offsetY = uicontrol(obj.Figure_, ...
                'Style', 'edit', 'String', '0.3', ...
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.57, y, panelW*0.43, rowH], ...
                'BackgroundColor', fieldBg, 'ForegroundColor', fgColor, ...
                'FontSize', 9, ...
                'Callback', @(~,~) obj.onApplyOffset_());

            % --- Apply Offset to All ---
            y = y - rowH - gap*2;
            uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', ...
                'String', 'Apply Offset to All', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW, rowH], ...
                'BackgroundColor', [0.25 0.45 0.7], ...
                'ForegroundColor', [1 1 1], ...
                'FontSize', 9, ...
                'Callback', @(~,~) obj.onApplyAllOffsets_());

            % --- Font Size ---
            y = y - rowH - gap*2;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Font Size:', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW*0.40, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                'FontSize', 9, 'HorizontalAlignment', 'left');

            obj.PropPanel_.fontSizeField = uicontrol(obj.Figure_, ...
                'Style', 'edit', 'String', '9', ...
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.42, y, panelW*0.28, rowH], ...
                'BackgroundColor', fieldBg, 'ForegroundColor', fgColor, ...
                'FontSize', 9, ...
                'Callback', @(~,~) obj.onApplyFontSize_());

            uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Apply', ...
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.72, y, panelW*0.28, rowH], ...
                'BackgroundColor', [0.25 0.45 0.7], ...
                'ForegroundColor', [1 1 1], ...
                'FontSize', 9, ...
                'Callback', @(~,~) obj.onApplyFontSize_());

            % --- Selected ID display ---
            y = y - rowH*1.5 - gap*2;
            
            uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', 'Route Active Line Here', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW, rowH*1.2], ...
                'BackgroundColor', [0.6 0.25 0.7], ...
                'ForegroundColor', [1 1 1], ...
                'FontSize', 9, 'FontWeight', 'bold', ...
                'Callback', @(~,~) obj.onRouteLineHere_());
            
            y = y - rowH*1.2 - gap*2;
            obj.PropPanel_.selectedLabel = uicontrol(obj.Figure_, ...
                'Style', 'text', ...
                'String', 'No station selected', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW, rowH*1.5], ...
                'BackgroundColor', bgColor, 'ForegroundColor', [0.9 0.8 0.4], ...
                'FontSize', 9, 'FontAngle', 'italic', ...
                'HorizontalAlignment', 'center');

            % ---- Path Draw Order (for curve intersections) ---- %
            y = y - rowH*1.8 - gap*2;
            uicontrol(obj.Figure_, 'Style', 'text', ...
                'String', 'Curve Draw Order:', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW*0.60, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', fgColor, ...
                'FontSize', 9, 'HorizontalAlignment', 'left');

            halfBtnW = panelW * 0.18;
            uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', char(9650), ... % ▲
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.62, y, halfBtnW, rowH], ...
                'BackgroundColor', [0.25 0.50 0.35], ...
                'ForegroundColor', [1 1 1], ...
                'FontSize', 11, ...
                'Callback', @(~,~) obj.onCurveDrawOrderUp_());

            uicontrol(obj.Figure_, ...
                'Style', 'pushbutton', 'String', char(9660), ... % ▼
                'Units', 'normalized', ...
                'Position', [panelX + panelW*0.82, y, halfBtnW, rowH], ...
                'BackgroundColor', [0.50 0.25 0.25], ...
                'ForegroundColor', [1 1 1], ...
                'FontSize', 11, ...
                'Callback', @(~,~) obj.onCurveDrawOrderDown_());

            y = y - rowH - gap;
            obj.PropPanel_.drawOrderLabel = uicontrol(obj.Figure_, ...
                'Style', 'text', ...
                'String', 'Active Curve: (none)', ...
                'Units', 'normalized', ...
                'Position', [panelX, y, panelW, rowH], ...
                'BackgroundColor', bgColor, 'ForegroundColor', [0.7 0.7 0.7], ...
                'FontSize', 8, 'FontAngle', 'italic', ...
                'HorizontalAlignment', 'center');
        end

        function updateStationVisuals_(obj, nodeId)
            % UPDATESTATIONVISUALS_  Update renderer to show specific station styles.
            % Prioritizes Intechange > Terminal > Standard.
            
            lines = obj.Transit_.getLinesForStation(nodeId);
            
            isInterchange = numel(lines) > 1;
            isTerminal = false;
            
            % We determine if a station is a terminal by checking the physical 
            % spatial topology. If it connects to exactly 1 other station 
            % belonging to the same transit line, it's a terminal.
            W = obj.Transit_.getGraph().getAdjacencyMatrix();
            if nodeId <= size(W, 1)
                % Find all unique stations physically connected to this node
                neighbors = find(W(nodeId, :) > 0 | W(:, nodeId)' > 0);
                
                for i = 1:numel(lines)
                    lnCode = lines{i};
                    lnObj = obj.Transit_.getLine(lnCode);
                    
                    lineNeighborCount = 0;
                    for j = 1:numel(neighbors)
                        if lnObj.hasStation(neighbors(j))
                            lineNeighborCount = lineNeighborCount + 1;
                        end
                    end
                    
                    if lineNeighborCount == 1
                        isTerminal = true;
                        break;
                    end
                end
            end
            
            if isInterchange
                style = 'interchange';
            elseif isTerminal
                style = 'terminal';
            else
                style = 'standard';
            end
            
            % Safely cache the topological priority into the Renderer
            if nodeId > numel(obj.Renderer_.LabelIsMajor_)
                obj.Renderer_.LabelIsMajor_(nodeId) = true; % Preallocate safely padding
            end
            obj.Renderer_.LabelIsMajor_(nodeId) = (isInterchange || isTerminal);
            
            % If Standard, color it mathematically according to the line
            nodeColor = [];
            if ~isInterchange && ~isempty(lines)
                lnFirst = obj.Transit_.getLine(lines{1});
                nodeColor = lnFirst.Color;
            end
            
            obj.Renderer_.updateStationStyle(nodeId, style, nodeColor);
        end

        function onRouteLineHere_(obj)
            % ONROUTELINEHERE_  Snap the active line to pass through the selected station.
            if obj.SelectedStation_ == 0, return; end
            if isempty(obj.ActiveLineCode_), return; end
            
            obj.pushUndo_();

            pos = obj.Transit_.getStation(obj.SelectedStation_).Position;
            
            % 1. Find all edges belonging to the active line
            bestEdgeIdx = 0;
            minDist = inf;
            bestSplitParam = 0;
            
            lineObj = obj.Transit_.getLine(obj.ActiveLineCode_);
            
            for i = 1:obj.EdgeCount_
                e = obj.Transit_.getRouteEdge(i);
                if isempty(e), continue; end
                
                s = e.Source;
                t = e.Target;
                if lineObj.hasStation(s) && lineObj.hasStation(t)
                    spl = obj.Transit_.getSpline(i);
                    if ~isempty(spl)
                        tVals = spl.parameterize(200);
                        pts = spl.evaluate(tVals);
                        minDim = min(size(pts,2), numel(pos));
                        dists = sqrt(sum((pts(:, 1:minDim) - pos(1:minDim)).^2, 2));
                        [d, nearIdx] = min(dists);
                        if d < minDist
                            minDist = d;
                            bestEdgeIdx = i;
                            bestSplitParam = tVals(nearIdx);
                        end
                    end
                end
            end
            
            if bestEdgeIdx == 0
                uiwait(warndlg(sprintf('Cannot route active line "%s" here.\nNo existing edges for this line were found nearby.', obj.ActiveLineCode_), 'Routing Failed'));
                return; % Stop processing
            end
            
            spline = obj.Transit_.getSpline(bestEdgeIdx);
            e = obj.Transit_.getRouteEdge(bestEdgeIdx);
            srcId = e.Source;
            tgtId = e.Target;
            
            obj.Transit_.removeRoute(srcId, tgtId);
            obj.Renderer_.deleteCurve(bestEdgeIdx);
            
            tSplit = bestSplitParam;
            isCatmull = isa(spline, 'OceanMath.Geometry.CatmullRomSpline');
            
            % Ensure pos matches spline dimensionality (2D vs 3D)
            testPt = spline.evaluate(0);
            if numel(testPt) > numel(pos)
                pos(end+1:numel(testPt)) = 0;
            end
            if isCatmull
                t1 = linspace(0, tSplit, 4)';
                cp1 = spline.evaluate(t1);
                cp1(end, :) = pos; % Force snap to node
                spl1 = OceanMath.Geometry.CatmullRomSpline(cp1);
                
                t2 = linspace(tSplit, 1, 4)';
                cp2 = spline.evaluate(t2);
                cp2(1, :) = pos; % Force snap to node
                spl2 = OceanMath.Geometry.CatmullRomSpline(cp2);
            else
                fitCubic = @(Q) [ ...
                    Q(1,:); ...
                    (-5*Q(1,:) + 18*Q(2,:) - 9*Q(3,:) + 2*Q(4,:)) / 6; ...
                    (2*Q(1,:) - 9*Q(2,:) + 18*Q(3,:) - 5*Q(4,:)) / 6; ...
                    Q(4,:) ];

                t1 = linspace(0, tSplit, 4)';
                q1 = spline.evaluate(t1);
                q1(end, :) = pos;
                cp1 = fitCubic(q1);
                spl1 = OceanMath.Geometry.BezierCurve(cp1);

                t2 = linspace(tSplit, 1, 4)';
                q2 = spline.evaluate(t2);
                q2(1, :) = pos;
                cp2 = fitCubic(q2);
                spl2 = OceanMath.Geometry.BezierCurve(cp2);
            end
            
            obj.EdgeCount_ = obj.EdgeCount_ + 1;
            e1Idx = obj.EdgeCount_;
            obj.Transit_.addRoute(srcId, obj.SelectedStation_, spl1);
            
            obj.EdgeCount_ = obj.EdgeCount_ + 1;
            e2Idx = obj.EdgeCount_;
            obj.Transit_.addRoute(obj.SelectedStation_, tgtId, spl2);
            
            lineObj.registerStation(obj.SelectedStation_);
            obj.Transit_.invalidateLineCache();
            
            edgeColor = lineObj.Color;
            obj.Renderer_.drawCurve(e1Idx, spl1, 'Color', edgeColor);
            obj.Renderer_.drawControlPoints(e1Idx, spl1, ...
                @(eIdx, cpIdx, newPos) obj.onControlPointMoved_(eIdx, cpIdx, newPos));
                
            obj.Renderer_.drawCurve(e2Idx, spl2, 'Color', edgeColor);
            obj.Renderer_.drawControlPoints(e2Idx, spl2, ...
                @(eIdx, cpIdx, newPos) obj.onControlPointMoved_(eIdx, cpIdx, newPos));
                
            obj.Renderer_.bringStationsToFront();
            obj.updateStationVisuals_(obj.SelectedStation_);
            
            obj.ActiveLineCode_ = obj.ActiveLineCode_; % trigger ui refresh if needed
            obj.updateStatus_();
            obj.onFocusModeChanged_();
        end

        function deleteStation_(obj, nodeId)
            % DELETESTATION_  Visually remove a station and totally purge its connected curves.
            obj.pushUndo_();
            
            % 1. Find all edges connected to this station and brutally delete them
            edgesToDelete = [];
            for i = 1:obj.EdgeCount_
                e = obj.Transit_.getRouteEdge(i);
                if ~isempty(e) && (e.Source == nodeId || e.Target == nodeId)
                    edgesToDelete(end+1) = i; %#ok<AGROW>
                end
            end
            
            for i = 1:numel(edgesToDelete)
                idx = edgesToDelete(i);
                e = obj.Transit_.getRouteEdge(idx);
                % Remove from Math Engine
                obj.Transit_.removeRoute(e.Source, e.Target);
                % Remove from Vis Engine
                if ~isempty(obj.Renderer_)
                    obj.Renderer_.deleteCurve(idx);
                end
            end
            
            % 2. Visually hide the station marker (its ID technically persists in the Math Engine array
            % but it is completely disconnected and effectively dead)
            if ~isempty(obj.Renderer_)
                obj.Renderer_.deleteStation(nodeId);
            end
            
            if obj.SelectedStation_ == nodeId
                obj.SelectedStation_ = 0;
            end
            
            % Note: We don't shrink obj.StationCount_ because TrackRenderer arrays don't shrink.
            obj.updateStatus_();
        end

        function deleteNearestEdge_(obj, clickPos)
            % DELETENEARESTEDGE_  Find and remove the nearest edge to click.
            obj.pushUndo_();
            
            xl = get(obj.Axes_, 'XLim');
            yl = get(obj.Axes_, 'YLim');
            axRange = max(xl(2) - xl(1), yl(2) - yl(1));
            threshold = axRange * 0.08;

            cp = get(obj.Axes_, 'CurrentPoint');
            [edgeIdx, ~] = obj.Renderer_.findNearestCurve(cp, threshold);
            if edgeIdx == 0, return; end

            % Find source/target from the graph edge
            e = obj.Transit_.getRouteEdge(edgeIdx);
            if isempty(e), return; end

            % Remove from backend
            obj.Transit_.removeRoute(e.Source, e.Target);

            % Remove from renderer
            obj.Renderer_.deleteCurve(edgeIdx);
            obj.updateStatus_();
        end

        function insertStationOnCurve_(obj, clickPos)
            % INSERTSTATIONONCURVE_  Place a new station on the nearest curve.
            %   Splits the edge into two sub-edges.
            obj.pushUndo_();
            
            xl = get(obj.Axes_, 'XLim');
            yl = get(obj.Axes_, 'YLim');
            axRange = max(xl(2) - xl(1), yl(2) - yl(1));
            threshold = axRange * 0.08;

            cp = get(obj.Axes_, 'CurrentPoint');
            [edgeIdx, ~] = obj.Renderer_.findNearestCurve(cp, threshold);
            if edgeIdx == 0, return; end

            % Get the edge source/target
            e = obj.Transit_.getRouteEdge(edgeIdx);
            if isempty(e), return; end
            srcId = e.Source;
            tgtId = e.Target;

            % Get the spline and project click ray onto it precisely
            spline = obj.Transit_.getSpline(edgeIdx);
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
            obj.StationCount_ = obj.StationCount_ + 1;
            name = sprintf('S%d', obj.StationCount_);
            newId = obj.Transit_.addStation(name, insertPos);

            % Determine edge color (from old edge or active line)
            if edgeIdx <= numel(obj.Renderer_.CurveColors_) && ...
                    ~isempty(obj.Renderer_.CurveColors_{edgeIdx})
                edgeColor = obj.Renderer_.CurveColors_{edgeIdx};
            else
                edgeColor = [0.4, 0.8, 1.0];
            end

            % Remove old edge (backend + visual)
            obj.Transit_.removeRoute(srcId, tgtId);
            obj.Renderer_.deleteCurve(edgeIdx);

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
            obj.EdgeCount_ = obj.EdgeCount_ + 1;
            e1Idx = obj.EdgeCount_;
            obj.Transit_.addRoute(srcId, newId, spl1);

            obj.EdgeCount_ = obj.EdgeCount_ + 1;
            e2Idx = obj.EdgeCount_;
            obj.Transit_.addRoute(newId, tgtId, spl2);

            % Inherit lines from the edge we just split
            lineCodes = obj.Transit_.getLineNames();
            for lcIdx = 1:numel(lineCodes)
                lc = lineCodes{lcIdx};
                lObj = obj.Transit_.getLine(lc);
                if lObj.hasStation(srcId) && lObj.hasStation(tgtId)
                    lObj.registerStation(newId);
                end
            end
            
            % Also ensure it's registered on active line if any
            if ~isempty(obj.ActiveLineCode_) && any(strcmp(obj.Transit_.getLineNames(), obj.ActiveLineCode_))
                lineObj = obj.Transit_.getLine(obj.ActiveLineCode_);
                lineObj.registerStation(newId);
            end

            % Render new station + sub-curves
            if ~isempty(obj.Renderer_)
                node = obj.Transit_.getStation(newId);
                obj.Renderer_.drawStation(node, @(nId) obj.onStationClick_(nId));
                obj.Renderer_.drawCurve(e1Idx, spl1, 'Color', edgeColor);
                obj.Renderer_.drawCurve(e2Idx, spl2, 'Color', edgeColor);
                obj.Renderer_.drawControlPoints(e1Idx, spl1, ...
                    @(eIdx, cpIdx, newPos) obj.onControlPointMoved_(eIdx, cpIdx, newPos));
                obj.Renderer_.drawControlPoints(e2Idx, spl2, ...
                    @(eIdx, cpIdx, newPos) obj.onControlPointMoved_(eIdx, cpIdx, newPos));
                obj.Renderer_.bringStationsToFront();
            end
            
            % Update visuals to show interchange if applicable
            obj.updateStationVisuals_(newId);
            obj.updateStatus_();
        end

        function selectStation_(obj, nodeId)
            % SELECTSTATION_  Select a station for property editing.
            obj.SelectedStation_ = nodeId;

            if nodeId == 0
                set(obj.PropPanel_.nameField, 'String', '');
                if isfield(obj.PropPanel_, 'elevationField')
                    set(obj.PropPanel_.elevationField, 'String', '0');
                end
                set(obj.PropPanel_.offsetX, 'String', '0.00');
                set(obj.PropPanel_.offsetY, 'String', '0.00');
                set(obj.PropPanel_.selectedLabel, 'String', 'No station selected');
                % Remove highlights
                for i = 1:obj.StationCount_
                    h = obj.Renderer_.StationHandles_{i};
                    if ~isempty(h) && isvalid(h)
                        set(h, 'MarkerEdgeColor', obj.Renderer_.StationColor * 0.6, 'LineWidth', 1.5);
                    end
                end
                return;
            end

            station = obj.Transit_.getStation(nodeId);
            
            % Update working position cached
            pos3 = [station.Position(1), station.Position(2), 0];
            if length(station.Position) >= 3, pos3(3) = station.Position(3); end
            obj.WorkingPos_ = pos3;

            % Populate panel fields
            set(obj.PropPanel_.nameField, 'String', station.Name);

            pos = station.Position;
            z = 0; if length(pos) >= 3, z = pos(3); end
            if isfield(obj.PropPanel_, 'elevationField')
                set(obj.PropPanel_.elevationField, 'String', num2str(z));
            end

            % Get current label offset
            if nodeId <= numel(obj.Renderer_.LabelOffsets_) && ...
                    ~isempty(obj.Renderer_.LabelOffsets_{nodeId})
                offset = obj.Renderer_.LabelOffsets_{nodeId};
            else
                offset = obj.Renderer_.DefaultLabelOffset;
            end
            set(obj.PropPanel_.offsetX, 'String', sprintf('%.2f', offset(1)));
            set(obj.PropPanel_.offsetY, 'String', sprintf('%.2f', offset(2)));

            set(obj.PropPanel_.selectedLabel, 'String', ...
                sprintf('Selected: %s (ID %d)', station.Name, nodeId));

            % Visual feedback: highlight selected station
            for i = 1:obj.StationCount_
                h = obj.Renderer_.StationHandles_{i};
                if ~isempty(h) && isvalid(h)
                    if i == nodeId
                        set(h, 'MarkerEdgeColor', [1 1 0], 'LineWidth', 2.5);
                    else
                        set(h, 'MarkerEdgeColor', obj.Renderer_.StationColor * 0.6, ...
                            'LineWidth', 1.5);
                    end
                end
            end
        end

        function onApplyName_(obj)
            % ONAPPLYNAME_  Apply the new station name from the panel.
            if obj.SelectedStation_ == 0, return; end
            obj.pushUndo_();
            newName = get(obj.PropPanel_.nameField, 'String');
            obj.Transit_.updateStationName(obj.SelectedStation_, newName);
            obj.Renderer_.updateLabelText(obj.SelectedStation_, newName);
            set(obj.PropPanel_.selectedLabel, 'String', ...
                sprintf('Selected: %s (ID %d)', newName, obj.SelectedStation_));
        end

        function onApplyElevation_(obj)
            % ONAPPLYELEVATION_  Update Z-coordinate height for the node and adjacent splines.
            if obj.SelectedStation_ == 0, return; end
            valStr = get(obj.PropPanel_.elevationField, 'String');
            val = str2double(valStr);
            if isnan(val), return; end
            
            obj.pushUndo_();
            
            node = obj.Transit_.getStation(obj.SelectedStation_);
            pos = node.Position;
            if length(pos) < 3
                pos = [pos(1), pos(2), val];
            else
                pos(3) = val;
            end
            
            % 1. Mathematical Backend Update
            obj.Transit_.updateStationPosition(obj.SelectedStation_, pos);
            
            % 2. Visual Backend Update
            sH = obj.Renderer_.StationHandles_{obj.SelectedStation_};
            if ~isempty(sH) && isvalid(sH)
                set(sH, 'ZData', val);
            end
            lH = obj.Renderer_.LabelHandles_{obj.SelectedStation_};
            if ~isempty(lH) && isvalid(lH)
                lp = get(lH, 'Position');
                set(lH, 'Position', [lp(1), lp(2), val]);
            end
            
            % 3. Refresh adjacent Splines mathematically to curve to the new Z point!
            edges = obj.Transit_.getGraph().getEdges();
            for i = 1:numel(edges)
                e = edges{i};
                if ~isempty(e) && (e.Source == obj.SelectedStation_ || e.Target == obj.SelectedStation_)
                    splineData = obj.Transit_.getGraph().getSplineData(i);
                    if ~isempty(splineData)
                        cps = splineData.getControlPoints();
                        if size(cps, 2) < 3
                            cps = [cps, zeros(size(cps,1), 1)]; %#ok<AGROW>
                        end
                        if e.Source == obj.SelectedStation_
                            cps(1, :) = pos;
                        else
                            cps(end, :) = pos;
                        end
                        if isa(splineData, 'OceanMath.Geometry.CatmullRomSpline')
                            newSpline = OceanMath.Geometry.CatmullRomSpline(cps);
                        else
                            newSpline = OceanMath.Geometry.BezierCurve(cps);
                        end
                        obj.Transit_.updateRouteSpline(i, newSpline);
                        obj.Renderer_.updateCurve(i, newSpline);
                        if obj.ActiveEditCurve_ == i
                            obj.Renderer_.updateControlPoints(i, newSpline);
                        end
                    end
                end
            end
            obj.updateStatus_();
        end

        function onApplyOffset_(obj)
            % ONAPPLYOFFSET_  Apply label offset from the panel fields.
            if obj.SelectedStation_ == 0, return; end
            dx = str2double(get(obj.PropPanel_.offsetX, 'String'));
            dy = str2double(get(obj.PropPanel_.offsetY, 'String'));
            if isnan(dx), dx = 0; end
            if isnan(dy), dy = 0.3; end
            obj.Renderer_.setLabelOffset(obj.SelectedStation_, [dx, dy]);
        end

        function onApplyAllOffsets_(obj)
            % ONAPPLYALLOFFSETS_  Apply current offset to all stations.
            dx = str2double(get(obj.PropPanel_.offsetX, 'String'));
            dy = str2double(get(obj.PropPanel_.offsetY, 'String'));
            if isnan(dx), dx = 0; end
            if isnan(dy), dy = 0.3; end
            obj.Renderer_.setAllLabelOffsets([dx, dy]);
        end

        % ---- Draw Order Callbacks ---- %
        function onCurveDrawOrderUp_(obj)
            % ONCURVEDRAWORDERUP_  Increment draw order of the active curve.
            if obj.ActiveEditCurve_ == 0, return; end
            eIdx = obj.ActiveEditCurve_;
            current = obj.Renderer_.getCurveDrawOrder(eIdx);
            newOrder = current + 1;
            obj.Renderer_.setCurveDrawOrder(eIdx, newOrder);
            % Create halo if crossing from 0 to 1
            if current == 0
                obj.Renderer_.ensureHalo(eIdx);
            end
            obj.Renderer_.reorderCurves();
            set(obj.PropPanel_.drawOrderLabel, 'String', ...
                sprintf('Active Curve: Edge %d  |  Order: %d', eIdx, newOrder));
        end

        function onCurveDrawOrderDown_(obj)
            % ONCURVEDRAWORDERDOWN_  Decrement draw order of the active curve.
            if obj.ActiveEditCurve_ == 0, return; end
            eIdx = obj.ActiveEditCurve_;
            current = obj.Renderer_.getCurveDrawOrder(eIdx);
            newOrder = max(0, current - 1);
            obj.Renderer_.setCurveDrawOrder(eIdx, newOrder);
            % Remove halo if dropping to 0
            if newOrder == 0
                obj.Renderer_.removeHalo(eIdx);
            end
            obj.Renderer_.reorderCurves();
            set(obj.PropPanel_.drawOrderLabel, 'String', ...
                sprintf('Active Curve: Edge %d  |  Order: %d', eIdx, newOrder));
        end

        % ---- Phase 1: Line Management Callbacks ---- %
        function onCreateLine_(obj)
            % ONCREATELINE_  Create a new transit line from panel fields.
            code = strtrim(get(obj.LinePanel_.codeField, 'String'));
            name = strtrim(get(obj.LinePanel_.nameField, 'String'));
            if isempty(code), return; end

            % Color palette corresponding to dropdown strings
            palette = [
                0.85, 0.20, 0.20;   % Red (NSL)
                0.00, 0.60, 0.30;   % Green (EWL)
                0.60, 0.20, 0.70;   % Purple (NEL)
                0.95, 0.55, 0.10;   % Orange (CCL)
                0.00, 0.35, 0.75;   % Blue (DTL)
                0.60, 0.35, 0.15;   % Brown (TEL)
                0.00, 0.65, 0.65;   % Teal (JRL)
                0.60, 0.85, 0.20;   % Lime (CRL)
                0.45, 0.50, 0.50;   % Grey (LRT)
                0.10, 0.75, 0.85;   % Cyan
                0.90, 0.85, 0.10;   % Yellow
                0.85, 0.20, 0.65;   % Magenta
                0.95, 0.50, 0.65;   % Pink
            ];
            colorIdx = get(obj.LinePanel_.colorPicker, 'Value');
            color = palette(colorIdx, :);

            % Add line to network
            obj.Transit_.addLine(code, name, color);

            % Update active line dropdown
            obj.refreshLineDropdown_();

            % Auto-select the new line as active
            items = get(obj.LinePanel_.activeDropdown, 'String');
            lineIdx = find(strcmp(items, sprintf('%s (%s)', code, name)));
            if ~isempty(lineIdx)
                set(obj.LinePanel_.activeDropdown, 'Value', lineIdx(1));
                obj.ActiveLineCode_ = code;
                set(obj.LinePanel_.colorIndicator, ...
                    'BackgroundColor', color, 'String', '');
            end

            obj.updateStatus_();
            
            % Trigger focus update
            obj.onFocusModeChanged_();
        end


        function refreshLineDropdown_(obj)
            % REFRESHLINEDROPDOWN_  Rebuild the active-line dropdown items.
            keys = obj.Transit_.getLineNames();
            items = {'(none)'};
            for i = 1:numel(keys)
                k = keys{i};
                lineObj = obj.Transit_.getLine(k);
                items{end+1} = sprintf('%s (%s)', k, lineObj.Name); %#ok<AGROW>
            end
            
            % Safely update to avoid "Value out of range" UI warnings
            currentVal = get(obj.LinePanel_.activeDropdown, 'Value');
            set(obj.LinePanel_.activeDropdown, 'Value', 1);
            set(obj.LinePanel_.activeDropdown, 'String', items);
            
            if currentVal <= numel(items)
                set(obj.LinePanel_.activeDropdown, 'Value', currentVal);
            else
                obj.ActiveLineCode_ = '';
                set(obj.LinePanel_.colorIndicator, ...
                    'BackgroundColor', [0.15, 0.15, 0.18], 'String', '');
            end
        end

        function onSelectActiveLine_(obj, src)
            % ONSELECTACTIVELINE_  User picked a line from the dropdown.
            idx = get(src, 'Value');
            items = get(src, 'String');
            if idx == 1
                obj.ActiveLineCode_ = '';
                set(obj.LinePanel_.colorIndicator, ...
                    'BackgroundColor', [0.15, 0.15, 0.18], 'String', '');
            else
                selected = items{idx};
                code = strtrim(extractBefore(selected, ' ('));
                obj.ActiveLineCode_ = code;
                lineObj = obj.Transit_.getLine(code);
                set(obj.LinePanel_.colorIndicator, ...
                    'BackgroundColor', lineObj.Color, 'String', '');
            end
            obj.updateStatus_();
            
            % Trigger focus update if active
            obj.onFocusModeChanged_();
        end

        % ---- Phase 3: Curve Type Callback ---- %
        function onFocusModeChanged_(obj)
            % ONFOCUSMODECHANGED_  Toggles line isolation.
            if isempty(obj.FocusCheckbox_) || ~isvalid(obj.FocusCheckbox_)
                return;
            end
            
            isFocus = get(obj.FocusCheckbox_, 'Value');
            activeCode = obj.ActiveLineCode_;
            
            if isempty(obj.Renderer_)
                return;
            end
            
            obj.Renderer_.applyFocus(isFocus, activeCode, obj.Transit_);
            
            % Update selection visuals carefully
            if isFocus && obj.SelectedStation_ > 0 && ~isempty(activeCode)
                lObj = obj.Transit_.getLine(activeCode);
                if ~lObj.hasStation(obj.SelectedStation_)
                    obj.selectStation_(0);
                end
            end
        end

        function onSelectCurveType_(obj, src)
            idx = get(src, 'Value');
            switch idx
                case 1, obj.CurveType_ = 'catmullrom';
                case 2, obj.CurveType_ = 'bezier';
                case 3, obj.CurveType_ = 'linear';
            end
        end

        % ---- Phase 4 & 5: Visual Toggles ---- %
        function onToggleLabels_(obj, val)
            obj.Renderer_.toggleLabels(logical(val));
        end

        function onToggleStations_(obj, val)
            obj.Renderer_.toggleStations(logical(val));
        end

        function toggle3DView_(obj, src)
            % Toggles MATLAB's native rotate3d camera interceptor
            val = get(src, 'Value');
            if val
                view(obj.Axes_, 3);
                rotate3d(obj.Axes_, 'on');
                % Hint user via status bar
                set(obj.StatusText_, 'String', 'Mode: 3D CAMERA ACTIVE  |  Click and drag Canvas to rotate view. Toggle button OFF to edit tracks.');
            else
                rotate3d(obj.Axes_, 'off');
                obj.updateStatus_();
            end
        end

        function onApplyFontSize_(obj)
            sz = str2double(get(obj.PropPanel_.fontSizeField, 'String'));
            if isnan(sz) || sz <= 0, sz = 9; end
            obj.Renderer_.setLabelFontSize(sz);
        end

        function updateParallelStyles_(obj)
            % UPDATEPARALLELSTYLES_ Dynamically layer widths for overlapping routes.
            E = obj.EdgeCount_;
            
            corridorMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
            for i = 1:E
                e = obj.Transit_.getRouteEdge(i);
                if isempty(e), continue; end
                
                key = sprintf('%d-%d', min(e.Source, e.Target), max(e.Source, e.Target));
                if corridorMap.isKey(key)
                    corridorMap(key) = [corridorMap(key), i];
                else
                    corridorMap(key) = i;
                end
            end
            
            keys = corridorMap.keys();
            for ki = 1:numel(keys)
                edgeIndices = corridorMap(keys{ki});
                k = numel(edgeIndices);
                
                % Layer widths: thickest on bottom (li=1), thinnest on top (li=k)
                % Layer widths: thickest on bottom (li=1), thinnest on top (li=k)
                baseWidth = obj.Renderer_.CurveLineWidth; % 2.5
                step = 5.0; % Increased from 3.5 to make the outer shell thicker and more visible
                
                % The first line drawn in this corridor dictates the physical shape
                masterSpl = obj.Transit_.getSpline(edgeIndices(1));
                
                for li = 1:k
                    ei = edgeIndices(li);
                    
                    % 1. Mathematical Synchronization
                    % Force slave lines to mathematically snap to the master line's shape
                    if li > 1 && ~isempty(masterSpl)
                        slaveSpl = eval(sprintf('%s(masterSpl.getControlPoints())', class(masterSpl)));
                        obj.Transit_.updateRouteSpline(ei, slaveSpl);
                    end
                    
                    % 2. Visual Layering
                    targetWidth = baseWidth + (k - li) * step;
                    
                    oldWidth = baseWidth;
                    if ei <= numel(obj.Renderer_.CurveWidths_) && ~isempty(obj.Renderer_.CurveWidths_{ei})
                        oldWidth = obj.Renderer_.CurveWidths_{ei};
                    end
                    
                    if oldWidth ~= targetWidth || li > 1
                        obj.Renderer_.setCurveWidth(ei, targetWidth);
                        spl = obj.Transit_.getSpline(ei);
                        if ~isempty(spl)
                            obj.Renderer_.updateCurve(ei, spl);
                        end
                    end
                end
            end
        end
    end
    
    % ================================================================== %
    methods (Static)
        function tn = loadNetwork(filepath)
            % LOADNETWORK  Reconstruct a strictly mathematical TransitNetwork from a TrackDesigner JSON map.
            %   Ignores GUI-specific visual attributes completely and reconstructs
            %   the underlying routing graph and parametric splines for simulations.
            text = fileread(filepath);
            data = jsondecode(text);
            
            tn = OceanMath.Transit.TransitNetwork();
            
            idMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
            
            % 1. Rebuild Stations
            if isfield(data, 'stations') && ~isempty(data.stations)
                for i = 1:numel(data.stations)
                    s = data.stations(i);
                    newId = tn.addStation(s.name, s.position(:)');
                    if isfield(s, 'id')
                        idMap(s.id) = newId; % Map legacy explicit ID to strictly sequential ID
                    else
                        idMap(i) = newId; % Fallback
                    end
                end
            end
            
            % 2. Rebuild Edges (with physics splines)
            if isfield(data, 'edges') && ~isempty(data.edges)
                for i = 1:numel(data.edges)
                    e = data.edges(i);
                    if ~idMap.isKey(e.source) || ~idMap.isKey(e.target)
                        continue; % Skip mathematically broken ghost edges
                    end
                    sId = idMap(e.source);
                    tId = idMap(e.target);
                    
                    if isfield(e, 'controlPoints') && ~isempty(e.controlPoints)
                        if isfield(e, 'curveType') && strcmpi(e.curveType, 'bezier')
                            spline = OceanMath.Geometry.BezierCurve(e.controlPoints);
                        else
                            spline = OceanMath.Geometry.CatmullRomSpline(e.controlPoints);
                        end
                        tn.addRoute(sId, tId, spline);
                    else
                        tn.addSimpleRoute(sId, tId, e.weight);
                    end
                end
            end
            
            % 3. Rebuild Lines
            if isfield(data, 'lines') && ~isempty(data.lines)
                for i = 1:numel(data.lines)
                    li = data.lines(i);
                    tn.addLine(li.code, li.name, li.color(:)');
                    lineObj = tn.getLine(li.code);
                    if isfield(li, 'route') && ~isempty(li.route)
                        for j = 1:numel(li.route)
                            if idMap.isKey(li.route(j))
                                lineObj.registerStation(idMap(li.route(j)));
                            end
                        end
                    end
                end
            end
        end
    end
end

