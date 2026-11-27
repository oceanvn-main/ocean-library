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
    %           OceanViz.Designer.Interaction.DraggablePoint

    % ------------------------------------------------------------------ %
    properties
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
        ActiveMode_     % Current IDesignerMode instance
        SelectedStation_ (1,1) double = 0 % currently selected station in edit mode

        LinePanel_      (1,1) struct
        PropPanel_      (1,1) struct

        ActiveLineCode_ (1,:) char = ''
        
        WorkingPos_      (1,3) double = [0, 0, 0] % Caches last edited position for 3D placement
        % Phase 3: Curve type
        CurveType_      (1,:) char = 'catmullrom'  % 'catmullrom' or 'bezier'
        FocusCheckbox_  % UI handle for Focus Mode
        InteractionHandler_ % Phase 2 handler
        OverlayRenderer_    % Phase 2 Overlay Handler
        SpatialIdx_         % OceanMath.Geometry.QuadTree — O(log N) station lookup

        % Pan/zoom state
        PanOrigin_      (1,2) double = [0 0]  % mouse start for panning
        IsPanning_      (1,1) logical = false

        % Centralized Event Bus for Local Undos
        TimelineController_
        
        % Extracted MVC Controllers
        WorkspaceManager_
        PropertyController_
        GeometryBuilder_

        % Weight display

        % Map title (draggable)
        TitleHandle_               % text handle for map title
    end

    properties (SetAccess = public)
        ActiveEditCurve_ (1,1) double = 0  % Shared with EditMode for draw order UI
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
            arguments
                obj
            end
            obj.Figure_ = figure('Name', 'Track Designer', ...
                'NumberTitle', 'off', ...
                'Color', [0.12, 0.12, 0.14], ...
                'MenuBar', 'none', ...
                'ToolBar', 'none', ...
                'WindowState', 'maximized', ...
                'CloseRequestFcn', 'closereq');
                
            % Force MATLAB UI thread to handshake and initialize the JS renderer immediately
            drawnow;
            pause(0.1);

            % Main axes — maximized, limits set dynamically after data load
            obj.Axes_ = axes(obj.Figure_, ...
                'Position', [0.03, 0.04, 0.67, 0.895], ... % Expanded left/bottom padding for tick numbers
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
            cbToolbar = struct( ...
                'setMode', @(mode) obj.InteractionHandler_.setMode_(mode), ...
                'onUndo', @() obj.TimelineController_.undo(), ...
                'undo', @() obj.TimelineController_.undo(), ...
                'redo', @() obj.TimelineController_.redo(), ...
                'onExport', @() obj.WorkspaceManager_.onExport(), ...
                'onNew', @() obj.WorkspaceManager_.onNew(), ...
                'onSave', @() obj.WorkspaceManager_.onSave(), ...
                'onLoad', @() obj.WorkspaceManager_.onLoad(), ...
                'onLaunchSim', @() obj.onLaunchSim_(), ...
                'toggleLabels', @(val) obj.onToggleLabels_(val), ...
                'toggleAllLabels', @(val) obj.onToggleAllLabels_(val), ...
                'toggleStations', @(val) obj.onToggleStations_(val), ...
                'toggle3DView', @(src) obj.toggle3DView_(src), ...
                'toggleRotate3D', @(src) obj.toggleRotate3D_(src), ...
                'viewAxes', obj.Axes_, ...
                'toggleWeights', @(val) obj.OverlayRenderer_.onToggleWeights_(val), ...
                'toggleWeightUnit', @(src) obj.OverlayRenderer_.onToggleWeightUnit_(src), ...
                'onSetScale', @() obj.onSetScale_() );
            obj.ToolButtons_ = OceanViz.Designer.Layout.ToolbarBuilder.buildToolbar(obj.Figure_, cbToolbar);

            % Property panel (right side)
            cbProp = struct( ...
                'onCreateLine', @() obj.PropertyController_.onCreateLine(), ...
                'onSelectActiveLine', @(src) obj.PropertyController_.onSelectActiveLine(src), ...
                'onFocusModeChanged', @() obj.PropertyController_.onFocusModeChanged(), ...
                'onSelectCurveType', @(src) obj.PropertyController_.onSelectCurveType(src), ...
                'onApplyName', @() obj.PropertyController_.onApplyName(), ...
                'onApplyElevation', @() obj.PropertyController_.onApplyElevation(), ...
                'onApplyOffset', @() obj.PropertyController_.onApplyOffset(), ...
                'onApplyAllOffsets', @() obj.PropertyController_.onApplyAllOffsets(), ...
                'onApplyFontSize', @() obj.PropertyController_.onApplyFontSize(), ...
                'onRouteLineHere', @() obj.PropertyController_.onRouteLineHere(), ...
                'onCurveDrawOrderUp', @() obj.PropertyController_.onCurveDrawOrderUp(), ...
                'onCurveDrawOrderDown', @() obj.PropertyController_.onCurveDrawOrderDown() );
            [obj.LinePanel_, obj.PropPanel_, obj.FocusCheckbox_] = ...
                OceanViz.Designer.Layout.ToolbarBuilder.buildPropertyPanel(obj.Figure_, cbProp);

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
            obj.InteractionHandler_ = OceanViz.Designer.Interaction.InteractionHandler(obj);
            obj.TimelineController_ = OceanViz.Designer.Commands.TimelineController();
            obj.WorkspaceManager_ = OceanViz.Designer.Persistence.WorkspaceManager(obj);
            obj.PropertyController_ = OceanViz.Designer.Interaction.PropertyController(obj);
            
            % Overlay Handlers
            obj.OverlayRenderer_ = OceanViz.Designer.Layout.OverlayRenderer(obj);
            set(obj.Axes_, 'ButtonDownFcn', @(~,~) obj.InteractionHandler_.onAxesClick_());

            % Scroll wheel zoom (zoom toward cursor)
            set(obj.Figure_, 'WindowScrollWheelFcn', @(~,evt) obj.InteractionHandler_.onScrollZoom_(evt));

            % Right-click drag to pan
            set(obj.Figure_, 'WindowButtonDownFcn',   @(~,~) obj.InteractionHandler_.onPanStart_());
            set(obj.Figure_, 'WindowButtonUpFcn',     @(~,~) obj.InteractionHandler_.onPanEnd_());
            
            % Listen for zoom/pan scaling
            addlistener(obj.Axes_, 'XLim', 'PostSet', @(~,~) obj.InteractionHandler_.onZoomCheck_());

            % Ctrl+Z for undo
            set(obj.Figure_, 'KeyPressFcn', @(~,evt) obj.InteractionHandler_.onKeyPress_(evt));

            % Initialize default mode state
            obj.InteractionHandler_.setMode_('place');

            obj.updateStatus_();
        end
        
        function onLaunchSim_(obj)
            % ONLAUNCHSIM_  Open the Transit Simulator App.
            OceanViz.Simulator.TransitSimulatorApp();
        end
        
        function onSetScale_(obj)
            % ONSETSCALE_ Prompt the user for real-world vs math-grid conversion.
            prompt = {'Enter real-world distance inside 100 Grid Units (e.g., if 100 units = 2.5 km, enter 2.5):'};
            dlgtitle = 'Map Math to Distance Scale';
            dims = [1 70];
            
            % Present the current mathematically derived scale as the default
            currentRealDistancePer100 = 100 / obj.OverlayRenderer_.UnitsPerKm_;
            definput = {sprintf('%.2f', currentRealDistancePer100)};
            
            answer = inputdlg(prompt, dlgtitle, dims, definput);
            if isempty(answer), return; end
            
            kmPer100 = str2double(answer{1});
            if ~isnan(kmPer100) && kmPer100 > 0
                % Calculate the ratio: how many units equals exactly 1.0 km?
                obj.OverlayRenderer_.UnitsPerKm_ = 100 / kmPer100;
                
                % Redraw immediately if the weights overlay is turned on
                if obj.OverlayRenderer_.ShowWeights_
                    obj.OverlayRenderer_.drawWeightLabels_();
                end
            end
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
            %record in the Math Engine
            id = obj.Transit_.addStation(name, position);
            obj.StationCount_ = obj.StationCount_ + 1;
            
            % Update working pos cache
            pos3 = [position(1), position(2), 0];
            if length(position) >= 3, pos3(3) = position(3); end
            obj.WorkingPos_ = pos3;

            % Rebuild spatial index for O(log N) lookups
            obj.rebuildSpatialIndex_();

            % Render if GUI is open
            if ~isempty(obj.Renderer_)
                node = obj.Transit_.getStation(id); %Ask Math
                obj.Renderer_.drawStation(node, @(nId) obj.InteractionHandler_.onStationClick_(nId)); %Handle to render, also inject mouse click listener
                
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
            
            % obj.pushUndo_(); % Replaced by specific localized AddEdgeCommand
            %Ask Math
            pos1 = obj.Transit_.getStation(id1).Position;
            pos2 = obj.Transit_.getStation(id2).Position;

            % Generate Spline Math using GeometryBuilder
            spline = OceanViz.Designer.Math.GeometryBuilder.generateConnectionSpline(pos1, pos2, obj.CurveType_, options.NumMidpoints);

            % Build execution options for the command
            edgeOpts = struct();
            edgeOpts.Label = options.Label;
            if ~isempty(obj.ActiveLineCode_) && any(strcmp(obj.Transit_.getLineNames(), obj.ActiveLineCode_))
                edgeOpts.Color = obj.Transit_.getLine(obj.ActiveLineCode_).Color;
            else
                edgeOpts.Color = [0.4, 0.8, 1.0]; % default cyan
            end
            
            % Dispatch Command
            cmd = OceanViz.Designer.Commands.AddEdgeCommand(obj, id1, id2, spline, edgeOpts);
            if ~isempty(obj.TimelineController_)
                obj.TimelineController_.execute(cmd);
            else
                cmd.execute();
            end
            
            % EdgeIdx is calculated inside Command execution
            edgeIdx = cmd.EdgeIdx_;
            
            % Only log the rest state explicitly
            if ~isempty(obj.ActiveLineCode_) && any(strcmp(obj.Transit_.getLineNames(), obj.ActiveLineCode_))
                lineObj = obj.Transit_.getLine(obj.ActiveLineCode_);
                lineObj.registerStation(id1);
                lineObj.registerStation(id2);
                obj.updateStationVisuals_(id1);
                obj.updateStationVisuals_(id2);
            end
        end

        function g = exportGraph(obj)
            % EXPORTGRAPH  Return the current NetworkGraph.
            arguments
                obj
            end
            g = obj.Transit_.getGraph();
        end

        function tn = exportTransitNetwork(obj)
            % EXPORTTRANSITNETWORK  Return the TransitNetwork wrapper.
            arguments
                obj
            end
            tn = obj.Transit_;
        end

        function s = getDesignStruct_(obj)
            % GETDESIGNSTRUCT_  Delegates to DesignerSerializer.
            s = OceanViz.Designer.Persistence.DesignerSerializer.toStruct(obj);
        end

        function saveDesign(obj, filepath)
            % SAVEDESIGN  Save the track design to a .mat file.
            arguments
                obj
                filepath (1,:) char
            end
            OceanViz.Designer.Persistence.DesignerSerializer.saveMAT( ...
                filepath, obj);
        end

        function loadDesignStruct_(obj, dataStr)
            % LOADDESIGNSTRUCT_  Delegates to DesignerSerializer.
            OceanViz.Designer.Persistence.DesignerSerializer.loadFromStruct(dataStr, obj);
        end

        function loadDesign(obj, filepath)
            % LOADDESIGN  Load a track design from a .mat file.
            arguments
                obj
                filepath (1,:) char
            end
            OceanViz.Designer.Persistence.DesignerSerializer.loadMAT(filepath, obj);
        end

        function describe(obj)
            % DESCRIBE  Print summary.
            arguments
                obj
            end
            fprintf('TrackDesigner:\n');
            fprintf('  Stations: %d\n', obj.StationCount_);
            fprintf('  Edges:    %d\n', obj.EdgeCount_);
            fprintf('  Mode:     %s\n', obj.Mode_);
            obj.Transit_.describe();
        end
    end

    % ------------------------------------------------------------------ %
    methods

        % ---- State Reset (used by DesignerSerializer) ---- %
        function resetState_(obj)
            % RESETSTATE_  Clear transit data and renderer for fresh rebuild.
            obj.Transit_ = OceanMath.Transit.TransitNetwork();
            obj.StationCount_ = 0;
            obj.EdgeCount_ = 0;
            obj.SpatialIdx_ = [];  % Purge spatial index
            if ~isempty(obj.Renderer_)
                obj.Renderer_.clearAll();
            end
            if ~isempty(obj.OverlayRenderer_)
                obj.OverlayRenderer_.clearAll();
            end
        end

        function rebuildSpatialIndex_(obj)
            % REBUILDSPATIALINDEX_  Reconstruct QuadTree from current stations.
            if obj.StationCount_ == 0
                obj.SpatialIdx_ = [];
                return;
            end
            positions = obj.Transit_.getStationPositions();
            ids = (1:size(positions, 1))';
            obj.SpatialIdx_ = OceanMath.Geometry.QuadTree(positions(:,1:2), ids);
        end
        
        %Notification dashboard
        function updateStatus_(obj)
            % Build active line suffix
            if ~isempty(obj.ActiveLineCode_)
                lineSuffix = sprintf('  [Line: %s]', obj.ActiveLineCode_);
            else
                lineSuffix = '';
            end
            
            if ~isempty(obj.ActiveMode_)
                msg = obj.ActiveMode_.getStatusText(obj);
            else
                msg = ['Mode: ', upper(obj.Mode_), lineSuffix];
            end
            
            % Refresh layered thicknesses if any routes overlap
            obj.updateParallelStyles_();
            
            if ~isempty(obj.StatusText_) && isvalid(obj.StatusText_)
                set(obj.StatusText_, 'String', msg);
            end
        end


        % ---- Undo System ---- %
        function clearUndo_(obj)
            % CLEARUNDO_  Flush history stack if an unpredictable bulk action occurred.
            if ~isempty(obj.TimelineController_)
                obj.TimelineController_.clear();
            end
        end


        function onClose_(obj)
            delete(obj);
        end

        function delete(obj)
            % DELETE  Destructor. Purges RAM and cleanly closes the UI window.
            if ~isempty(obj.Figure_) && isgraphics(obj.Figure_) && isvalid(obj.Figure_)
                delete(obj.Figure_);
            end
        end


        function updateAxisLimits_(obj)
            %  UPDATEAXISLIMITS_  Expand viewport only if stations fall outside.
            %   Never shrinks the current view — avoids jarring camera jumps.
            if obj.StationCount_ < 2, return; end
            positions = obj.Transit_.getStationPositions();
            
            xMin = min(positions(:,1));  xMax = max(positions(:,1));
            yMin = min(positions(:,2));  yMax = max(positions(:,2));
            zMin = 0; zMax = 0;
            if size(positions, 2) >= 3
                zMin = min(positions(:,3));  zMax = max(positions(:,3));
            end

            %How far apart two stations are, at least 1
            xSpread = max(xMax - xMin, 1);
            ySpread = max(yMax - yMin, 1);
            zSpread = max(zMax - zMin, 1);
            
            % Enforce a perfectly cubic spatial volume!
            % This guarantees the floor grid remains a pristine square and 
            % the Z working volume never becomes a skyscraper or a flat pancake.
            cubicSize = max([xSpread, ySpread, zSpread]); %Take the largest spread

            %Find midpoint of stations in each axis
            xCenter = (xMax + xMin) / 2;
            yCenter = (yMax + yMin) / 2;
            zCenter = (zMax + zMin) / 2;
            
            % Add a 10% padding boundary around the cube
            pad = cubicSize * 0.10;
            halfSize = (cubicSize / 2) + pad; %Distance from center to each war of viewing box
            
            dataXL = [xCenter - halfSize, xCenter + halfSize];
            dataYL = [yCenter - halfSize, yCenter + halfSize];
            dataZL = [zCenter - halfSize, zCenter + halfSize];

            if obj.StationCount_ == 2
                % First time computing limits exactly around the first 2 stations
                newXL = dataXL;
                newYL = dataYL;
                newZL = dataZL;
            else
                % Get current view bounds and only expand (never shrink)
                curXL = xlim(obj.Axes_); %What are camera currently show
                curYL = ylim(obj.Axes_);
                curZL = zlim(obj.Axes_);
                newXL = [min(curXL(1), dataXL(1)),  max(curXL(2), dataXL(2))];
                newYL = [min(curYL(1), dataYL(1)),  max(curYL(2), dataYL(2))];
                newZL = [min(curZL(1), dataZL(1)),  max(curZL(2), dataZL(2))];
            end

            xlim(obj.Axes_, newXL);
            ylim(obj.Axes_, newYL);
            zlim(obj.Axes_, newZL);

            % Lock 1:1:1 data ratio WITHOUT overriding our limits.
            % axis('equal') destructively rewrites xlim/ylim/zlim — daspect does not.
            daspect(obj.Axes_, [1 1 1]);
        end

        % ---- Property Panel ---- %

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



        function deleteStation_(obj, nodeId, isSilent)
            % DELETESTATION_  Visually remove a station and totally purge its connected curves.
            if nargin < 3, isSilent = false; end
            
            if ~isSilent
                % Confirm deletion and wipe undo stack
                ansStr = questdlg('Delete this station? (Cannot be undone)', 'Confirm', 'Yes', 'No', 'No');
                if ~strcmp(ansStr, 'Yes'), return; end
                
                if ~isempty(obj.TimelineController_)
                    obj.TimelineController_.clear();
                end
            end
            
            % 1. Find all edges connected to this station and brutally delete them
            % Uses pure O(1) Graph Theory Topology instead of linear database scan
            nbrs = obj.Transit_.getGraph().neighbors(nodeId);
            edgesToDelete = zeros(1, numel(nbrs));
            delCount = 0;
            for j = 1:numel(nbrs)
                eId = obj.Transit_.findEdgeIndex(nodeId, nbrs(j));
                if eId > 0
                    delCount = delCount + 1;
                    edgesToDelete(delCount) = eId;
                end
            end
            edgesToDelete = edgesToDelete(1:delCount);
            
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
            
            % 2. Unregister the station from ALL transit lines
            %    Without this, the line's route keeps the dead ID and
            %    reconcileLineRoutes() reports it as orphaned on export.
            codes = obj.Transit_.getLineNames();
            for ci = 1:numel(codes)
                lineObj = obj.Transit_.getLine(codes{ci});
                if lineObj.hasStation(nodeId)
                    lineObj.removeStation(nodeId);
                end
            end
            obj.Transit_.invalidateLineCache();
            
            % 3. Visually hide the station marker (its ID technically persists in the Math Engine array
            % but it is completely disconnected and effectively dead)
            if ~isempty(obj.Renderer_)
                obj.Renderer_.deleteStation(nodeId);
            end
            
            if obj.SelectedStation_ == nodeId
                obj.SelectedStation_ = 0;
            end
            
            % Note: We don't shrink obj.StationCount_ because TrackRenderer arrays don't shrink.
            obj.rebuildSpatialIndex_();
            obj.updateStatus_();
        end



        function selectStation_(obj, nodeId)
            % SELECTSTATION_  Select a station for property editing.
            oldId = obj.SelectedStation_;
            obj.SelectedStation_ = nodeId;

            if nodeId == 0
                set(obj.PropPanel_.nameField, 'String', '');
                if isfield(obj.PropPanel_, 'elevationField')
                    set(obj.PropPanel_.elevationField, 'String', '0');
                end
                set(obj.PropPanel_.offsetX, 'String', '0.00');
                set(obj.PropPanel_.offsetY, 'String', '0.00');
                set(obj.PropPanel_.selectedLabel, 'String', 'No station selected');
                % Remove highlight from previously selected station
                if oldId > 0 && oldId <= obj.StationCount_
                    hOld = obj.Renderer_.StationHandles_{oldId};
                    if ~isempty(hOld) && isvalid(hOld)
                        set(hOld, 'MarkerEdgeColor', obj.Renderer_.StationColor * 0.6, 'LineWidth', 1.5);
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

            % O(1) Visual feedback: Demote old station, highlight new station
            if oldId > 0 && oldId ~= nodeId && oldId <= obj.StationCount_
                hOld = obj.Renderer_.StationHandles_{oldId};
                if ~isempty(hOld) && isvalid(hOld)
                    set(hOld, 'MarkerEdgeColor', obj.Renderer_.StationColor * 0.6, 'LineWidth', 1.5);
                end
            end
            
            hNew = obj.Renderer_.StationHandles_{nodeId};
            if ~isempty(hNew) && isvalid(hNew)
                set(hNew, 'MarkerEdgeColor', [1 1 0], 'LineWidth', 2.5);
            end
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



        % ---- Phase 4 & 5: Visual Toggles ---- %
        function onToggleLabels_(obj, val)
            obj.Renderer_.toggleLabels(logical(val));
        end
        
        function onToggleAllLabels_(obj, val)
            obj.Renderer_.toggleAllLabels(logical(val));
        end

        function onToggleStations_(obj, val)
            obj.Renderer_.toggleStations(logical(val));
        end

        function toggle3DView_(obj, src)
            % Toggles MATLAB's native rotate3d camera interceptor
            val = get(src, 'Value');
            if val
                view(obj.Axes_, 3);
                %rotate3d(obj.Axes_, 'on');
                % Hint user via status bar
                set(obj.StatusText_, 'String', 'Mode: 3D CAMERA ACTIVE  |  Click and drag Canvas to rotate view. Toggle button OFF to edit tracks.');
            else
                rotate3d(obj.Axes_, 'off');
                view(obj.Axes_, 2); %Return to top-down 2D
                obj.updateStatus_();
            end
        end

        function toggleRotate3D_(obj, src)
            val = get(src, 'Value');
            if val
                rotate3d(obj.Axes_, 'on');
                set(obj.StatusText_, 'String', ...
                    'Mode: ROTATE CAMERA  |  Drag to orbit. Toggle OFF to resume editing.');
            else
                rotate3d(obj.Axes_, 'off');
                % Restore callbacks stolen by rotate3d
                set(obj.Axes_, 'ButtonDownFcn', ...
                    @(~,~) obj.InteractionHandler_.onAxesClick_());
                set(obj.Figure_, 'WindowButtonDownFcn', ...
                    @(~,~) obj.InteractionHandler_.onPanStart_());
                set(obj.Figure_, 'WindowButtonUpFcn', ...
                    @(~,~) obj.InteractionHandler_.onPanEnd_());
                set(obj.Figure_, 'WindowScrollWheelFcn', ...
                    @(~,evt) obj.InteractionHandler_.onScrollZoom_(evt));
                obj.updateStatus_();
            end
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
                        %Get a clone of the master
                        slaveSpl = eval(sprintf('%s(masterSpl.getControlPoints())', class(masterSpl))); %class(masterSpl) : what type of class are you?
                        %'OceanMath.Geometry.[e.g...BezierSpline](masterSpl.getControlPoints())'
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
                            % li=1 (thickest) goes base, li=2,3 go slightly above
                            obj.Renderer_.updateCurve(ei, spl, 'ZOffset', li * 0.2);
                            
                            % Traditional Mode: Use the beautiful candy-shell layer visual.
                            obj.Renderer_.setCurveVisibility(ei, true);
                            
                            %Share path is top most level. force halo to hide
                            if li > 1
                                obj.Renderer_.setHaloVisibility(ei, false);
                            else
                                obj.Renderer_.setHaloVisibility(ei, true);
                            end
                        end
                    end
                end
            end
        end
    end
    
    % ================================================================== %
    methods (Static)
        function tn = loadNetwork(filepath)
            % LOADNETWORK  Delegates to DesignerSerializer.
            tn = OceanViz.Designer.Persistence.DesignerSerializer.loadNetwork(filepath);
        end
    end
end
