classdef DesignerSerializer
    % DESIGNERSERIALIZER  Stateless serialization utilities for TrackDesigner.
    %
    %   Handles all persistence: JSON export/import, MAT save/load,
    %   struct packing/unpacking, and headless network reconstruction.
    %
    %   All methods are Static — this class holds no state.
    %
    % See also: OceanViz.Designer.TrackDesigner

    methods (Static)

        function s = toStruct(designer)
            % TOSTRUCT  Package graph + UI visual state into a memory struct.
            %
            %   s = DesignerSerializer.toStruct(designer)
            transit = designer.Transit_;
            renderer = designer.Renderer_;
            stationCount = designer.StationCount_;
            
            s = struct();
            s.graphStruct = transit.toStruct(); %The Math logic
            
            
            % Save UI visual state that the abstract math backend intentionally drops
            uiState = struct();
            if ~isempty(renderer)
                % Compact CurveColors and CurveWidths strictly matching the compacted Edges array
                compactColors = {};
                compactWidths = {};
                compactDrawOrders = {};
                compactCurveTypes = {};
                for i = 1:transit.getGraph().getEdgeCount()
                    if ~isempty(transit.getGraph().getEdge(i))
                        % Curve Colors
                        if i <= numel(renderer.CurveColors_) && ~isempty(renderer.CurveColors_{i})
                            compactColors{end+1} = renderer.CurveColors_{i}; %#ok<AGROW>
                        else
                            compactColors{end+1} = [0.4 0.8 1.0]; %#ok<AGROW>
                        end
                        % Curve Widths
                        if i <= numel(renderer.CurveWidths_) && ~isempty(renderer.CurveWidths_{i})
                            compactWidths{end+1} = renderer.CurveWidths_{i}; %#ok<AGROW>
                        else
                            compactWidths{end+1} = renderer.CurveLineWidth; %#ok<AGROW>
                        end
                        % Draw Orders
                        compactDrawOrders{end+1} = renderer.getCurveDrawOrder(i); %#ok<AGROW>
                        
                        % Spline Classes
                        cType = 'catmullrom';
                        spl = transit.getSpline(i);
                        if ~isempty(spl)
                            if contains(class(spl), 'Bezier')
                                cType = 'bezier';
                            end
                        end
                        compactCurveTypes{end+1} = cType; %#ok<AGROW>
                    end
                end
                uiState.CurveColors = compactColors;
                uiState.CurveWidths = compactWidths;
                uiState.CurveDrawOrders = compactDrawOrders;
                uiState.CurveTypes = compactCurveTypes;
                uiState.LabelOffsets = renderer.LabelOffsets_;
                
                % Track visually deleted stations so they don't resurrect
                deletedStations = [];
                for i = 1:stationCount
                    if isempty(renderer.StationHandles_) || i > numel(renderer.StationHandles_) ...
                            || isempty(renderer.StationHandles_{i}) || ~isvalid(renderer.StationHandles_{i})
                        deletedStations(end+1) = i; %#ok<AGROW>
                    end
                end
                uiState.DeletedStations = deletedStations;
                
                
                % Add Overlay Visual states
                if ~isempty(designer.OverlayRenderer_)
                    uiState.LegendPos1 = designer.OverlayRenderer_.LegendPos1_;
                    uiState.LegendPos2 = designer.OverlayRenderer_.LegendPos2_;
                    uiState.TitlePos = designer.OverlayRenderer_.TitlePos_;
                    uiState.MapTitle = designer.OverlayRenderer_.MapTitle_;
                end
            else
                uiState.CurveColors = {};
                uiState.CurveWidths = {};
                uiState.CurveDrawOrders = {};
                uiState.CurveTypes = {};
                uiState.LabelOffsets = {};
                uiState.DeletedStations = [];
            end
            s.uiState = uiState; %The visual Graphics
        end

        function saveMAT(filepath, designer)
            % SAVEMAT  Save the track design to a .mat file.
            %
            %   DesignerSerializer.saveMAT(filepath, designer)
            s = OceanViz.Designer.Persistence.DesignerSerializer.toStruct(designer);
            graphStruct = s.graphStruct; %#ok<NASGU>
            uiState = s.uiState; %#ok<NASGU>
            save(filepath, 'graphStruct', 'uiState');
        end

        function loadMAT(filepath, designer)
            % LOADMAT  Load a track design from a .mat file.
            %
            %   DesignerSerializer.loadMAT(filepath, designer)
            %   designer is the TrackDesigner handle, used for callbacks.
            dataStr = load(filepath);
            OceanViz.Designer.Persistence.DesignerSerializer.loadFromStruct(dataStr, designer);
        end

        function loadFromStruct(dataStr, designer)
            % LOADFROMSTRUCT  Rebuild graph + UI state from a memory struct.
            %
            %   DesignerSerializer.loadFromStruct(dataStr, designer)
            %   designer is the TrackDesigner handle object.
            if ~isfield(dataStr, 'graphStruct')
                return;
            end
            
            % Clear current state via designer's public interface
            designer.resetState_();

            % Rebuild nodes
            if ~isempty(dataStr.graphStruct.Nodes)
                designer.Transit_.reserveStations(numel(dataStr.graphStruct.Nodes));
            end
            for i = 1:numel(dataStr.graphStruct.Nodes)
                n = dataStr.graphStruct.Nodes{i};
                designer.addStation(n.Name, n.Position, 'SuspendGraphics', true); %SuspendGraphics Immediate exist without any unrelated extra steps
                
                % If it was supposed to be dead, brutally delete the handles that addStation just created
                if isfield(dataStr, 'uiState') && isfield(dataStr.uiState, 'DeletedStations') && ...
                        ismember(i, dataStr.uiState.DeletedStations)
                    if ~isempty(designer.Renderer_)
                        designer.Renderer_.deleteStation(i);
                    end
                end
                
                % Restore label offset if available
                if isfield(dataStr, 'uiState') && isfield(dataStr.uiState, 'LabelOffsets') && ...
                        i <= numel(dataStr.uiState.LabelOffsets) && ~isempty(dataStr.uiState.LabelOffsets{i})
                    designer.Renderer_.setLabelOffset(i, dataStr.uiState.LabelOffsets{i});
                end
            end

            % Rebuild edges with splines
            for i = 1:numel(dataStr.graphStruct.Edges)
                e = dataStr.graphStruct.Edges{i};
                splineCtrl = dataStr.graphStruct.SplineData{i};
                if ~isempty(splineCtrl)
                    % Polymorphic load with backward-compatibility default
                    cType = 'catmullrom';
                    if isfield(dataStr, 'uiState') && isfield(dataStr.uiState, 'CurveTypes') && ...
                            i <= numel(dataStr.uiState.CurveTypes) && ~isempty(dataStr.uiState.CurveTypes{i})
                        cType = dataStr.uiState.CurveTypes{i};
                    end
                    
                    if strcmpi(cType, 'bezier')
                        spline = OceanMath.Geometry.BezierCurve(splineCtrl);
                    else
                        spline = OceanMath.Geometry.CatmullRomSpline(splineCtrl);
                    end
                    designer.EdgeCount_ = designer.EdgeCount_ + 1;
                    designer.Transit_.addRoute(e.Source, e.Target, spline, ...
                        'Label', e.Label);
                    if ~isempty(designer.Renderer_)
                        % Restore edge curve color
                        c = [0.4 0.8 1.0];
                        %Get Colors
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
                        %Send to TrackRenderer
                        designer.Renderer_.drawCurve(designer.EdgeCount_, spline, 'Color', c, 'DrawOrder', dOrder);
                        %Make those points clickable after loaded
                        designer.Renderer_.drawControlPoints(designer.EdgeCount_, spline, ...
                            @(eIdx, cpIdx, newPos) designer.onControlPointMoved_(eIdx, cpIdx, newPos));
                    end
                else
                    designer.EdgeCount_ = designer.EdgeCount_ + 1;
                    designer.Transit_.addSimpleRoute(e.Source, e.Target, e.Weight, ...
                        'Label', e.Label);
                end
            end
            
            % Rebuild transit lines
            if isfield(dataStr.graphStruct, 'LineData')
                for i = 1:numel(dataStr.graphStruct.LineData)
                    ld = dataStr.graphStruct.LineData{i};
                    designer.Transit_.addLine(ld.Code, ld.Name, ld.Color);
                    lineObj = designer.Transit_.getLine(ld.Code);
                    
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
                designer.refreshLineDropdown_();
            end
            
            % Restore interchange visuals globally
            for i = 1:designer.StationCount_
                designer.updateStationVisuals_(i);
            end
            designer.updateStatus_();
            if ~isempty(designer.Renderer_)
                designer.Renderer_.reorderCurves();
                
                % Force GUI update limits. Without this, the Zoom Listener 
                % never fires, and Minor Labels stay visible instead of hiding!
                designer.updateAxisLimits_();
                view(designer.Axes_, 2);
                axis(designer.Axes_, 'normal');
                
                % Explicitly trigger LOD visual culling for the first frame
                designer.Renderer_.toggleLabels(designer.Renderer_.LabelsVisible_);
            end
            
            % Hide control points that were permanently activated by drawing math
            if ~isempty(designer.Renderer_)
                designer.Renderer_.destroyAllControlPoints();
            end
            
            %  Rebuild legend cleanly using the restored mathematical lines and visual positions
            if ~isempty(designer.OverlayRenderer_)
                if isfield(dataStr, 'uiState')
                    if isfield(dataStr.uiState, 'LegendPos1') && numel(dataStr.uiState.LegendPos1)==2
                        designer.OverlayRenderer_.LegendPos1_ = dataStr.uiState.LegendPos1(:)';
                    end
                    if isfield(dataStr.uiState, 'LegendPos2') && numel(dataStr.uiState.LegendPos2)==2
                        designer.OverlayRenderer_.LegendPos2_ = dataStr.uiState.LegendPos2(:)';
                    end
                    if isfield(dataStr.uiState, 'TitlePos') && numel(dataStr.uiState.TitlePos)==2
                        designer.OverlayRenderer_.TitlePos_ = dataStr.uiState.TitlePos(:)';
                    end
                    if isfield(dataStr.uiState, 'MapTitle') && ~isempty(dataStr.uiState.MapTitle)
                        designer.OverlayRenderer_.MapTitle_ = dataStr.uiState.MapTitle;
                    end
                end
                designer.OverlayRenderer_.drawLegend_();
            end
        end

        function exportJSON(filepath, transit, renderer, stationCount, edgeCount, overlayState)
            % EXPORTJSON  Write the design to a JSON file.
            %
            %   DesignerSerializer.exportJSON(filepath, transit, renderer, ...
            %       stationCount, edgeCount, overlayState)
            %
            %   overlayState is a struct with fields:
            %     .legendPos1, .legendPos2, .titlePos, .mapTitle

            % Create dynamic ID mapping map cleanly
            idMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
            newIdCounter = 1;
            
            % Extract Global Meta Data
            data = struct();
            data.meta = struct();
            data.meta.defaultLabelOffset = renderer.DefaultLabelOffset(:)';
            
            % Build stations array
            stations = cell(0);
            for i = 1:stationCount
                % Check if visually deleted
                h = renderer.StationHandles_{i};
                if isempty(h) || ~isvalid(h)
                    continue;
                end
                
                st = transit.getStation(i);
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
                if i <= numel(renderer.LabelOffsets_) && ~isempty(renderer.LabelOffsets_{i})
                    overrideOffset = renderer.LabelOffsets_{i}(:)';
                    if ~isequal(overrideOffset, data.meta.defaultLabelOffset)
                        s.labelOffset = overrideOffset;
                    end
                end
                
                stations{end+1} = s; %#ok<AGROW>
                newIdCounter = newIdCounter + 1;
            end

            % Build edges array with spline control points and exact colors
            edges = cell(0);
            for i = 1:edgeCount
                e = transit.getRouteEdge(i);
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
                if i <= numel(renderer.CurveColors_) && ~isempty(renderer.CurveColors_{i})
                    ed.color = renderer.CurveColors_{i};
                else
                    ed.color = renderer.CurveColor; % fallback cyan
                end

                %Extract DNA of the curve
                spl = transit.getSpline(i);
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
                ed.drawOrder = renderer.getCurveDrawOrder(i);
                edges{end+1} = ed; %#ok<AGROW>
            end

            % Reconcile line routes to match actual edge topology before export
            transit.reconcileLineRoutes();

            % Build transit lines
            lines = cell(0);
            codes = transit.getLineNames();
            for i = 1:numel(codes)
                ln = transit.getLine(codes{i});
                
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
            data.legendPos1 = overlayState.legendPos1;
            data.legendPos2 = overlayState.legendPos2;
            data.titlePos = overlayState.titlePos;
            data.mapTitle = overlayState.mapTitle;

            jsonStr = jsonencode(data, 'PrettyPrint', true);
            fid = fopen(filepath, 'w');
            fprintf(fid, '%s', jsonStr);
            fclose(fid);

            msgbox(sprintf('Exported to: %s', filepath), 'JSON Export');
        end

        function importJSON(filepath, designer)
            % IMPORTJSON  Load a design from a JSON file.
            %
            %   DesignerSerializer.importJSON(filepath, designer)

            try
                txt = fileread(filepath);
                data = jsondecode(txt);
            catch ME
                uiwait(errordlg(sprintf('Failed to read JSON file:\n%s', ME.message), 'Import Error'));
                return;
            end
            
            % Normalize jsondecode output: struct arrays -> cell arrays
            toCell = @(x) arrayfun(@(v) v, x, 'UniformOutput', false); %UniformOutput = false, force output a Cell array instead matrix
            if isfield(data, 'stations') && isstruct(data.stations)
                data.stations = toCell(data.stations);
            end
            if isfield(data, 'edges') && isstruct(data.edges)
                data.edges = toCell(data.edges);
            end
            if isfield(data, 'lines') && isstruct(data.lines)
                data.lines = toCell(data.lines);
            end
            
            designer.clearUndo_(); % Prevent rollback to a previous file's map
            
            % Clear current state
            designer.resetState_();
            
            idMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
            
            % Read Global Meta Data
            defaultOffset = [0, 0.3]; % Hard fallback
            if isfield(data, 'meta') && isfield(data.meta, 'defaultLabelOffset')
                defaultOffset = data.meta.defaultLabelOffset(:)';
            end

            % 1. Rebuild Stations
            if isfield(data, 'stations') && ~isempty(data.stations)
                designer.Transit_.reserveStations(numel(data.stations));
                for i = 1:numel(data.stations)
                    if iscell(data.stations)
                        s = data.stations{i};
                    else
                        s = data.stations(i);
                    end
                    designer.addStation(s.name, s.position(:)', 'SuspendGraphics', true);
                    newId = designer.StationCount_;
                    
                    if isfield(s, 'id')
                        idMap(s.id) = newId;
                    else
                        idMap(i) = newId;
                    end
                    
                    if ~isempty(designer.Renderer_)
                        if isfield(s, 'labelOffset')
                            designer.Renderer_.setLabelOffset(newId, s.labelOffset(:)');
                        else
                            designer.Renderer_.setLabelOffset(newId, defaultOffset);
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
                        
                        designer.EdgeCount_ = designer.EdgeCount_ + 1;
                        
                        if isfield(e, 'controlPoints') && ~isempty(e.controlPoints)
                            cp = e.controlPoints;
                            if iscell(cp)
                                cp = cell2mat(reshape(cp, [], 1)); %Auto flaten into variable(Matlab auto calculated) x 1
                            end
                            if isfield(e, 'curveType') && strcmpi(e.curveType, 'bezier')
                                spline = OceanMath.Geometry.BezierCurve(cp);
                            else
                                spline = OceanMath.Geometry.CatmullRomSpline(cp);
                            end
                            designer.Transit_.addRoute(sId, tId, spline);
                            if ~isempty(designer.Renderer_)
                                dOrder = 0;
                                if isfield(e, 'drawOrder'), dOrder = e.drawOrder; end
                                eCol = designer.Renderer_.CurveColor;
                                if isfield(e, 'color') && ~isempty(e.color), eCol = e.color(:)'; end
                                designer.Renderer_.drawCurve(designer.EdgeCount_, spline, 'Color', eCol, 'DrawOrder', dOrder);
                            end
                        else
                            designer.Transit_.addSimpleRoute(sId, tId, e.weight);
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
                    designer.Transit_.addLine(li.code, li.name, li.color(:)');
                    lineObj = designer.Transit_.getLine(li.code);
                    if isfield(li, 'route') && ~isempty(li.route)
                        for j = 1:numel(li.route)
                            if idMap.isKey(li.route(j))
                                lineObj.registerStation(idMap(li.route(j)));
                            end
                        end
                    end
                end
                %Mark system as dirty state 
                designer.Transit_.invalidateLineCache();
                designer.refreshLineDropdown_();
            end
            
            % Restore legend positions if saved (delegated to OverlayRenderer)
            if isfield(data, 'legendPos1') && numel(data.legendPos1) == 2
                designer.OverlayRenderer_.LegendPos1_ = data.legendPos1(:)';
            end
            if isfield(data, 'legendPos2') && numel(data.legendPos2) == 2
                designer.OverlayRenderer_.LegendPos2_ = data.legendPos2(:)';
            end
            if isfield(data, 'titlePos') && numel(data.titlePos) == 2
                designer.OverlayRenderer_.TitlePos_ = data.titlePos(:)';
            end
            if isfield(data, 'mapTitle') && ~isempty(data.mapTitle)
                designer.OverlayRenderer_.MapTitle_ = data.mapTitle;
            end

            % Restore interchange visuals globally and enforce axis limits
            for i = 1:designer.StationCount_
                designer.updateStationVisuals_(i);
            end
            designer.updateStatus_();
            if ~isempty(designer.Renderer_)
                designer.Renderer_.reorderCurves();
                designer.updateAxisLimits_();
                % Force 2D view and stretch-to-fill after all plot3 calls
                view(designer.Axes_, 2);
                axis(designer.Axes_, 'normal');
            end
            
            % Draw transit legend at bottom
            if ~isempty(designer.OverlayRenderer_)
                designer.OverlayRenderer_.drawLegend_();
            end
            
            % Re-register axes interaction callbacks (view/axis can silently clear them)
            set(designer.Axes_, 'ButtonDownFcn', @(~,~) designer.InteractionHandler_.onAxesClick_());
        end

        function [tn, vizMeta] = loadNetwork(filepath)
            % LOADNETWORK  Reconstruct a TransitNetwork from a TrackDesigner JSON map.
            %   Also extracts per-edge visual metadata (colors, label offsets)
            %   as a second output for rendering consumers.
            %
            %   [tn, vizMeta] = DesignerSerializer.loadNetwork(filepath)
            %
            %   vizMeta fields:
            %     .EdgeColors    — {E x 1} cell of [1x3] RGB or []
            %     .LabelOffsets  — {V x 1} cell of [1x2] offset or []
            %     .DefaultLabelOffset — [1x2] global default
            txt = fileread(filepath);
            data = jsondecode(txt);
            
            tn = OceanMath.Transit.TransitNetwork();
            
            idMap = containers.Map('KeyType', 'double', 'ValueType', 'double');
            
            % Visual metadata accumulators
            labelOffsets = {};
            edgeColors   = {};
            defaultLabelOffset = [0, 0.3];
            if isfield(data, 'meta') && isfield(data.meta, 'defaultLabelOffset')
                defaultLabelOffset = data.meta.defaultLabelOffset(:)';
            end
            
            % 1. Rebuild Stations
            if isfield(data, 'stations') && ~isempty(data.stations)
                isCellArray = iscell(data.stations);
                for i = 1:numel(data.stations)
                    if isCellArray
                        s = data.stations{i};
                    else
                        s = data.stations(i);
                    end
                    newId = tn.addStation(s.name, s.position(:)');
                    if isfield(s, 'id')
                        idMap(s.id) = newId;
                    else
                        idMap(i) = newId;
                    end
                    
                    % Capture label offset
                    if isfield(s, 'labelOffset')
                        labelOffsets{newId} = s.labelOffset(:)'; %#ok<AGROW>
                    else
                        labelOffsets{newId} = defaultLabelOffset; %#ok<AGROW>
                    end
                end
            end
            
            % 2. Rebuild Edges (with physics splines)
            if isfield(data, 'edges') && ~isempty(data.edges)
                isCellArray = iscell(data.edges);
                for i = 1:numel(data.edges)
                    if isCellArray
                        e = data.edges{i};
                    else
                        e = data.edges(i);
                    end
                    if ~idMap.isKey(e.source) || ~idMap.isKey(e.target)
                        continue;
                    end
                    sId = idMap(e.source);
                    tId = idMap(e.target);
                    
                    edgeIdx = 0;
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
                        edgeIdx = tn.addRoute(sId, tId, spline);
                    else
                        edgeIdx = tn.addSimpleRoute(sId, tId, e.weight);
                    end
                    
                    % Capture per-edge color
                    if edgeIdx > 0 && isfield(e, 'color') && ~isempty(e.color)
                        edgeColors{edgeIdx} = e.color(:)'; %#ok<AGROW>
                    end
                end
            end
            
            % 3. Rebuild Lines
            if isfield(data, 'lines') && ~isempty(data.lines)
                isCellArray = iscell(data.lines);
                for i = 1:numel(data.lines)
                    if isCellArray
                        li = data.lines{i};
                    else
                        li = data.lines(i);
                    end
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
            
            % Package visual metadata
            vizMeta.EdgeColors         = edgeColors;
            vizMeta.LabelOffsets       = labelOffsets;
            vizMeta.DefaultLabelOffset = defaultLabelOffset;
        end

    end
    
    methods (Static)
        function [tn, uiState] = loadNetworkFromMAT(filepath)
            % LOADNETWORKFROMMAT  Reconstruct a strictly mathematical TransitNetwork from a .mat file.
            %   Ignores GUI overhead (no TrackDesigner) and reconstruction.
            %   Returns both the TransitNetwork and the saved UI visual attributes block.
            %
            %   [tn, uiState] = OceanViz.Designer.Persistence.DesignerSerializer.loadNetworkFromMAT(filepath)
            dataStr = load(filepath);
            tn = OceanMath.Transit.TransitNetwork();
            
            if isfield(dataStr, 'graphStruct')
                g = dataStr.graphStruct;
                % Rebuild Nodes
                if isfield(g, 'Nodes') && ~isempty(g.Nodes)
                    tn.reserveStations(numel(g.Nodes));
                    for i = 1:numel(g.Nodes)
                        n = g.Nodes{i};
                        if isempty(n)
                            tn.addStation(sprintf('Deleted_%d', i), [0 0]);
                        else
                            tn.addStation(n.Name, n.Position);
                        end
                    end
                end
                
                % Rebuild Edges
                if isfield(g, 'Edges') && ~isempty(g.Edges)
                    for i = 1:numel(g.Edges)
                        e = g.Edges{i};
                        if ~isempty(e)
                            splineCtrl = g.SplineData{i};
                            if ~isempty(splineCtrl)
                                cType = 'catmullrom';
                                if isfield(dataStr, 'uiState') && isfield(dataStr.uiState, 'CurveTypes') && ...
                                   i <= numel(dataStr.uiState.CurveTypes) && ~isempty(dataStr.uiState.CurveTypes{i})
                                    cType = dataStr.uiState.CurveTypes{i};
                                end
                                if strcmpi(cType, 'bezier')
                                    spline = OceanMath.Geometry.BezierCurve(splineCtrl);
                                else
                                    spline = OceanMath.Geometry.CatmullRomSpline(splineCtrl);
                                end
                                tn.addRoute(e.Source, e.Target, spline);
                            else
                                tn.addSimpleRoute(e.Source, e.Target, e.Weight);
                            end
                        end
                    end
                end
                
                % Rebuild Lines
                if isfield(g, 'Lines') && ~isempty(g.Lines)
                    for i = 1:numel(g.Lines)
                        l = g.Lines{i};
                        if ~isempty(l)
                            tn.addLine(l.Code, l.Name, l.Color);
                            lineObj = tn.getLine(l.Code);
                            if isfield(l, 'Route')
                                for j = 1:numel(l.Route)
                                    lineObj.registerStation(l.Route(j));
                                end
                            end
                            if isfield(l, 'Policy') && ~isempty(l.Policy)
                                lineObj.DwellTimeParams = l.Policy.DwellTimeParams;
                                lineObj.MaxSpeedParams = l.Policy.MaxSpeedParams;
                            end
                        end
                    end
                end
            end
            
            uiState = struct();
            if isfield(dataStr, 'uiState')
                uiState = dataStr.uiState;
            end
        end
    end
end
