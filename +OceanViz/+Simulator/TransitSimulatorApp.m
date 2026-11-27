classdef TransitSimulatorApp < handle
    % TRANSITSIMULATORAPP  Standalone MVC Viewer for the Transit Simulation Engine.
    % 
    % Usage:
    %   app = OceanViz.Simulator.TransitSimulatorApp();
    %
    
    properties (Access = private)
        % --- Core Architecture ---
        Layout_          % The pure UI view builder (OceanViz.Simulator.TransitSimulatorLayout)
        Network_         % Math graph engine (Nodes, Edges)
        Simulation_      % Math TransitSimulation engine
        MapData_         % Raw struct loaded from DesignerSerializer for legacy compatibility
        
        % --- Simulation State ---
        Timer_           % the 60hz execution loop
        IsPlaying_ = false
        TimeWarp_ = 1.0
        RealClock_  (1,1) double = 0;   % Tracks real-life internal stopwatch time
        UnitsPerKm_ (1,1) double = 1.0; % Scaling factor from TrackDesigner or User
        
        % --- Visual Assets ---
        AgentMarkers_ = struct('Handles', [], 'IDs', []); 
        GraphStrategy_  % OceanViz.Strategies.TransitGraphStrategy
        VizCore_        % Theme cache (OceanViz.Core)
        VizMeta_        % visual metadata from JSON (EdgeColors, LabelOffsets)
        
        % --- Interaction State ---
        CurrentPathNodes_ = [];
        TableLastUpdate_ = 0;
    end
    
    methods
        function obj = TransitSimulatorApp()
            % TRANSITSIMULATORAPP Constructor. Instantiates the layout and callbacks.
            obj.launch();
        end
        
        function launch(obj)
            % LAUNCH Initializes the backend Math layer, configures UI callbacks,
            arguments
                obj
            end
            % and hands them off securely to the Layout View for rendering.
            % 1. Core Services Setup
            obj.VizCore_ = OceanViz.Core('Theme', 'Dark');
            
            % 2. Graph Drawing Strategy
            obj.GraphStrategy_ = OceanViz.Strategies.TransitGraphStrategy();
            
            % 3. Write Callbacks Mapping to Layout (String mapping)
            callbacks.onClose = @(~,~) obj.onClose_();
            callbacks.onLoad = @(~,~) obj.onLoadClicked_();
            callbacks.onPlay = @(~,~) obj.onPlayClicked_();
            callbacks.onReset = @(~,~) obj.onResetClicked_();
            callbacks.onTimeWarpChanged = @(~,~) obj.onTimeWarpChanged_();
            callbacks.onLabelsChanged = @(~,~) obj.drawMap_();
            callbacks.onFindPath = @(~,~) obj.onFindPathClicked_();
            callbacks.onClearPath = @(~,~) obj.onClearPathClicked_();
            callbacks.onSpawnLineSelected = @(~,~) obj.onSpawnLineSelected_();
            callbacks.onSpawnPhysicsChanged = @(~,~) obj.onSpawnPhysicsChanged_();
            callbacks.onSetScale = @(~,~) obj.onSetScaleClicked_();
            callbacks.onSpawnTrain = @(~,~) obj.onSpawnTrainClicked_();
            
            % 4. Build Layout View
            obj.Layout_ = OceanViz.Simulator.TransitSimulatorLayout(obj.VizCore_, callbacks);
            
            % 5. Timer setup (30 FPS)
            obj.Timer_ = timer('ExecutionMode', 'fixedRate', ...
                'Period', round(1.0 / 30.0, 3), ...
                'TimerFcn', @(~,~) obj.onTimerTick_());
        end
        

        
        function onLoadClicked_(obj)
            % ONLOADCLICKED_ Triggered when 'Load Map' is pressed.
            [file, path] = uigetfile({'*.mat;*.json', 'All Track Maps (*.mat, *.json)'; '*.mat', 'MATLAB Map (*.mat)'; '*.json', 'JSON Map (*.json)'}, 'Select Track Designer Map');
            if isequal(file,0), return; end
            filepath = fullfile(path, file);
            obj.loadMap(filepath);
        end

        function loadMap(obj, filepath)
            % LOADMAP Load a track map from a .mat or .json file Programmatically
            arguments
                obj
                filepath (1,:) char
            end
            % Parses track designer files (.json or legacy .mat), initializes the
            % math network, sets up the camera bounds, and populates UI dropdowns.
            
            % Stop simulation if running
            if obj.IsPlaying_
                obj.onPlayClicked_();
            end
            
            % Clear old graphics
            cla(obj.Layout_.Axes_);
            
            % 1. Load headless math and ui blocks
            [~, ~, ext] = fileparts(filepath);
            if strcmpi(ext, '.json')
                [obj.Network_, obj.VizMeta_] = OceanViz.Designer.Persistence.DesignerSerializer.loadNetwork(filepath);
                obj.MapData_ = struct();
            else
                [obj.Network_, obj.MapData_] = OceanViz.Designer.Persistence.DesignerSerializer.loadNetworkFromMAT(filepath);
                obj.VizMeta_ = struct('EdgeColors', {{}}, 'LabelOffsets', {{}}, 'DefaultLabelOffset', [0, 0.3]);
            end
            
            % 2. Initialize pure math simulation engine
            obj.Simulation_ = OceanMath.Transit.TransitSimulation(obj.Network_);
            
            % Populate Spawner UI instead of auto-spawning
            lineCodes = obj.Network_.getLineNames();
            if ~isempty(lineCodes)
                set(obj.Layout_.Controls_.CmbLine, 'String', lineCodes, 'Value', 1);
                obj.onSpawnLineSelected_();
            end
            
            % 3. Draw static UI layout (Map geometry)
            obj.drawMap_();
            
            % Hack: set camera limits based on Graph bounds
            pos = obj.Network_.getStationPositions();
            if ~isempty(pos)
                pad = max(max(pos) - min(pos)) * 0.1;
                xlim(obj.Layout_.Axes_, [min(pos(:,1))-pad, max(pos(:,1))+pad]);
                ylim(obj.Layout_.Axes_, [min(pos(:,2))-pad, max(pos(:,2))+pad]);
            end
            
            set(obj.Layout_.Controls_.LblClock, 'String', 'T: 0.00s');
            
             % Setup Path Finder Dropdowns
             data = obj.Network_.toVizData();
             if ~isempty(data.NodeNames)
                 set(obj.Layout_.Controls_.CmbSrc, 'String', data.NodeNames);
                 set(obj.Layout_.Controls_.CmbDst, 'String', data.NodeNames);
             end
             obj.CurrentPathNodes_ = [];
        end
        
        function onFindPathClicked_(obj)
            % ONFINDPATHCLICKED_ Queries the Math Network for the shortest path
            % (Dijkstra) between two selected stations and instructs the View to highlight it.
            if isempty(obj.Network_), return; end
            sId = get(obj.Layout_.Controls_.CmbSrc, 'Value');
            tId = get(obj.Layout_.Controls_.CmbDst, 'Value');
            
            if sId == tId
                disp('Source and target are the same.');
                return;
            end
            
            [path, dist] = obj.Network_.shortestPath(sId, tId);
            if isempty(path)
                disp('No path found!');
            else
                fprintf('Path found! Distance = %.2f units\n', dist);
                obj.CurrentPathNodes_ = path;
                obj.drawMap_();
            end
        end
        
        function onClearPathClicked_(obj)
            % ONCLEARPATHCLICKED_ Wipes the stored path highlight sequence and redraws map.
            if ~isempty(obj.CurrentPathNodes_)
                obj.CurrentPathNodes_ = [];
                obj.drawMap_();
            end
        end
        
        function onSpawnLineSelected_(obj)
            % ONSPAWNLINESELECTED_ Cascades populates the Station dropdown based
            % on the currently selected Line in the Spawner panel.
            if isempty(obj.Network_), return; end
            idx = get(obj.Layout_.Controls_.CmbLine, 'Value');
            lineCodes = get(obj.Layout_.Controls_.CmbLine, 'String');
            code = lineCodes{idx};
            line = obj.Network_.getLine(code);
            route = line.getRoute();
            
            % Get station names for the route
            data = obj.Network_.toVizData();
            names = data.NodeNames(route);
            set(obj.Layout_.Controls_.CmbStation, 'String', names, 'Value', 1);
            obj.onSpawnPhysicsChanged_();
            
            % Aesthetic: Tint policy label to the line color!
            c = line.Color; 
            if sum(c) < 0.5, c = c + 0.3; end % brighten dark colors for UI
            set(obj.Layout_.Controls_.LblPolicy, 'ForegroundColor', c);
        end
        function onSetScaleClicked_(obj)
            % ONSETSCALECLICKED_ Prompt the user for real-world vs math-grid conversion.
            prompt = {'Enter real-world distance inside 100 Grid Units (e.g., if 100 units = 2.5 km, enter 2.5):'};
            dlgtitle = 'Map Math to Distance Scale';
            dims = [1 70];
            
            % Present the current mathematically derived scale as the default
            currentRealDistancePer100 = 100 / obj.UnitsPerKm_;
            definput = {sprintf('%.2f', currentRealDistancePer100)};
            
            answer = inputdlg(prompt, dlgtitle, dims, definput);
            if isempty(answer), return; end
            
            kmPer100 = str2double(answer{1});
            if ~isnan(kmPer100) && kmPer100 > 0
                obj.UnitsPerKm_ = 100 / kmPer100;
                obj.onSpawnPhysicsChanged_(); % Refresh UI labels
            end
        end

        function onSpawnPhysicsChanged_(obj)
            % ONSPAWNPHYSICSCHANGED_ Reads raw math UI sliders and converts them
            % into real-world km/h and km/h/s, formatting them for the Agent Settings readout.
            spd = get(obj.Layout_.Controls_.SldSpeed, 'Value');
            acc = get(obj.Layout_.Controls_.SldAccel, 'Value');
            
            % Calculate real-world km/h mathematically
            kmPerSec = spd / obj.UnitsPerKm_;
            kmh = kmPerSec * 3600;
            
            % Acceleration
            kmhs = (acc / obj.UnitsPerKm_) * 3600;
            
            % Update dedicated speed/accel labels
            set(obj.Layout_.Controls_.LblSpeedValue, 'String', sprintf('%.2f u/s (%.0f km/h)', spd, kmh));
            
            if acc == 0
                set(obj.Layout_.Controls_.LblAccelValue, 'String', 'Instantaneous');
            else
                set(obj.Layout_.Controls_.LblAccelValue, 'String', sprintf('%.2f u/s² (%.0f km/h/s)', acc, kmhs));
            end
            
            % Maintain Manager Policy Text at the bottom right
            policyTxt = 'MANAGER POLICY\nAll system normal.';
            
            % Speed policy visualization limit (does not cap mathematical physics)
            if kmh > 200
                policyTxt = [policyTxt '\n\n[WARNING: SPEED VIOLATION]'];
            end
            
            set(obj.Layout_.Controls_.LblPolicy, 'String', sprintf(policyTxt));
        end
        
        function onSpawnTrainClicked_(obj)
            % ONSPAWNTRAINCLICKED_ Harvests all Agent Spawner UI variables and
            % sequentially injects N new agents into the physics engine.
            if isempty(obj.Simulation_), return; end
            
            lineIdx = get(obj.Layout_.Controls_.CmbLine, 'Value');
            lines = get(obj.Layout_.Controls_.CmbLine, 'String');
            lineCode = lines{lineIdx};
            
            segIdx = get(obj.Layout_.Controls_.CmbStation, 'Value');
            spd = get(obj.Layout_.Controls_.SldSpeed, 'Value');
            acc = get(obj.Layout_.Controls_.SldAccel, 'Value');
            brk = acc * 1.5; % Brakes are 150% stronger than acceleration
            
            countIdx = get(obj.Layout_.Controls_.CmbCount, 'Value');
            counts = get(obj.Layout_.Controls_.CmbCount, 'String');
            numToSpawn = str2double(counts{countIdx});
            
            for i = 1:numToSpawn
                try
                    obj.Simulation_.addAgent(lineCode, 'Speed', spd, ...
                        'AccelRate', acc, 'BrakeRate', brk, 'StartSegment', segIdx);
                catch ex
                    disp(ex.message);
                end
            end
            
            % Force update UI immediately so we see the injection
            obj.updateTableGraphics_();
            obj.updateAgentGraphics_();
            drawnow;
        end
        
        function drawMap_(obj)
            % Ask strategy to draw the graph
            data = obj.Network_.toVizData();
            
            style = OceanViz.Adapters.GraphStyle(obj.Network_.getStationCount(), obj.Network_.getRouteCount());
            
            % 1. Apply per-edge colors from JSON visual metadata (exact match to TrackDesigner)
            if ~isempty(obj.VizMeta_) && isfield(obj.VizMeta_, 'EdgeColors')
                for ei = 1:numel(obj.VizMeta_.EdgeColors)
                    if ~isempty(obj.VizMeta_.EdgeColors{ei})
                        style.setEdgeColor(ei, obj.VizMeta_.EdgeColors{ei});
                        style.setEdgeWidth(ei, 2.5);
                    end
                end
            end
            
            % Pass label offsets into vizData so the strategy can use them
            if ~isempty(obj.VizMeta_) && isfield(obj.VizMeta_, 'LabelOffsets')
                data.LabelOffsets = obj.VizMeta_.LabelOffsets;
            end
            
            % 2. Map Toggle Labels rules
            labelMode = get(obj.Layout_.Controls_.CmbLabels, 'Value');
            isInterch = false(1, obj.Network_.getStationCount());
            isInterch(data.InterchangeIds) = true;
            for ni = 1:obj.Network_.getStationCount()
                if labelMode == 3
                    style.setNodeLabelVisible(ni, false);
                elseif labelMode == 2 && ~isInterch(ni)
                    style.setNodeLabelVisible(ni, false);
                end
            end
            
            % 3. If there's a Shortest Path highlighted, dynamically inject it into the style!
            if ~isempty(obj.CurrentPathNodes_) && numel(obj.CurrentPathNodes_) > 1
                % Cyan glow across the path sequence
                style.highlightPath(obj.CurrentPathNodes_, [0, 1, 1], 3.0);
            end
            
            data.Style = style;
                
            % Flush graphics caches natively to prevent train handle ghosting errors
            cla(obj.Layout_.Axes_); 
            obj.AgentMarkers_.Handles = [];
            
            % Render via fallback styling inheritance
            obj.GraphStrategy_.render(obj.Layout_.Axes_, data, obj.VizCore_.Colors);
        end
        
        function onPlayClicked_(obj)
            if isempty(obj.Simulation_)
                disp('Please load a map first!');
                return;
            end
            
            % ALWAYS rely directly on the Timer's raw hardware state, never a software boolean race condition.
            if strcmp(obj.Timer_.Running, 'on')
                stop(obj.Timer_);
                if isvalid(obj.Layout_.Controls_.BtnPlay)
                    set(obj.Layout_.Controls_.BtnPlay, 'String', 'PLAY');
                end
            else
                % If it was in the middle of closing/stopping, force it safely.
                if strcmp(obj.Timer_.Running, 'off')
                    start(obj.Timer_);
                    if isvalid(obj.Layout_.Controls_.BtnPlay)
                        set(obj.Layout_.Controls_.BtnPlay, 'String', 'PAUSE');
                    end
                end
            end
        end
        
        function onResetClicked_(obj)
            if strcmp(obj.Timer_.Running, 'on')
                obj.onPlayClicked_();
            end
            if ~isempty(obj.Simulation_)
                obj.Simulation_.reset();
                obj.RealClock_ = 0; % Reset our stopwatch
                obj.updateAgentGraphics_();
                set(obj.Layout_.Controls_.LblClock, 'String', 'Sim Time: 00:00:00.0 | Real Time: 00:00:00.0');
            end
        end
        
        function onTimeWarpChanged_(obj)
            obj.TimeWarp_ = get(obj.Layout_.Controls_.SldTime, 'Value');
            set(obj.Layout_.Controls_.LblTime, 'String', sprintf('%.1fx', obj.TimeWarp_));
            set(obj.Layout_.Controls_.LblWarpNote, 'String', sprintf('(1.0 real s = %.1f sim s)', obj.TimeWarp_));
        end
        
        function onTimerTick_(obj)
            if ~isvalid(obj.Layout_.Figure_)
                if ~isempty(obj.Timer_) && isvalid(obj.Timer_) && strcmp(obj.Timer_.Running, 'on')
                    stop(obj.Timer_);
                end
                return;
            end
            
            if isempty(obj.Simulation_), return; end
            
            % Calculate the total chunk of time this visual frame represents
            baseTickTime = 1.0 / 30.0;
            totalTimeToSimulate = baseTickTime * obj.TimeWarp_;
            
            % Update Real Life Clock Tracker (stopwatch)
            obj.RealClock_ = obj.RealClock_ + baseTickTime;
            
            % Sub-stepping Engine to preserve physics integrity and prevent tunneling!
            % We cap the maximum allowable physics step to baseTickTime.
            nSteps = ceil(totalTimeToSimulate / baseTickTime);
            safeDt = totalTimeToSimulate / nSteps; 
            realDt = baseTickTime / nSteps;
            
            % Run the math loop N times before updating the screen
            for k = 1:nSteps
                obj.Simulation_.step(safeDt, realDt);
            end
            
            % Update visual frontend once
            obj.updateAgentGraphics_();
            
            % Format Times
            hS = floor(obj.Simulation_.Clock_ / 3600);
            mS = floor(mod(obj.Simulation_.Clock_, 3600) / 60);
            sS = mod(obj.Simulation_.Clock_, 60);
            
            hR = floor(obj.RealClock_ / 3600);
            mR = floor(mod(obj.RealClock_, 3600) / 60);
            sR = mod(obj.RealClock_, 60);
            
            txt = sprintf('Sim Time: %02d:%02d:%04.1f | Real Time: %02d:%02d:%04.1f', hS, mS, sS, hR, mR, sR);
            set(obj.Layout_.Controls_.LblClock, 'String', txt);
            
            % Table Throttler: Update 10 times per REAL-WORLD second for fluid telemetry
            if (obj.RealClock_ - obj.TableLastUpdate_) >= 0.01 
                obj.updateTableGraphics_();
                obj.TableLastUpdate_ = obj.RealClock_;
            end
            
            % Flush visual pipeline normally. DO NOT use 'nocallbacks' here because it blocks 
            % the UI Thread from processing your Button Clicks and Close Window requests!
            drawnow;
        end
        
        function updateTableGraphics_(obj)
            agents = obj.Simulation_.getAgents();
            n = numel(agents);
            if n == 0, return; end
            
            tableData = cell(n, 6);
            for i = 1:n
                a = agents{i};
                tableData{i, 1} = num2str(i);
                tableData{i, 2} = char(a.getLineCode());
                st = char(a.getAgentState());
                if strcmp(st, 'dwelling')
                    st = sprintf('dwelling (%.1fs)', a.getDwellRemaining());
                end
                tableData{i, 3} = st;
                
                s = a.getState();
                if ~isempty(s)
                    % Convert velocity to km/h
                    km_p_s = s.Velocity / obj.UnitsPerKm_;
                    km_p_h = km_p_s * 3600;
                    
                    % Convert dist to km
                    dist_km = s.ArcDistance / obj.UnitsPerKm_;
                    
                    tableData{i, 4} = sprintf('%.0f', km_p_h);
                    tableData{i, 5} = sprintf('%.1f', dist_km);
                    tableData{i, 6} = sprintf('%.0f%%', a.getProgress() * 100);
                end
            end
            set(obj.Layout_.Controls_.TeleTable, 'Data', tableData);
        end
        
        function updateAgentGraphics_(obj)
            agents = obj.Simulation_.getAgents();
            
            % Verify handles haven't been purged by cla()
            if ~isempty(obj.AgentMarkers_.Handles) && any(~ishandle(obj.AgentMarkers_.Handles))
                obj.AgentMarkers_.Handles = [];
            end
            
            if numel(obj.AgentMarkers_.Handles) < numel(agents)
                for i = (numel(obj.AgentMarkers_.Handles)+1):numel(agents)
                    h = line(obj.Layout_.Axes_, NaN, NaN, NaN, 'Marker', 'o', ...
                        'MarkerFaceColor', [1 0.2 0.2], 'MarkerEdgeColor', 'w', ...
                        'MarkerSize', 10, 'LineWidth', 1.5, 'LineStyle','none');
                    obj.AgentMarkers_.Handles(i) = h;
                end
            end
            
            nAgents = numel(agents);
            for i = 1:numel(obj.AgentMarkers_.Handles)
                h = obj.AgentMarkers_.Handles(i);
                if i <= nAgents
                    a = agents{i};
                    p = a.getPosition();
                    z = 1.0; 
                    if numel(p) >= 3, z = p(3) + 1.0; end
                    
                    c = a.getColor();
                    if isempty(c), c = [1 0.2 0.2]; end
                    
                    set(h, 'XData', p(1), 'YData', p(2), 'ZData', z, 'MarkerFaceColor', c);
                else
                    set(h, 'XData', NaN, 'YData', NaN, 'ZData', NaN);
                end
            end
        end
        
        function onClose_(obj)
            if ~isempty(obj.Timer_) && isvalid(obj.Timer_)
                stop(obj.Timer_);
                delete(obj.Timer_);
            end
            delete(obj.Layout_.Figure_);
        end
    end
end
