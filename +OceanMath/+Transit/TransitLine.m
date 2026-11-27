classdef TransitLine < handle
    % TRANSITLINE  Manager for one transit line (e.g. a single MRT line).
    %
    % Acts as a "highway authority": owns the line's identity (code, name,
    % color), maintains an ordered station sequence (the route), and holds
    % extensible operational policies (speed, tickets, frequency).
    %
    % Usage:
    %   line = OceanMath.Transit.TransitLine('NSL', 'North-South Line', [0.85 0.15 0.15]);
    %   line.registerStations([1, 2, 3, 4]);    % bulk register
    %   line.setPolicy('SpeedLimit', 80);
    %   route = line.getRoute();                 % → [1, 2, 3, 4]
    %   line.getPolicy('SpeedLimit')             % → 80
    %
    % See also: OceanMath.Transit.TransitNetwork

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Code        (1,:) char      % Short identifier: 'NSL', 'EWL'
        Name        (1,:) char      % Full name: 'North-South Line'
        Color       (1,3) double    % RGB visual identity
        StationIds_ (:,1) double    % Ordered station IDs (the route)
        Policy_     (1,1) struct    % Extensible operational attributes
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = TransitLine(code, name, color)
            % TRANSITLINE  Create a new line manager.
            %
            %   line = TransitLine('NSL', 'North-South Line', [0.85 0.15 0.15])
            arguments
                code   (1,:) char
                name   (1,:) char
                color  (1,3) double
            end
            obj.Code  = code;
            obj.Name  = name;
            obj.Color = color;
            obj.StationIds_ = [];
            obj.Policy_ = struct();
        end

        % ============================================================== %
        %  Route Management
        % ============================================================== %

        function registerStation(obj, stationId)
            % REGISTERSTATION  Append a station to end of the route.
            %   Skips if station is already registered.
            arguments
                obj
                stationId (1,1) double {mustBePositive, mustBeInteger}
            end
            if ~obj.hasStation(stationId)
                obj.StationIds_(end+1, 1) = stationId;
            end
        end

        function registerStations(obj, stationIds)
            % REGISTERSTATIONS  Bulk-register an ordered list of stations.
            %
            %   line.registerStations([1, 2, 3, 4])
            %
            %   Appends each station in order. Skips duplicates.
            arguments
                obj
                stationIds (1,:) double {mustBePositive, mustBeInteger}
            end
            for i = 1:numel(stationIds)
                obj.registerStation(stationIds(i));
            end
        end

        function insertStation(obj, position, stationId)
            % INSERTSTATION  Insert a station at a specific position.
            %
            %   line.insertStation(2, 5)  — insert station 5 at position 2
            arguments
                obj
                position  (1,1) double {mustBePositive, mustBeInteger}
                stationId (1,1) double {mustBePositive, mustBeInteger}
            end
            if obj.hasStation(stationId), return; end
            if position > numel(obj.StationIds_) + 1 % Insert at the end if position is greater than the number of stations
                position = numel(obj.StationIds_) + 1;
            end
            obj.StationIds_ = [obj.StationIds_(1:position-1); ...
                               stationId; ...
                               obj.StationIds_(position:end)];
        end

        function removeStation(obj, stationId)
            % REMOVESTATION  Remove a station from the route.
            arguments
                obj
                stationId (1,1) double {mustBePositive, mustBeInteger}
            end
            obj.StationIds_(obj.StationIds_ == stationId) = [];
        end

        function setRoute(obj, stationIds)
            % SETROUTE  Replace the route with an explicitly ordered sequence.
            %   Used by reconciliation to fix click-order mismatches.
            %
            %   line.setRoute([1, 5, 3, 8])
            arguments
                obj
                stationIds (1,:) double {mustBePositive, mustBeInteger}
            end
            obj.StationIds_ = stationIds(:);  % store as column
        end

        function route = getRoute(obj)
            % GETROUTE  Return the ordered station IDs.
            arguments
                obj
            end
            route = obj.StationIds_';  % return as row vector
        end

        function N = getStationCount(obj)
            % GETSTATIONCOUNT  Number of stations on this line.
            arguments
                obj
            end
            N = numel(obj.StationIds_);
        end

        function tf = hasStation(obj, stationId)
            % HASSTATION  Check if a station belongs to this line.
            arguments
                obj
                stationId (1,1) double
            end
            tf = any(obj.StationIds_ == stationId);
        end

        function L = getLength(obj, graph)
            % GETLENGTH  Total line length (sum of edge weights along route).
            %
            %   L = line.getLength(tn.getGraph())
            %
            %   Sums the weights between consecutive stations in the route.
            arguments
                obj
                graph   % DigraphCus — provides getAdjacencyMatrix
            end
            L = 0;
            W = graph.getAdjacencyMatrix();
            route = obj.StationIds_;
            for i = 1:numel(route)-1
                s = route(i);
                t = route(i+1);
                if s <= size(W,1) && t <= size(W,2) && W(s,t) > 0
                    L = L + W(s, t);
                end
            end
        end

        % ============================================================== %
        %  Policy Management
        % ============================================================== %

        function setPolicy(obj, key, value)
            % SETPOLICY  Set an operational attribute.
            %
            %   line.setPolicy('SpeedLimit', 80)
            %   line.setPolicy('TicketZone', 2)
            %   line.setPolicy('Frequency', 3)
            arguments
                obj
                key   (1,:) char
                value
            end
            obj.Policy_.(key) = value;
        end

        function value = getPolicy(obj, key)
            % GETPOLICY  Get an operational attribute.
            %
            %   speed = line.getPolicy('SpeedLimit')
            arguments
                obj
                key (1,:) char
            end
            if isfield(obj.Policy_, key)
                value = obj.Policy_.(key);
            else
                value = [];
            end
        end

        function p = getAllPolicies(obj)
            % GETALLPOLICIES  Return the full policy struct.
            p = obj.Policy_;
        end

        function tf = hasPolicy(obj, key)
            % HASPOLICY  Check if a policy key exists.
            tf = isfield(obj.Policy_, key);
        end

        % ============================================================== %
        %  Display
        % ============================================================== %

        function describe(obj)
            fprintf('  [%s] %s (%d stations)\n', obj.Code, obj.Name, ...
                obj.getStationCount());
            if obj.getStationCount() > 0
                ids = arrayfun(@num2str, obj.StationIds_, 'UniformOutput', false);
                fprintf('    Route: %s\n', strjoin(ids, ' -> '));
            end
            fields = fieldnames(obj.Policy_);
            if ~isempty(fields)
                for i = 1:numel(fields)
                    val = obj.Policy_.(fields{i});
                    if isnumeric(val)
                        fprintf('    %s: %g\n', fields{i}, val);
                    else
                        fprintf('    %s: %s\n', fields{i}, string(val));
                    end
                end
            end
        end

        function s = toStruct(obj)
            % TOSTRUCT  Export for serialization.
            s.Code       = obj.Code;
            s.Name       = obj.Name;
            s.Color      = obj.Color;
            s.StationIds = obj.StationIds_';
            s.Policy     = obj.Policy_;
        end
    end
end
