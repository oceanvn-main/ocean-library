classdef TransitAgent < handle
    % TRANSITAGENT  A train that traverses a multi-segment transit route.
    %
    % Uses the State Pattern to manage behavior across Running, Dwelling,
    % and Arrived states. Each state is an IAgentState object that handles
    % step(), getPosition(), getTangent(), getProgress(), and isArrived().
    %
    % Bridges CurveSimulator (single-spline physics) with TransitNetwork
    % (multi-segment graph). Chains simulators across consecutive splines,
    % with configurable dwell time at each intermediate station.
    %
    % Usage:
    %   agent = OceanMath.Transit.TransitAgent(tn, 'NSL');
    %   while ~agent.isArrived()
    %       agent.step();
    %       pos = agent.getPosition();
    %   end
    %
    % See also: OceanMath.Simulation.CurveSimulator,
    %           OceanMath.Transit.TransitNetwork,
    %           OceanMath.Transit.AgentStates.IAgentState

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Network_        % TransitNetwork reference
        LineCode_       (1,:) char          % which transit line
        Route_          (1,:) double        % ordered station IDs
        Splines_        cell                % {1 x nSeg} splines for each segment
        EdgeIndices_    (1,:) double        % edge index per segment

        Speed_          (1,1) double = 1    % cruising speed (units/s)
        Dt_             (1,1) double = 0.01 % time step
        AccelRate_      (1,1) double = 0    % acceleration magnitude (units/s^2), 0 = constant speed
        BrakeRate_      (1,1) double = 0    % braking magnitude (units/s^2), 0 = constant speed

        History_        cell = {}           % {step, stationId, KinematicState}
        Clock_          (1,1) double = 0    % total elapsed time
    end

    % Properties that state objects need read/write access to
    properties (Access = {?OceanMath.Transit.AgentStates.IAgentState, ...
                          ?OceanMath.Transit.AgentStates.RunningState, ...
                          ?OceanMath.Transit.AgentStates.DwellingState, ...
                          ?OceanMath.Transit.AgentStates.ArrivedState})
        State_                              % active IAgentState object
        CurrentSegIdx_  (1,1) double = 1    % which segment (1..nSeg)
        Simulator_                          % active CurveSimulator
        DwellTime_      (1,1) double = 3.0  % pause at stations (seconds)
        DwellRemaining_ (1,1) double = 0    % countdown at current station
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = TransitAgent(network, lineCode, options)
            % TRANSITAGENT  Create a train agent on a transit line.
            %
            %   agent = TransitAgent(tn, 'NSL')
            %   agent = TransitAgent(tn, 'NSL', 'Speed', 5, 'DwellTime', 1)
            arguments
                network
                lineCode   (1,:) char
                options.Speed     (1,1) double {mustBePositive} = 1
                options.Dt        (1,1) double {mustBePositive} = 0.01
                options.DwellTime (1,1) double {mustBeNonnegative} = 3.0
                options.AccelRate (1,1) double {mustBeNonnegative} = 0
                options.BrakeRate (1,1) double {mustBeNonnegative} = 0
                options.StartSegment (1,1) double {mustBePositive, mustBeInteger} = 1
            end

            obj.Network_  = network;
            obj.LineCode_ = lineCode;
            obj.Speed_    = options.Speed;
            obj.Dt_       = options.Dt;
            obj.DwellTime_ = options.DwellTime;
            obj.AccelRate_ = options.AccelRate;
            obj.BrakeRate_ = options.BrakeRate;

            % Resolve route from TransitLine
            line = network.getLine(lineCode);
            obj.Route_ = line.getRoute();

            if numel(obj.Route_) < 2
                error('OceanMath:Transit:TransitAgent:TooFewStations', ...
                    'Line "%s" must have at least 2 stations.', lineCode);
            end

            % Resolve splines for each segment
            nSeg = numel(obj.Route_) - 1;
            
            if options.StartSegment > nSeg
                error('OceanMath:Transit:TransitAgent:InvalidStart', ...
                    'StartSegment %d exceeds total line segments %d.', options.StartSegment, nSeg);
            end
            
            obj.Splines_     = cell(1, nSeg);
            obj.EdgeIndices_ = zeros(1, nSeg);

            for i = 1:nSeg
                src = obj.Route_(i);
                tgt = obj.Route_(i+1);
                edgeIdx = network.findEdgeIndex(src, tgt);
                if edgeIdx == 0
                    error('OceanMath:Transit:TransitAgent:MissingRoute', ...
                        'No route between stations %d and %d on line "%s".', ...
                        src, tgt, lineCode);
                end
                
                spl = network.getSpline(edgeIdx);
                edge = network.getRouteEdge(edgeIdx);
                
                % If the network's directed edge actually points backwards
                % relative to our travel direction, reverse the spline's parameterization!
                if edge.Target == src
                    spl = OceanMath.Geometry.ReversedCurve(spl);
                end
                
                obj.EdgeIndices_(i) = edgeIdx;
                obj.Splines_{i} = spl;
            end

            % Create simulator for the first segment
            obj.CurrentSegIdx_ = options.StartSegment;
            obj.Simulator_ = obj.buildSimulator_(obj.CurrentSegIdx_);

            % Initial state: Running
            obj.State_ = OceanMath.Transit.AgentStates.RunningState();

            % Record initial state
            obj.recordHistory_(obj.Simulator_.getState());
        end

        % ============================================================== %
        %  Simulation — delegated to State_
        % ============================================================== %

        function step(obj, dtOverride, realDtOverride)
            % STEP  Advance the agent by one time step dt.
            %   Behavior depends on the active state (Running/Dwelling/Arrived).
            if nargin > 1
                obj.Dt_ = dtOverride;
            end
            realDt = obj.Dt_;
            if nargin > 2
                realDt = realDtOverride;
            end
            
            obj.Clock_ = obj.Clock_ + obj.Dt_;
            obj.State_.step(obj, obj.Dt_, realDt);
        end

        function runToEnd(obj)
            % RUNTOEND  Step until the agent has arrived.
            arguments
                obj
            end
            maxIter = 1e6;  % safety
            count = 0;
            while ~obj.isArrived() && count < maxIter
                obj.step();
                count = count + 1;
            end
        end

        function stepN(obj, n)
            % STEPN  Step N times.
            arguments
                obj
                n (1,1) double {mustBePositive, mustBeInteger}
            end
            for i = 1:n
                obj.step();
                if obj.isArrived(), break; end
            end
        end

        function code = getLineCode(obj)
            % GETLINECODE  Returns the associated transit line's code string.
            code = obj.LineCode_;
        end

        % ============================================================== %
        %  Queries — delegated to State_
        % ============================================================== %

        function pos = getPosition(obj)
            % GETPOSITION  Current [x, y] position on the network.
            arguments
                obj
            end
            pos = obj.State_.getPosition(obj);
        end

        function T = getTangent(obj)
            % GETTANGENT  Current heading direction.
            T = obj.State_.getTangent(obj);
        end

        function p = getProgress(obj)
            % GETPROGRESS  Fraction of total route completed [0, 1].
            arguments
                obj
            end
            p = obj.State_.getProgress(obj);
        end

        function tf = isArrived(obj)
            % ISARRIVED  True when the train has reached the terminal station.
            arguments
                obj
            end
            tf = obj.State_.isArrived();
        end

        % ============================================================== %
        %  State-independent queries
        % ============================================================== %

        function s = getState(obj)
            % GETSTATE  Return the active KinematicState.
            if ~isempty(obj.Simulator_)
                s = obj.Simulator_.getState();
            else
                s = [];
            end
        end

        function h = getHistory(obj)
            % GETHISTORY  Return all recorded snapshots.
            h = obj.History_;
        end

        function t = getClock(obj)
            % GETCLOCK  Current simulation clock.
            t = obj.Clock_;
        end

        function st = getAgentState(obj)
            % GETAGENTSTATE  Return 'running', 'dwelling', or 'arrived'.
            st = obj.State_.getName();
        end

        function d = getDwellRemaining(obj)
            % GETDWELLREMAINING  Return seconds until door closes.
            d = obj.DwellRemaining_;
        end

        function color = getColor(obj)
            % GETCOLOR  Return the line's RGB color.
            line = obj.Network_.getLine(obj.LineCode_);
            color = line.Color;
        end

        function idx = getCurrentSegment(obj)
            % GETCURRENTSEGMENT  Return current segment index.
            idx = obj.CurrentSegIdx_;
        end

        function n = getSegmentCount(obj)
            % GETSEGMENTCOUNT  Return total number of segments on route.
            n = numel(obj.Route_) - 1;
        end

        function spline = getActiveSpline(obj)
            % GETACTIVESPLINE  Return the IParametricCurve for the current segment.
            if obj.CurrentSegIdx_ <= numel(obj.Splines_)
                spline = obj.Splines_{obj.CurrentSegIdx_};
            else
                spline = [];
            end
        end

        % ============================================================== %
        %  State management (called by state objects)
        % ============================================================== %

        function setState(obj, newState)
            % SETSTATE  Transition to a new IAgentState.
            obj.State_ = newState;
        end

        function recordHistory_(obj, s)
            % RECORDHISTORY_  Append a KinematicState snapshot to history.
            obj.History_{end+1} = struct( ...
                'Time', obj.Clock_, ...
                'StationIdx', obj.CurrentSegIdx_, ...
                'SegIdx', obj.CurrentSegIdx_, ...
                'State', s);
        end

        function advanceToNextSegment_(obj)
            % ADVANCETONEXTSEGMENT_  Move to the next segment's spline.
            obj.CurrentSegIdx_ = obj.CurrentSegIdx_ + 1;
            obj.Simulator_ = obj.buildSimulator_(obj.CurrentSegIdx_);
            obj.State_ = OceanMath.Transit.AgentStates.RunningState();
        end

        % ============================================================== %
        %  Display
        % ============================================================== %

        function describe(obj)
            % DESCRIBE  Print a human-readable status.
            arguments
                obj
            end
            names = obj.Network_.getStationNames();
            nSeg  = numel(obj.Route_) - 1;
            fprintf('TransitAgent [%s]:\n', obj.LineCode_);
            fprintf('  Route:    %s\n', ...
                strjoin(names(obj.Route_), ' -> '));
            fprintf('  State:    %s\n', obj.State_.getName());
            fprintf('  Segment:  %d / %d\n', obj.CurrentSegIdx_, nSeg);
            fprintf('  Progress: %.1f%%\n', obj.getProgress() * 100);
            fprintf('  Clock:    %.2f s\n', obj.Clock_);
            if obj.State_.getName() == "dwelling"
                fprintf('  Dwell:    %.2f s remaining\n', obj.DwellRemaining_);
            end
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function sim = buildSimulator_(obj, segIdx)
            %Theory: Kinematics Energy K = 1/2 mv^2, Work W = Fd
            %Since F = ma, W = mad
            %To stop a moving object, we must remove all its Kinetic Energy: 
            %   1/2 m v^2 = mad => d = v^2 / (2a)
        
            spline = obj.Splines_{segIdx};
        
            if obj.AccelRate_ > 0 && obj.BrakeRate_ > 0
                vCruise = obj.Speed_;     % target cruise speed
                aRate   = obj.AccelRate_; % acceleration magnitude
                bRate   = obj.BrakeRate_; % braking magnitude
                L       = spline.totalLength(); % segment arc length
        
                % Distances needed to accel to cruise and brake from cruise
                %   v^2 = 2*a*d  =>  d = v^2 / (2*a)
                dAccel = vCruise^2 / (2 * aRate);
                %The amount of distance needed to brake from cruise speed to 0
                dBrake = vCruise^2 / (2 * bRate);
        
                if dAccel + dBrake <= L % Long track
                    %Cruise = Speed is constant
                    % --- Full trapezoidal profile (can reach cruise) ---
                    % vf = vi + at => t = (vf - vi) / a
                    
                    %Time to Accelerate to Cruise Speed
                    tAccel     = vCruise / aRate;
                    
                    %Distance to Cruise. L = d_accel + d_cruise + d_brake
                    %d_cruise = L - d_accel - d_brake
                    dCruise    = L - dAccel - dBrake;
                    
                    %Time to Cruise. d = vt + 1/2 at^2. Cruise phase, a is zero, so d = vt, t = d / v
                    tCruise    = dCruise / vCruise;
                    
                    tBrakeStart = tAccel + tCruise;
                    % Assumption: constant acceleration vf = vi + at -> 0 = v_cruise + (-bRate)*t_brake -> t_brake = v_cruise / bRate
                    tTotal     = tBrakeStart + vCruise / bRate; % Absolute time to stop
                else %Short Track
                    % --- Triangle profile (segment too short for cruise) ---
                    %Because of no cruise phase, L = d_accel + d_brake
                    % L = v_peak^2 / (2*aRate) + v_peak^2 / (2*bRate) ( Substitute d = v^2 / (2*a) into L = d_accel + d_brake )
                    vPeak   = sqrt(2 * L / (1/aRate + 1/bRate));
                    %Time to accelerate to peak speed. v = a * t -> t = v / a
                    tAccel  = vPeak / aRate;
                    
                    %Because the cruise duration is 0, the moment acce finish is the exact same moment braking begin
                    tBrakeStart = tAccel;
                    tTotal  = tAccel + vPeak / bRate; % vPeak/bRate calculate is the same as if part
                end
        
                % Build acceleration function a(t)
                accelFn = @(t) OceanMath.Transit.TransitAgent.trapezoidalAccel_(t, tAccel, tBrakeStart, tTotal, aRate, bRate);
        
                sim = OceanMath.Simulation.CurveSimulator(spline, ...
                    'InitialVelocity', 0, ...
                    'AccelProfile', accelFn, ...
                    'Dt', obj.Dt_, ...
                    'Loop', false, ...
                    'MaxVelocity', obj.Speed_ * 1.1);
            else
                % --- Constant velocity (original behavior) ---
                sim = OceanMath.Simulation.CurveSimulator(spline, ...
                    'InitialVelocity', obj.Speed_, ...
                    'Acceleration', 0, ...
                    'Dt', obj.Dt_, ...
                    'Loop', false, ...
                    'MaxVelocity', obj.Speed_ * 2);
            end
        end
    end

    % ------------------------------------------------------------------ %
    methods (Static, Hidden)
        function a = trapezoidalAccel_(t, tAccel, tBrakeStart, tTotal, aRate, bRate)
            if t < tAccel
                a = aRate;
            elseif t < tBrakeStart
                a = 0;
            elseif t < tTotal
                a = -bRate;
            else
                a = aRate * 0.1;  % gentle push to reach segment end
            end
        end
    end
end
