classdef TransitSimulation < handle
    % TRANSITSIMULATION  Orchestrator that runs multiple TransitAgents in lockstep.
    %
    % Creates and ticks agents across a TransitNetwork. Provides a
    % per-frame snapshot of all agent positions for rendering.
    %
    % Usage:
    %   sim = OceanMath.Transit.TransitSimulation(tn, 'Dt', 0.02);
    %   sim.addAgent('NSL', 'Speed', 3);
    %   sim.addAgent('EWL', 'Speed', 4);
    %   while ~sim.isAllArrived()
    %       sim.step();
    %       positions = sim.getPositions();
    %   end
    %
    % See also: OceanMath.Transit.TransitAgent, OceanMath.Transit.TransitNetwork

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        Network_    % TransitNetwork reference
        Agents_     cell = {}   % cell array of TransitAgent
        Dt_         (1,1) double = 0.02
        Clock_      (1,1) double = 0
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = TransitSimulation(network, options)
            % TRANSITSIMULATION  Construct the orchestrator.
            %
            %   sim = TransitSimulation(tn)
            %   sim = TransitSimulation(tn, 'Dt', 0.02)
            arguments
                network
                options.Dt (1,1) double {mustBePositive} = 0.02
            end
            obj.Network_ = network;
            obj.Dt_      = options.Dt;
        end

        % ============================================================== %
        %  Agent Management
        % ============================================================== %

        function agent = addAgent(obj, lineCode, options)
            % ADDAGENT  Create a TransitAgent on a given line and add to the sim.
            %
            %   sim.addAgent('NSL')
            %   sim.addAgent('NSL', 'Speed', 5, 'DwellTime', 1)
            %   sim.addAgent('NSL', 'Speed', 5, 'AccelRate', 1, 'BrakeRate', 1.2, 'StartSegment', 3)
            arguments
                obj
                lineCode   (1,:) char
                options.Speed     (1,1) double {mustBePositive} = 1
                options.DwellTime (1,1) double {mustBeNonnegative} = 0.5
                options.AccelRate (1,1) double {mustBeNonnegative} = 0
                options.BrakeRate (1,1) double {mustBeNonnegative} = 0
                options.StartSegment (1,1) double {mustBePositive, mustBeInteger} = 1
            end
            agent = OceanMath.Transit.TransitAgent(obj.Network_, lineCode, ...
                'Speed', options.Speed, ...
                'Dt', obj.Dt_, ...
                'DwellTime', options.DwellTime, ...
                'AccelRate', options.AccelRate, ...
                'BrakeRate', options.BrakeRate, ...
                'StartSegment', options.StartSegment);
            obj.Agents_{end+1} = agent;
        end

        function N = getAgentCount(obj)
            N = numel(obj.Agents_);
        end

        function agent = getAgent(obj, idx)
            agent = obj.Agents_{idx};
        end
        
        function agents = getAgents(obj)
            agents = obj.Agents_;
        end
        
        function reset(obj)
            % RESET  Clear all agents and reset clock to zero.
            obj.Agents_ = {};
            obj.Clock_  = 0;
        end

        % ============================================================== %
        %  Simulation
        % ============================================================== %

        function step(obj, dtOverride, realDtOverride)
            % STEP  Tick all agents by one dt or an overridden dt.
            if nargin > 1
                dt = dtOverride;
            else
                dt = obj.Dt_;
            end
            
            realDt = dt;
            if nargin > 2
                realDt = realDtOverride;
            end
            
            obj.Clock_ = obj.Clock_ + dt;
            for i = 1:numel(obj.Agents_)
                obj.Agents_{i}.step(dt, realDt);
            end
        end

        function runFor(obj, nSteps)
            % RUNFOR  Step N times.
            arguments
                obj
                nSteps (1,1) double {mustBePositive, mustBeInteger}
            end
            for i = 1:nSteps
                obj.step();
                if obj.isAllArrived(), break; end
            end
        end

        function runUntil(obj, targetTime)
            % RUNUNTIL  Step until simulation clock reaches targetTime.
            arguments
                obj
                targetTime (1,1) double {mustBePositive}
            end
            while obj.Clock_ < targetTime
                obj.step();
                if obj.isAllArrived(), break; end
            end
        end

        function runToEnd(obj)
            % RUNTOEND  Run until all agents have arrived.
            maxIter = 1e6;
            count = 0;
            while ~obj.isAllArrived() && count < maxIter
                obj.step();
                count = count + 1;
            end
        end

        % ============================================================== %
        %  Queries
        % ============================================================== %

        function positions = getPositions(obj)
            % GETPOSITIONS  [nAgents x 2] matrix of current positions.
            nA = numel(obj.Agents_);
            positions = zeros(nA, 2);
            for i = 1:nA
                positions(i, :) = obj.Agents_{i}.getPosition();
            end
        end

        function colors = getColors(obj)
            % GETCOLORS  [nAgents x 3] matrix of agent line colors.
            nA = numel(obj.Agents_);
            colors = zeros(nA, 3);
            for i = 1:nA
                colors(i, :) = obj.Agents_{i}.getColor();
            end
        end

        function tf = isAllArrived(obj)
            % ISALLARRIVED  True when every agent has arrived.
            tf = true;
            for i = 1:numel(obj.Agents_)
                if ~obj.Agents_{i}.isArrived()
                    tf = false;
                    return;
                end
            end
        end

        function t = getClock(obj)
            t = obj.Clock_;
        end

        % ============================================================== %
        %  Display
        % ============================================================== %

        function describe(obj)
            % DESCRIBE  Print a summary of all agents.
            fprintf('TransitSimulation:\n');
            fprintf('  Agents: %d | Clock: %.2f s | dt: %.4f s\n', ...
                numel(obj.Agents_), obj.Clock_, obj.Dt_);
            for i = 1:numel(obj.Agents_)
                obj.Agents_{i}.describe();
            end
        end
    end
end
