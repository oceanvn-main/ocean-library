classdef SimulationHistory < handle
    % SIMULATIONHISTORY  Memento tape recorder for ISimulator event streams.
    %
    % Listens to the 'StateChanged' event of an ISimulator and records every
    % transition into a pre-allocated fixed-capacity buffer (circular or
    % growing). Enables playback scrubbing, rewinding, and re-analysis of
    % any recorded trajectory — without re-running the math engine.
    %
    % Optimization Notes:
    %   - Uses a pre-allocated uint32 array (StateBuffer_) to store the
    %     trajectory. This avoids repeated memory reallocation and is 4x
    %     more compact than a double array for state indices.
    %   - When capacity is exceeded, the buffer doubles in size
    %     (amortized O(1) append), in line with MATLAB best practices.
    %   - getStateSequence() returns a view slice — zero extra allocation.
    %   - Optional metric cache: stores computed metric values alongside the
    %     trajectory so repeated queries don't recompute.
    %
    % Usage:
    %   P   = [0.7 0.3; 0.4 0.6];
    %   mc  = OceanMath.Information.Processes.MarkovChain(P);
    %   sim = OceanMath.Simulation.MarkovSimulator(mc);
    %   hist = OceanMath.Simulation.SimulationHistory(sim);
    %
    %   sim.runFor(1000);
    %   seq = hist.getStateSequence();       % (1001 x 1) uint32 vector
    %   h   = hist.getMetricAtStep(500);     % entropy at t=500 (if computed)
    %
    % See also: OceanMath.Simulation.ISimulator
    %           OceanMath.Simulation.MarkovSimulator

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        % Source simulator reference
        Simulator_     % OceanMath.Simulation.ISimulator (typed in constructor args)

        % State trajectory buffer
        % Pre-allocated as uint32 to minimize memory; grows as needed.
        StateBuffer_   (:,1) uint32
        Count_         (1,1) double = 0   % valid entries in StateBuffer_

        % Metric value cache (optional)
        % MetricBuffer_(t) = metric value recorded at time t, NaN if not set.
        MetricBuffer_  (:,1) double

        % Listener handle, stored so we can cleanly disconnect
        Listener_
    end

    properties (Constant, Access = private)
        INITIAL_CAPACITY = 1024   % starting buffer blocks — 4KB for uint32
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = SimulationHistory(simulator, options)
            % SIMULATIONHISTORY  Attach to an ISimulator and start recording.
            %
            % INPUTS:
            %   simulator - ISimulator instance to listen to
            %
            % Name-Value:
            %   InitialCapacity (double): number of steps to pre-allocate.
            %       Default: 1024. For long runs, set higher to avoid resizing.
            arguments
                simulator OceanMath.Simulation.ISimulator
                options.InitialCapacity (1,1) double ...
                    {mustBePositive, mustBeInteger} = ...
                    OceanMath.Simulation.SimulationHistory.INITIAL_CAPACITY
            end

            obj.Simulator_  = simulator;

            % Pre-allocate buffers
            cap                = options.InitialCapacity;
            obj.StateBuffer_   = zeros(cap, 1, 'uint32');
            obj.MetricBuffer_  = NaN(cap, 1);

            % --- Record the initial state (t=0) before any stepping ---
            s0 = uint32(simulator.getCurrentState());
            obj.appendEntry_(s0, NaN);

            % --- Wire the listener ---
            obj.Listener_ = addlistener(simulator, 'StateChanged', ...
                @(src, evt) obj.onStateChanged_(evt));
        end

        % ---- Read API ---- %

        function seq = getStateSequence(obj)
            % GETSTATESEQUENCE  Return the recorded state index trajectory.
            %
            % OUTPUTS:
            %   seq - (N x 1) uint32, states from t=0 to t=T (no extra copy)
            seq = obj.StateBuffer_(1:obj.Count_);
        end

        function T = getLength(obj)
            % GETLENGTH  Return the number of recorded steps (including t=0).
            T = obj.Count_;
        end

        function clearHistory(obj)
            % CLEARHISTORY  Wipe all recorded history and reset the count to zero.
            %
            % The simulator listener is kept attached, so new steps will
            % continue to be recorded after calling clearHistory().
            % Typically called together with sim.reset() for a fresh start.
            obj.Count_       = 0;
            obj.StateBuffer_(:) = 0;
            obj.MetricBuffer_(:) = NaN;
        end

        function s = getStateAtStep(obj, t)
            % GETSTATEATSTEP  Retrieve the recorded state at a specific step.
            %
            % INPUTS:
            %   t - integer index, 1-indexed. t=1 is the initial state (t=0).
            obj.validateIndex_(t);
            s = double(obj.StateBuffer_(t));
        end

        function m = getMetricAtStep(obj, t)
            % GETMETRICATSTEP  Retrieve any cached metric value at step t.
            %
            % Metric values are not automatically computed — they must be
            % injected via recordMetricAtStep() externally (e.g., by the GUI).
            %
            % OUTPUTS:
            %   m - scalar double, NaN if no metric was recorded at this step.
            obj.validateIndex_(t);
            m = obj.MetricBuffer_(t);
        end

        function recordMetricAtStep(obj, t, value)
            % RECORDMETRICATSTEP  Store a metric value for the given step.
            %
            % Called by the simulator/GUI to inject externally-computed metric
            % values (e.g., entropy rate) into the history for later retrieval.
            obj.validateIndex_(t);
            obj.MetricBuffer_(t) = value;
        end

        function clear(obj)
            % CLEAR  Wipe the recorded history and reset counters.
            % Preserves allocated capacity — does not free memory.
            obj.StateBuffer_(:)  = 0;
            obj.MetricBuffer_(:) = NaN;
            obj.Count_           = 0;

            % Re-record current state as the new t=0
            s0 = uint32(obj.Simulator_.getCurrentState());
            obj.appendEntry_(s0, NaN);
        end

        function disconnect(obj)
            % DISCONNECT  Stop listening to the simulator without destroying.
            if ~isempty(obj.Listener_)
                delete(obj.Listener_);
                obj.Listener_ = [];
            end
        end

        function delete(obj)
            obj.disconnect();
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function onStateChanged_(obj, eventData)
            % ONSTATECHANGED_  Listener callback — O(1) amortized per step.
            obj.appendEntry_(uint32(eventData.NewState), NaN);
        end

        function appendEntry_(obj, stateUint, metricVal)
            % APPENDENTRY_  Append one entry; double buffer if full.
            %
            % Growth strategy: double capacity when full (amortized O(1)).
            % This matches MATLAB end+1 semantics but on a pre-typed buffer.
            obj.Count_ = obj.Count_ + 1;

            if obj.Count_ > numel(obj.StateBuffer_)
                newCap = numel(obj.StateBuffer_) * 2;
                obj.StateBuffer_(end+1:newCap) = uint32(0);
                obj.MetricBuffer_(end+1:newCap) = NaN;
                % Note: MATLAB extends arrays with end+1:N via one allocation
            end

            obj.StateBuffer_(obj.Count_)  = stateUint;
            obj.MetricBuffer_(obj.Count_) = metricVal;
        end

        function validateIndex_(obj, t)
            if t < 1 || t > obj.Count_ || mod(t, 1) ~= 0
                error('OceanMath:Simulation:SimulationHistory:BadIndex', ...
                    'Step index must be an integer in [1, %d]. Got %g.', ...
                    obj.Count_, t);
            end
        end
    end
end
