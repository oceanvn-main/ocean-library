classdef MarkovSimulator < OceanMath.Simulation.ISimulator
    % MARKOVSIMULATOR  Concrete time-stepping engine for a Markov Chain.
    %
    % Implements ISimulator for a finite, discrete-time Markov Chain.
    % Each call to step() samples the next state using inverse-transform
    % sampling from the current row of the transition matrix, updates the
    % internal clock, and fires the 'StateChanged' event.
    %
    % Optimization:
    %   - The full (K x K) transition matrix and CDF matrix (cumulative row
    %     sums) are pre-computed and cached on construction, so each step()
    %     is a pure O(K) operation: one rand() + one find().
    %   - State indices are stored as uint32 to minimize memory footprint
    %     for long trajectories.
    %
    % Usage:
    %   P   = [0.7 0.3; 0.4 0.6];
    %   mc  = OceanMath.Information.Processes.MarkovChain(P);
    %   sim = OceanMath.Simulation.MarkovSimulator(mc);
    %
    %   % Listen to state changes:
    %   hist = OceanMath.Simulation.SimulationHistory(sim);
    %   for t = 1:100; sim.step(); end
    %   seq = hist.getStateSequence();
    %
    % See also: OceanMath.Simulation.ISimulator
    %           OceanMath.Simulation.SimulationHistory
    %           OceanMath.Information.Processes.MarkovChain

    % ------------------------------------------------------------------ %
    properties (Access = private)
        % Pre-computed CDF matrix for O(K) sampling per step
        % CdfMatrix_(i, :) = cumsum(P(i, :)) — built once, reused every step
        CdfMatrix_ (:,:) double
        K_         (1,1) double
    end

    % ------------------------------------------------------------------ %
    methods
        function obj = MarkovSimulator(process, options)
            % MARKOVSIMULATOR  Construct and immediately configure.
            %
            % INPUTS:
            %   process - IStochasticProcess (e.g. MarkovChain)
            %
            % Name-Value:
            %   StartState (double): initial state index (1-indexed).
            %       Default: 0 — sample from stationary distribution.
            arguments
                process OceanMath.Information.Processes.IStochasticProcess
                options.StartState (1,1) double = 0
            end

            obj.Process_ = process;
            obj.K_       = process.getNumStates();

            % --- Pre-compute CDF matrix (core optimization) ---
            % This moves the cumsum() call OUT of step(). Each step() is now
            % just: find(rand() <= CdfMatrix_(currentRow, :))
            P              = process.getTransitionMatrix();
            obj.CdfMatrix_ = cumsum(P, 2);   % row-wise CDF, (K x K)

            % --- Initialize starting state ---
            if options.StartState == 0
                obj.State_ = obj.sampleFromStationary_();
            else
                obj.State_ = options.StartState;
            end
            obj.Time_ = 0;
        end

        % ---- ISimulator Implementation ---- %

        function step(obj)
            % STEP  Sample next state from pre-computed CDF row and fire event.
            %
            % O(K) per call: one rand(), one find() on a cached row vector.
            % No matrix allocation, no cumsum recomputed.
            oldState   = obj.State_;
            cdfRow     = obj.CdfMatrix_(oldState, :);
            obj.State_ = find(rand() <= cdfRow, 1, 'first');
            obj.Time_  = obj.Time_ + 1;

            obj.notifyStateChange_(oldState, obj.State_, obj.Time_);
        end

        function reset(obj, startState)
            % RESET  Return to t=0 and optionally set a new start state.
            arguments
                obj
                startState (1,1) double = 0
            end
            obj.Time_ = 0;
            if startState == 0
                obj.State_ = obj.sampleFromStationary_();
            else
                obj.State_ = startState;
            end
        end

        function teleport(obj, s)
            % TELEPORT  Force the current state to s and fire StateChanged.
            %
            % This is equivalent to an external injection of state — the chain
            % jumps to state s without respecting transition probabilities.
            % SimulationHistory will record this as a new time step.
            %
            % INPUTS:
            %   s - integer state index in [1, K]
            arguments
                obj
                s (1,1) double {mustBePositive, mustBeInteger}
            end
            oldState   = obj.State_;
            obj.State_ = s;
            obj.Time_  = obj.Time_ + 1;
            obj.notifyStateChange_(oldState, s, obj.Time_);
        end

        function runFor(obj, T)
            % RUNFOR  Step T times in a tight loop.
            %
            % Equivalent to calling step() T times, but avoids the small
            % overhead of the public method dispatch per iteration.
            arguments
                obj
                T (1,1) double {mustBePositive, mustBeInteger}
            end
            for t = 1:T
                obj.step();
            end
        end
    end

    % ------------------------------------------------------------------ %
    methods (Access = private)
        function s = sampleFromStationary_(obj)
            % SAMPLEFROMSTATIONARY_  Draw start state from stationary dist.
            mu = obj.Process_.getStationaryDistribution();
            p  = mu.evaluate(mu.getSupport());
            s  = find(rand() <= cumsum(p(:)'), 1, 'first');
        end
    end
end
