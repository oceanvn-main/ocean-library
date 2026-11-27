classdef IStochasticProcess < handle
    % ISTOCHASTICPROCESS  Abstract interface for all stochastic processes.
    %
    % A stochastic process describes how a system evolves over time through
    % probabilistic state transitions. This interface is the contract that
    % all process implementations must fulfill.
    %
    % By programming against this interface (not concrete classes), the
    % metric strategies in Layer 3 — EntropyRate, HMM bounds, etc. — remain
    % completely decoupled from the specific process type being analyzed.
    %
    % Required Implementations:
    %   dist  = getStationaryDistribution(obj)      - Limiting distribution mu
    %   p     = getTransitionProbability(obj, i, j) - P(X_{t+1}=j | X_t=i)
    %   P     = getTransitionMatrix(obj)             - Full (K x K) matrix P
    %   K     = getNumStates(obj)                    - Cardinality of state space
    %   tf    = isErgodic(obj)                       - Irreducible and aperiodic?
    %
    % Optional Convenience (default implementations provided):
    %   step  = sampleNextState(obj, currentState)  - Monte Carlo single step
    %   seq   = simulate(obj, T, startState)         - Full trajectory of length T
    %
    % See also: OceanMath.Information.Processes.MarkovChain
    %           OceanMath.Information.Processes.HiddenMarkovModel
    %           OceanMath.Information.Processes.RandomWalkGraph

    methods (Abstract)
        % GETSTATIONARYDISTRIBUTION  Compute the stationary distribution mu.
        %
        % The stationary distribution satisfies: mu * P = mu, sum(mu) = 1.
        % This is the long-run fraction of time the process spends in each state.
        %
        % OUTPUTS:
        %   dist - OceanMath.Information.Distributions.DiscreteDistribution
        %          representing the stationary PMF over the K states.
        dist = getStationaryDistribution(obj)

        % GETTRANSITIONPROBABILITY  Query P(X_{t+1} = j | X_t = i).
        %
        % INPUTS:
        %   stateI - integer index of the source state (1-indexed)
        %   stateJ - integer index of the destination state (1-indexed)
        %
        % OUTPUTS:
        %   p - scalar double in [0, 1]
        p = getTransitionProbability(obj, stateI, stateJ)

        % GETTRANSITIONMATRIX  Return the full (K x K) transition matrix P.
        %
        % Rows represent the current state, columns the next state.
        % Each row must sum to 1: P(i,:) is the conditional PMF from state i.
        %
        % OUTPUTS:
        %   P - (K x K) double, row-stochastic matrix
        P = getTransitionMatrix(obj)

        % GETNUMSTATES  Return the number of states K in the process.
        %
        % OUTPUTS:
        %   K - positive integer
        K = getNumStates(obj)

        % ISERGODIC  Check if the process is irreducible and aperiodic.
        %
        % An ergodic chain guarantees a unique stationary distribution and
        % that the chain will converge to it from any starting state.
        %
        % OUTPUTS:
        %   tf - logical scalar
        tf = isErgodic(obj)
    end

    methods
        % SAMPLENEXTSTATE  Draw the next state via Monte Carlo.
        %
        % INPUTS:
        %   currentState - integer index of the current state (1-indexed)
        %
        % OUTPUTS:
        %   nextState - integer index sampled from P(currentState, :)
        function nextState = sampleNextState(obj, currentState)
            row = obj.getTransitionMatrix();
            row = row(currentState, :);
            % Inverse-transform sampling on the transition row
            nextState = find(rand() <= cumsum(row), 1, 'first');
        end

        % SIMULATE  Generate a trajectory of T steps.
        %
        % INPUTS:
        %   T          - positive integer, number of time steps
        %   startState - integer, initial state index (1-indexed).
        %                If empty, sampled from stationary distribution.
        %
        % OUTPUTS:
        %   seq - (T+1 x 1) integer vector of state indices
        function seq = simulate(obj, T, startState)
            arguments
                obj
                T          (1,1) double {mustBePositive, mustBeInteger}
                startState (1,1) double = 0   % 0 = sample from stationary dist
            end
            if startState == 0
                mu         = obj.getStationaryDistribution();
                probs      = mu.evaluate(mu.getSupport());
                startState = find(rand() <= cumsum(probs), 1, 'first');
            end
            seq    = zeros(T + 1, 1);
            seq(1) = startState;
            for t = 1:T
                seq(t+1) = obj.sampleNextState(seq(t));
            end
        end
    end
end
