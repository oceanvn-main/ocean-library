classdef (Abstract) IMonteCarloStrategy < handle
    % IMONTECARLOSTRATEGY Abstract base class for Monte Carlo Algorithm Strategies.
    %
    % Encapsulates tabular Monte Carlo algorithms from Sutton & Barto Chapter 5
    % and Section 5.8 using the Strategy Design Pattern.
    %
    % Concrete strategy subclasses:
    %   - FirstVisitPredictionStrategy
    %   - EveryVisitPredictionStrategy
    %   - ExploringStartsStrategy
    %   - OnPolicyControlStrategy
    %   - OffPolicyPredictionStrategy
    %   - OffPolicyControlStrategy
    %   - DiscountingAwarePredictionStrategy
    %
    % 3-PILLAR COMPONENT CONTRACT:
    %   - Pillar 1 (Symbolic φ): getMetadata() returning LaTeX formulas and symbols dict.
    %   - Pillar 2 (Computational f): execute() running tabular RL loop.
    %   - Pillar 3 (Diagram D): getPipelineSpec() defining ports spec.
    %
    % Package: OceanStats.RL.MonteCarlo.Strategies
    % Reference: Sutton & Barto (2018) Chapter 5

    methods (Abstract)
        % EXECUTE Executes the Monte Carlo algorithm using injected solver and env handles.
        %
        % Outputs:
        %   V       - (N x 1) state-value table V(s)
        %   Q       - (N x A) action-value table Q(s,a)
        %   pi      - (N x 1) greedy policy (1-based action index)
        %   history - struct with convergence fields
        [V, Q, pi, history] = execute(obj, solver, env)
    end

    methods (Static, Abstract)
        meta = getMetadata()
        spec = getPipelineSpec()
    end
end
