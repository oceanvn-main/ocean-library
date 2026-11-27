classdef TransitionCounts < handle
    % TRANSITIONCOUNTS  Maximum-likelihood tabular model from visit counts (Strehl, Li & Littman 2009, Section
    % 2.1): for every pair (s,a) the empirical mean reward and next-state distribution
    %
    %   R̂(s,a)    = (1/n(s,a)) Σ_i r[i]
    %   T̂(s'|s,a) = n(s,a,s') / n(s,a)        T̂_end(s,a) = n(s,a,end) / n(s,a)   (episode ended)
    %
    %   c = OceanStats.RL.PAC.TransitionCounts(numStates, numActions, 'Cap', m)
    %   reached = c.record(s, a, r, s2, terminated)   true exactly when n(s,a) reaches Cap with this sample
    %   [R, P, Pend] = c.estimates()                  R, Pend: S x A;  P: sparse (S*A) x S, row sub2ind([S A], s, a)
    %
    % Cap = m keeps only the FIRST m samples of each pair, as R-MAX requires ("R-MAX uses only the first m
    % samples in the empirical model"); Cap = Inf keeps all (E³). Seen(s,a) counts every try regardless of Cap.
    % A terminated transition records the episode end instead of a next state (no continuation value).
    properties (SetAccess = private)
        NumStates (1,1) double
        NumActions (1,1) double
        Cap (1,1) double
        N double            % S x A samples in the model (<= Cap)
        Seen double         % S x A all tries
        RewardSum double    % S x A
        EndCount double     % S x A terminated transitions
        NextCount           % sparse (S*A) x S
    end
    methods
        function obj = TransitionCounts(numStates, numActions, opts)
            arguments
                numStates (1,1) double {mustBeInteger, mustBePositive}
                numActions (1,1) double {mustBeInteger, mustBePositive}
                opts.Cap (1,1) double {mustBePositive} = Inf
            end
            obj.NumStates = numStates; obj.NumActions = numActions; obj.Cap = opts.Cap;
            obj.N = zeros(numStates, numActions); obj.Seen = obj.N; obj.RewardSum = obj.N; obj.EndCount = obj.N;
            obj.NextCount = sparse(numStates * numActions, numStates);
        end

        function reached = record(obj, s, a, r, s2, terminated)
            arguments
                obj
                s (1,1) double {mustBeInteger, mustBePositive}
                a (1,1) double {mustBeInteger, mustBePositive}
                r (1,1) double {mustBeFinite}
                s2 (1,1) double
                terminated (1,1) logical
            end
            obj.Seen(s, a) = obj.Seen(s, a) + 1;
            reached = false;
            if obj.N(s, a) >= obj.Cap, return; end
            obj.N(s, a) = obj.N(s, a) + 1;
            obj.RewardSum(s, a) = obj.RewardSum(s, a) + r;
            if terminated
                obj.EndCount(s, a) = obj.EndCount(s, a) + 1;
            else
                if ~(s2 == fix(s2) && s2 >= 1 && s2 <= obj.NumStates)
                    error('OceanStats:RL:PAC:BadState', 'A non-terminal next state must be a state index (1..%d).', obj.NumStates);
                end
                row = sub2ind([obj.NumStates obj.NumActions], s, a);
                obj.NextCount(row, s2) = obj.NextCount(row, s2) + 1;
            end
            reached = obj.N(s, a) == obj.Cap;
        end

        function [R, P, Pend] = estimates(obj)
            n = obj.N; inv = zeros(size(n)); inv(n > 0) = 1 ./ n(n > 0);
            R = obj.RewardSum .* inv;
            Pend = obj.EndCount .* inv;
            k = numel(n);
            P = spdiags(inv(:), 0, k, k) * obj.NextCount;
        end
    end
end
