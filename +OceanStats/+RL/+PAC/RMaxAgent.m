classdef RMaxAgent < OceanStats.RL.PAC.IExplorationAgent
    % RMAXAGENT  R-MAX (Brafman & Tennenholtz 2002), in the discounted form and pseudo-code of Strehl, Li &
    % Littman 2009, "Reinforcement learning in finite MDPs: PAC analysis", Section 2.1 and Algorithm 1.
    %
    %   agent = OceanStats.RL.PAC.RMaxAgent(numStates, numActions, 'KnownCount', m, 'MaxReward', Rmax, ...
    %               'DiscountFactor', 0.95, 'Epsilon1', 0.01, 'Optimism', [])
    %   tr = agent.run(env, 'Steps', 10000, 'Seed', 1)        (or act / observe yourself)
    %   pi = agent.greedyPolicy()
    %
    % Optimism in the face of uncertainty. A pair (s,a) is KNOWN once tried m times; only its first m samples
    % form the empirical model R̂, T̂ (TransitionCounts with Cap = m). Action values solve
    %
    %   Q(s,a) = R̂(s,a) + γ Σ_s' T̂(s'|s,a) max_a' Q(s',a')     if n(s,a) >= m          (Eq. 4)
    %   Q(s,a) = U(s,a)                                          otherwise
    %
    % with U(s,a) an upper bound on the true value, by default Vmax = Rmax / (1 - γ) (Optimism = scalar or
    % S x A to supply a tighter heuristic U). The agent always acts greedily: a = argmax_a Q(s,a) over the
    % admissible actions (ties -> lowest index). Q starts at U; each time a pair becomes known, value iteration
    % on the known pairs runs ceil( ln(max|U| / ε1) / (1-γ) ) sweeps from the current Q (Algorithm 1, lines
    % 17-24; the paper's count for rewards in [0,1], here scaled so ε1 is in reward-value units), giving action
    % values within ε1 of the model's (Proposition 4). Sweeps update all known pairs synchronously (the
    % pseudo-code updates in place; same contraction bound).
    % Extensions (labelled, not in the paper): a terminated transition contributes no continuation value
    % (episodic tasks); a state whose admissible actions are all reported absent has value 0.
    %
    % Rewards must lie in (-Inf, MaxReward]; a larger reward raises an error (U would no longer be an upper
    % bound). Deterministic: no random numbers are drawn by the agent.
    % LastMode: "known" (greedy action already known) or "optimistic" (greedy action still unknown).
    properties (SetAccess = private)
        KnownCount (1,1) double
        Epsilon1 (1,1) double
        Q double                      % S x A action values (admissible actions)
        U double                      % S x A optimistic values of unknown pairs
        Sweeps (1,1) double           % value-iteration sweeps per replanning
        Replans (1,1) double = 0
    end
    methods
        function obj = RMaxAgent(numStates, numActions, opts)
            arguments
                numStates (1,1) double {mustBeInteger, mustBePositive}
                numActions (1,1) double {mustBeInteger, mustBePositive}
                opts.KnownCount (1,1) double {mustBeInteger, mustBePositive} = 10
                opts.MaxReward (1,1) double {mustBeFinite} = 1
                opts.DiscountFactor (1,1) double {mustBeGreaterThanOrEqual(opts.DiscountFactor, 0), mustBeLessThan(opts.DiscountFactor, 1)} = 0.95
                opts.Epsilon1 (1,1) double {mustBePositive} = 0.01
                opts.Optimism double = []
            end
            obj.init_(numStates, numActions, opts.DiscountFactor, opts.MaxReward, opts.KnownCount);
            obj.KnownCount = opts.KnownCount; obj.Epsilon1 = opts.Epsilon1;
            vmax = opts.MaxReward / (1 - opts.DiscountFactor);
            if isempty(opts.Optimism)
                obj.U = vmax * ones(numStates, numActions);
            elseif isscalar(opts.Optimism)
                obj.U = opts.Optimism * ones(numStates, numActions);
            elseif isequal(size(opts.Optimism), [numStates numActions])
                obj.U = opts.Optimism;
            else
                error('OceanStats:RL:PAC:BadOptimism', 'Optimism must be a scalar or a %d x %d matrix.', numStates, numActions);
            end
            obj.Q = obj.U;
            % Proposition 4 in reward units: value iteration from values within Vmax of Q* is within ε1 after
            % ceil( ln(Vmax / ε1) / (1-γ) ) sweeps, Vmax = max|U|; with rewards in [0,1] (U = 1/(1-γ)) this is the
            % paper's ceil( ln(1/(ε1 (1-γ))) / (1-γ) ). At least one sweep.
            vbound = max(max(abs(obj.U(:))), eps);
            obj.Sweeps = max(1, ceil(log(vbound / opts.Epsilon1) / (1 - opts.DiscountFactor)));
        end

        function a = act(obj, s, actionMask)
            obj.noteMask_(s, actionMask);
            adm = find(obj.Mask(s, :));
            [~, k] = max(obj.Q(s, adm));
            a = adm(k);
            if obj.Counts.N(s, a) >= obj.KnownCount, obj.LastMode = "known"; else, obj.LastMode = "optimistic"; end
        end

        function observe(obj, s, a, r, s2, terminated)
            obj.checkReward_(r, -Inf);
            if obj.Counts.record(s, a, r, s2, terminated)
                obj.replan_();
            end
        end

        function tf = isKnown(obj), tf = obj.Counts.N >= obj.KnownCount; end
        function pi = greedyPolicy(obj)
            % Greedy action of every state under the current Q (admissible actions; ties -> lowest index).
            Qm = obj.Q; Qm(~obj.Mask) = -Inf;
            [~, pi] = max(Qm, [], 2);
        end
    end
    methods (Access = private)
        function replan_(obj)
            % Algorithm 1, lines 18-24: value iteration over the known pairs, unknown pairs stay at U.
            [R, P] = obj.Counts.estimates();
            known = obj.isKnown() & obj.Mask;
            S = obj.NumStates; A = obj.NumActions;
            for i = 1:obj.Sweeps
                Qm = obj.Q; Qm(~obj.Mask) = -Inf;
                V = max(Qm, [], 2); V(~any(obj.Mask, 2)) = 0;
                Qk = R + obj.DiscountFactor * reshape(full(P * V), S, A);
                obj.Q(known) = Qk(known);
            end
            obj.Replans = obj.Replans + 1;
        end
    end
end
