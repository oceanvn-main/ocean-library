classdef ChainEnv < OceanStats.RL.Env.TabularEnvironment
    % CHAINENV  The Chain problem (Strens 2000, "A Bayesian framework for reinforcement learning"; Dearden,
    % Friedman & Russell 1998, "Bayesian Q-learning"): a continuing task that punishes myopic exploration.
    %
    %   env = OceanStats.RL.Env.Classic.ChainEnv('NumStates', 5, 'Slip', 0.2, 'SmallReward', 2, 'LargeReward', 10)
    %
    % States 1..N, start in state 1, no terminal state. Action 1 ("a", forward): move to s+1 with reward 0; in
    % state N stay and receive LargeReward. Action 2 ("b", back): return to state 1 with reward SmallReward.
    % With probability Slip the OTHER action's effect happens instead. A greedy or ε-greedy learner tends to
    % settle on "b" (immediate SmallReward); the optimal policy (for discount factors close to 1) walks to the
    % end and collects LargeReward.
    methods
        function obj = ChainEnv(options)
            arguments
                options.NumStates (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.NumStates, 2)} = 5
                options.Slip (1,1) double {mustBeInRange(options.Slip, 0, 1)} = 0.2
                options.SmallReward (1,1) double {mustBeFinite} = 2
                options.LargeReward (1,1) double {mustBeFinite} = 10
            end
            N = options.NumStates; p = options.Slip;
            model = cell(N, 2);
            for s = 1:N
                if s < N, fwd = s + 1; rf = 0; else, fwd = N; rf = options.LargeReward; end
                rb = options.SmallReward;
                model{s, 1} = obj.merge_([fwd; 1], [1 - p; p], [rf; rb]);
                model{s, 2} = obj.merge_([1; fwd], [1 - p; p], [rb; rf]);
            end
            st = zeros(N, 1); st(1) = 1;
            obj.setModel_(model, false(N, 1), st);
        end
    end
    methods (Static, Access = private)
        function m = merge_(next, prob, reward)
            % Drop zero-probability outcomes (Slip = 0 or 1).
            k = prob > 0;
            m = struct('Next', next(k), 'Prob', prob(k), 'Reward', reward(k));
        end
    end
end
