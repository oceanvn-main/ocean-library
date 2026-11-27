classdef RandomWalkEnv < OceanStats.RL.Env.TabularEnvironment
    % RANDOMWALKENV  The N-state random walk (Sutton & Barto Example 6.2 with N=5; Example 7.1 with N=19).
    %
    % Non-terminal states 2..N+1 (A..E for N=5), terminal ends 1 and N+2, start in the middle.
    % Actions: 1 = left, 2 = right (deterministic). Reward RightReward on entering the right end, LeftReward
    % on entering the left end, 0 otherwise. Under the equiprobable policy and gamma=1 the true values are
    % V(k) = k/(N+1) for k = 1..N (defaults: LeftReward 0, RightReward 1) - a classic prediction benchmark.
    %
    %   env = RandomWalkEnv('NumInner', 5);  env.setSeed(1);
    properties (SetAccess = private)
        NumInner (1,1) double = 5
    end
    methods
        function obj = RandomWalkEnv(options)
            arguments
                options.NumInner (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.NumInner, 1)} = 5
                options.LeftReward (1,1) double = 0
                options.RightReward (1,1) double = 1
            end
            N = options.NumInner;
            S = N + 2;
            model = cell(S, 2);
            for s = 2:S-1
                for a = 1:2
                    nxt = s + (2 * a - 3);                 % a=1 -> s-1, a=2 -> s+1
                    r = 0;
                    if nxt == S, r = options.RightReward; end
                    if nxt == 1, r = options.LeftReward; end
                    model{s, a} = struct('Next', nxt, 'Prob', 1, 'Reward', r);
                end
            end
            terminal = false(S, 1);  terminal([1 S]) = true;
            start = zeros(S, 1);  start(ceil(S / 2)) = 1;                % middle state (left-middle when S is even)
            obj.NumInner = N;
            obj.setModel_(model, terminal, start);
        end
    end
end
