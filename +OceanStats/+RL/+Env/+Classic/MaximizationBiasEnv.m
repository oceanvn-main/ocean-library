classdef MaximizationBiasEnv < OceanStats.RL.Env.TabularEnvironment
    % MAXIMIZATIONBIASENV  The maximization-bias MDP (Sutton & Barto Example 6.7, Fig. 6.5).
    % State 1 = A (start): action 1 = left -> B (reward 0), action 2 = right -> terminal (reward 0).
    % State 2 = B: NumBActions actions, each -> terminal with reward N(MeanB, StdB^2), MeanB < 0.
    % So 'left' is truly worse in expectation (value MeanB < 0), but the noisy rewards at B make an estimate that
    % maximizes over them look better than it is (the "maximization bias"). get_transitions gives the EXPECTED reward
    % (MeanB), so the exact model is the expected MDP; step() samples the noise. States 3, 4 are terminal.
    methods
        function obj = MaximizationBiasEnv(options)
            arguments
                options.NumBActions (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(options.NumBActions, 1)} = 10
                options.MeanB (1,1) double = -0.1
                options.StdB (1,1) double {mustBeNonnegative} = 1
            end
            nb = options.NumBActions;
            A = max(2, nb);
            model = cell(4, A);
            model{1, 1} = struct('Next', 2, 'Prob', 1, 'Reward', 0);
            model{1, 2} = struct('Next', 3, 'Prob', 1, 'Reward', 0);
            for a = 1:nb
                model{2, a} = struct('Next', 4, 'Prob', 1, 'Reward', options.MeanB);
            end
            terminal = [false; false; true; true];
            obj.setModel_(model, terminal, [1; 0; 0; 0]);
            obj.RewardStd(2, 1:nb) = options.StdB;
        end
    end
end
