function t = benchmark_td_speed()
% BENCHMARK_TD_SPEED  Wall-clock of representative learner runs (best of 3), used as the +40% speed gate of the TDLearner
% decomposition. Baseline (pre-refactor learner) is stored in td_speed_baseline.mat.
here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(fileparts(here))));
cliff = OceanStats.RL.Env.Classic.CliffWalkingEnv();
walk = OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 19, 'LeftReward', -1, 'RightReward', 1);
cfg = {
    'qlearning onestep 300 ep',   @() OceanStats.RL.TD.TDLearner('Env', cliff, 'Target', 'qlearning', 'StepSize', 0.5, 'Exploration', 0.1, 'Episodes', 300, 'Seed', 1)
    'sarsa nstep=4 200 ep',       @() OceanStats.RL.TD.TDLearner('Env', cliff, 'Target', 'sarsa', 'Mode', 'nstep', 'N', 4, 'StepSize', 0.3, 'Exploration', 0.1, 'Episodes', 200, 'Seed', 1)
    'sarsa(lambda) 100 ep',       @() OceanStats.RL.TD.TDLearner('Env', cliff, 'Target', 'sarsa', 'Mode', 'trace', 'Lambda', 0.8, 'StepSize', 0.2, 'Exploration', 0.1, 'Episodes', 100, 'Seed', 1)
    'double q 300 ep',            @() OceanStats.RL.TD.TDLearner('Env', cliff, 'Target', 'qlearning', 'Double_learning', true, 'StepSize', 0.3, 'Exploration', 0.1, 'Episodes', 300, 'Seed', 1)
    'td0 walk19 2000 ep',         @() OceanStats.RL.TD.TDLearner('Env', walk, 'Task', 'prediction', 'StepSize', 0.1, 'Episodes', 2000, 'Seed', 1)
    'td(lambda) walk19 500 ep',   @() OceanStats.RL.TD.TDLearner('Env', walk, 'Task', 'prediction', 'Mode', 'trace', 'Lambda', 0.9, 'StepSize', 0.1, 'Episodes', 500, 'Seed', 1)
    };
t = struct('name', cases_names_(cfg), 'seconds', zeros(1, size(cfg, 1)));
for k = 1:size(cfg, 1)
    best = Inf;
    for rep = 1:3
        L = cfg{k, 2}();
        tic;  L.run();  best = min(best, toc);
    end
    t.seconds(k) = best;
    fprintf('%-30s %7.3f s\n', cfg{k, 1}, best);
end
end
function n = cases_names_(cfg), n = string(cfg(:, 1))'; end
