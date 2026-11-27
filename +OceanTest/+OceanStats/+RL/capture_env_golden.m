function capture_env_golden()
% CAPTURE_ENV_GOLDEN  Records what the ORIGINAL tabular environments (OceanStats.RL.TD.Envs.*, three-output step,
% integrated TimeLimit) did: for each configuration, the full transition model and a seeded trajectory of 400 steps under
% a fixed, RNG-free action pattern (cycling over the admissible actions), continuing across episode ends.
% The refactored OceanStats.RL.Env.Classic.* environments must reproduce these numbers exactly (EnvGoldenParityTest).
%
% HISTORICAL RECORD: this ran ONCE against the pre-refactor code. The classes it calls (OceanStats.RL.TD.Envs.*) were
% removed after the capture, so it cannot run any more; env_golden.mat is the frozen result.
% Output: env_golden.mat in this folder (checked in; it is the frozen reference).

here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(fileparts(here))));
E = 'OceanStats.RL.TD.Envs.';
cfgs = {
    'walk5',            @() OceanStats.RL.TD.Envs.RandomWalkEnv()
    'walk19',           @() OceanStats.RL.TD.Envs.RandomWalkEnv('NumInner', 19, 'LeftReward', -1, 'RightReward', 1)
    'walk4even',        @() OceanStats.RL.TD.Envs.RandomWalkEnv('NumInner', 4)
    'walk5_limit4',     @() OceanStats.RL.TD.Envs.RandomWalkEnv('TimeLimit', 4)
    'walk1_limit1',     @() OceanStats.RL.TD.Envs.RandomWalkEnv('NumInner', 1, 'TimeLimit', 1)
    'cliff',            @() OceanStats.RL.TD.Envs.CliffWalkingEnv()
    'cliff3x5',         @() OceanStats.RL.TD.Envs.CliffWalkingEnv('Rows', 3, 'Cols', 5)
    'cliff_limit20',    @() OceanStats.RL.TD.Envs.CliffWalkingEnv('TimeLimit', 20)
    'windy',            @() OceanStats.RL.TD.Envs.WindyGridworldEnv()
    'windy_king',       @() OceanStats.RL.TD.Envs.WindyGridworldEnv('KingMoves', true)
    'windy_king_stay',  @() OceanStats.RL.TD.Envs.WindyGridworldEnv('KingMoves', true, 'StayAction', true)
    'windy_stoch',      @() OceanStats.RL.TD.Envs.WindyGridworldEnv('StochasticWind', true)
    'windy_small',      @() OceanStats.RL.TD.Envs.WindyGridworldEnv('StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8])
    'maxbias',          @() OceanStats.RL.TD.Envs.MaximizationBiasEnv()
    'maxbias3',         @() OceanStats.RL.TD.Envs.MaximizationBiasEnv('NumBActions', 3, 'MeanB', 0.5, 'StdB', 2)
    };
golden = struct('name', {}, 'model', {}, 'terminal', {}, 'validmask', {}, 'traj', {});
seeds = [1 2 3];
for k = 1:size(cfgs, 1)
    env = cfgs{k, 2}();
    S = env.get_num_states();  A = env.get_num_actions();
    model = zeros(0, 5);                                    % rows: s a next prob reward
    for s = 1:S
        for a = env.get_valid_actions(s)
            [n, p, r] = env.get_transitions(s, a);
            model = [model; repmat([s a], numel(n), 1), n(:), p(:), r(:)]; %#ok<AGROW>
        end
    end
    valid = false(S, A);  term = false(S, 1);
    for s = 1:S
        term(s) = env.is_terminal(s);
        valid(s, env.get_valid_actions(s)) = true;
    end
    trajs = cell(1, numel(seeds));
    for j = 1:numel(seeds)
        env = cfgs{k, 2}();
        env.setSeed(seeds(j));
        s = env.reset();
        rows = zeros(400, 6);                               % s_before, action, r, done, truncated, s_for_next_action
        for t = 1:400
            v = env.get_valid_actions(s);
            a = v(1 + mod(t - 1, numel(v)));
            [s2, r, done] = env.step(a);
            rows(t, 1:5) = [s, a, r, done, env.was_truncated()];
            if done, s = env.reset(); else, s = s2; end
            rows(t, 6) = s;                                 % state the NEXT action is chosen in
        end
        trajs{j} = rows;
    end
    golden(k).name = cfgs{k, 1};  golden(k).model = model;  golden(k).terminal = term;
    golden(k).validmask = valid;  golden(k).traj = trajs;
end
save(fullfile(here, 'env_golden.mat'), 'golden', 'seeds');
fprintf('wrote env_golden.mat (%d configurations, %d seeds, 400 steps each)\n', numel(golden), numel(seeds));
end
