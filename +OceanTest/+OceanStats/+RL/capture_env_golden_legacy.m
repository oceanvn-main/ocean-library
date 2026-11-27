function capture_env_golden_legacy(oldPackageRoot)
% CAPTURE_ENV_GOLDEN_LEGACY  HISTORICAL RECORD. Recorded, from the ORIGINAL BlackjackEnv and RacetrackEnv (extracted from
% version control into a scratch package OldMC on the path), seeded trajectories under MATLAB's global stream:
%   rng(seed); reset; 500 (Blackjack) / 600 (Racetrack) steps with a fixed RNG-free action rule, resetting at episode end.
% Output env_golden_legacy.mat is the frozen reference for EnvLegacyGoldenParityTest. The original classes no longer exist
% in the tree, so this cannot be re-run without re-extracting them.
here = fileparts(mfilename('fullpath'));
addpath(oldPackageRoot);
seeds = [1 2 3];
G = struct('name', {}, 'traj', {});
cfgs = {'blackjack', @() OldMC.BlackjackEnv(), 500; 'track1', @() OldMC.RacetrackEnv("Track1"), 600; ...
        'track2', @() OldMC.RacetrackEnv("Track2"), 600; 'track1_nofail', @() OldMC.RacetrackEnv("Track1", 'FailProb', 0), 600};
for k = 1:size(cfgs, 1)
    trajs = cell(1, numel(seeds));
    for j = 1:numel(seeds)
        rng(seeds(j));
        env = cfgs{k, 2}();
        s = env.reset();
        rows = [];
        for t = 1:cfgs{k, 3}
            v = env.get_valid_actions(s);
            a = v(1 + mod(t - 1, numel(v)));
            [s2, r, done] = env.step(a);
            rows(end+1, :) = [s, a, r, done, s2]; %#ok<AGROW>
            if done, s = env.reset(); else, s = s2; end
        end
        trajs{j} = rows;
    end
    G(k).name = cfgs{k, 1};  G(k).traj = trajs;
end
save(fullfile(here, 'env_golden_legacy.mat'), 'G', 'seeds');
fprintf('wrote env_golden_legacy.mat\n');
end
