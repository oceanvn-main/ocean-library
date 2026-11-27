function compare_td_envs_in_matlab()
% COMPARE_TD_ENVS_IN_MATLAB  Cliff Walking environment parity against Gymnasium's CliffWalking-v1 transition table
% (gym_cliff_reference.json from gym_cliff_reference.py): next state, reward and terminated flag for EVERY
% (state, action) pair of the 4 x 12 grid (state s0 = row*12 + col 0-based; actions 0-3 = up, right, down, left).
% Cliff cells are never occupied, so only the 46 live states + the goal's neighbours are compared (the goal and the
% cliff row interior have no admissible actions in our model).
% Run:  OceanTest.OceanStats.RL.compare_td_envs_in_matlab()
here = fileparts(mfilename('fullpath'));
addpath(fileparts(fileparts(fileparts(here))));
ref = jsondecode(fileread(fullfile(here, 'gym_cliff_reference.json')));
env = OceanStats.RL.Env.Classic.CliffWalkingEnv();
nCmp = 0; bad = 0; first = "";
for s0 = 0:ref.nS - 1
    s = s0 + 1;
    if env.is_terminal(s), continue; end
    for a0 = 0:3
        [nx, ~, r] = env.get_transitions(s, a0 + 1);
        gymNext = ref.next(s0 + 1, a0 + 1);  gymRew = ref.reward(s0 + 1, a0 + 1);  gymTerm = ref.terminated(s0 + 1, a0 + 1) ~= 0;
        nCmp = nCmp + 1;
        ok = (nx - 1 == gymNext) && (r == gymRew) && (env.is_terminal(nx) == gymTerm);
        if ~ok
            bad = bad + 1;
            if first == "", first = sprintf('s0=%d a0=%d: ours next=%d r=%g term=%d, gym next=%d r=%g term=%d', s0, a0, nx - 1, r, env.is_terminal(nx), gymNext, gymRew, gymTerm); end
        end
    end
end
if bad == 0, st = 'PASS'; else, st = 'FAIL'; end
fprintf('[%s] CliffWalking vs Gymnasium CliffWalking-v1: %d (state, action) pairs compared, %d mismatches\n', st, nCmp, bad);
if bad > 0, fprintf('      first: %s\n', first); end
end
