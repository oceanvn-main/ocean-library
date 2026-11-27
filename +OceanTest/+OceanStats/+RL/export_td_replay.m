function export_td_replay(outFile)
% EXPORT_TD_REPLAY  Runs a battery of TDLearner configurations with Record=true and writes everything an
% independent implementation needs to replay them: the configuration, the admissibility mask, every
% recorded episode (states, actions, rewards, termination/truncation, double-learning coin flips) and the
% learner's FINAL tables. validate_td_replay.py then re-implements each update rule from the textbook
% definitions (NOT from this code), replays the same transitions and compares the tables.
%
% Output: td_replay_cases.json in this folder (or the file given as argument). Run:  OceanTest.OceanStats.RL.export_td_replay()

here = fileparts(mfilename('fullpath'));
if nargin < 1, outFile = fullfile(here, 'td_replay_cases.json'); end
addpath(fileparts(fileparts(fileparts(here))));
import OceanStats.RL.TD.*

cliff = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
walk = OceanStats.RL.Env.Classic.RandomWalkEnv();
walkCut = OceanStats.RL.Env.Wrappers.TimeLimit(OceanStats.RL.Env.Classic.RandomWalkEnv(), 4);                  % exercises truncation (bootstrap kept)
maxb = OceanStats.RL.Env.Classic.MaximizationBiasEnv();
windy = OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8]);

cases = struct('name', {}, 'cfg', {});
    function add(name, envName, env, varargin)
        cases(end+1) = struct('name', name, 'cfg', struct('env', envName, 'opts', {varargin}, 'envObj', env)); %#ok<AGROW>
    end
% ---- one-step control
add('sarsa const', 'cliff', cliff, 'Target', 'sarsa', 'StepSize', 0.3, 'Gamma', 0.9, 'Exploration', 0.2, 'Episodes', 60, 'Seed', 1);
add('expected sarsa const', 'cliff', cliff, 'Target', 'expectedsarsa', 'StepSize', 0.3, 'Gamma', 0.9, 'Exploration', 0.2, 'Episodes', 60, 'Seed', 2);
add('q-learning const', 'cliff', cliff, 'Target', 'qlearning', 'StepSize', 0.3, 'Gamma', 0.9, 'Exploration', 0.2, 'Episodes', 60, 'Seed', 3);
add('q-learning harmonic', 'cliff', cliff, 'Target', 'qlearning', 'StepSize', StepSize.HarmonicStepSize('Power', 0.7, 'Scale', 0.9), 'Exploration', 0.3, 'Episodes', 60, 'Seed', 4);
add('sarsa stochastic wind', 'windy', windy, 'Target', 'sarsa', 'StepSize', 0.4, 'Gamma', 0.95, 'Exploration', 0.1, 'Episodes', 40, 'Seed', 5);
add('double q-learning', 'maxbias', maxb, 'Target', 'qlearning', 'Double', true, 'StepSize', 0.2, 'Exploration', 0.2, 'Episodes', 120, 'Seed', 6);
% ---- prediction
add('td0 harmonic', 'walk', walk, 'Task', 'prediction', 'StepSize', StepSize.HarmonicStepSize('Power', 0.6), 'Episodes', 80, 'Seed', 7);
add('td0 const gamma', 'walk', walk, 'Task', 'prediction', 'StepSize', 0.1, 'Gamma', 0.9, 'Episodes', 80, 'Seed', 8);
add('td0 truncated', 'walkCut', walkCut, 'Task', 'prediction', 'StepSize', 0.2, 'Episodes', 80, 'Seed', 9, 'InitialValue', 0.3);
add('n-step td n=3', 'walk', walk, 'Task', 'prediction', 'Mode', 'nstep', 'N', 3, 'StepSize', 0.1, 'Gamma', 0.95, 'Episodes', 80, 'Seed', 10);
add('n-step td n=4 truncated', 'walkCut', walkCut, 'Task', 'prediction', 'Mode', 'nstep', 'N', 4, 'StepSize', 0.1, 'Episodes', 80, 'Seed', 11, 'InitialValue', 0.2);
add('td(lambda) accumulating', 'walk', walk, 'Task', 'prediction', 'Mode', 'trace', 'Lambda', 0.8, 'StepSize', 0.05, 'Gamma', 0.95, 'Episodes', 80, 'Seed', 12);
add('td(lambda) replacing', 'walk', walk, 'Task', 'prediction', 'Mode', 'trace', 'Lambda', 0.8, 'Trace', 'replacing', 'StepSize', 0.05, 'Episodes', 80, 'Seed', 13);
add('td(lambda) offline', 'walk', walk, 'Task', 'prediction', 'Mode', 'trace', 'Offline', true, 'Lambda', 0.6, 'StepSize', 0.1, 'Episodes', 80, 'Seed', 14);
add('td(lambda) truncated', 'walkCut', walkCut, 'Task', 'prediction', 'Mode', 'trace', 'Lambda', 0.7, 'StepSize', 0.1, 'Episodes', 80, 'Seed', 15, 'InitialValue', 0.4);
add('lambda-return', 'walk', walk, 'Task', 'prediction', 'Mode', 'lambdareturn', 'Lambda', 0.7, 'StepSize', 0.1, 'Gamma', 0.95, 'Episodes', 80, 'Seed', 16);
add('lambda-return truncated', 'walkCut', walkCut, 'Task', 'prediction', 'Mode', 'lambdareturn', 'Lambda', 0.5, 'StepSize', 0.1, 'Episodes', 80, 'Seed', 17, 'InitialValue', 0.4);
% ---- n-step / trace control
add('n-step sarsa n=4', 'cliff', cliff, 'Target', 'sarsa', 'Mode', 'nstep', 'N', 4, 'StepSize', 0.2, 'Gamma', 0.95, 'Exploration', 0.2, 'Episodes', 50, 'Seed', 18);
add('sarsa(lambda) replacing', 'cliff', cliff, 'Target', 'sarsa', 'Mode', 'trace', 'Lambda', 0.8, 'Trace', 'replacing', 'StepSize', 0.1, 'Gamma', 0.95, 'Exploration', 0.2, 'Episodes', 50, 'Seed', 19);
add('watkins q(lambda)', 'cliff', cliff, 'Target', 'qlearning', 'Mode', 'trace', 'Lambda', 0.7, 'StepSize', 0.1, 'Gamma', 0.95, 'Exploration', 0.3, 'Episodes', 50, 'Seed', 20);
add('n-step expected sarsa n=3', 'cliff', cliff, 'Target', 'expectedsarsa', 'Mode', 'nstep', 'N', 3, 'StepSize', 0.2, 'Gamma', 0.95, 'Exploration', 0.2, 'Episodes', 50, 'Seed', 22);
add('expected sarsa(lambda)', 'cliff', cliff, 'Target', 'expectedsarsa', 'Mode', 'trace', 'Lambda', 0.6, 'Trace', 'replacing', 'StepSize', 0.1, 'Gamma', 0.95, 'Exploration', 0.2, 'Episodes', 50, 'Seed', 23);
add('sarsa maxsteps truncation', 'cliff', cliff, 'Target', 'sarsa', 'StepSize', 0.3, 'Gamma', 0.9, 'Exploration', 0.3, 'Episodes', 60, 'MaxSteps', 7, 'InitialValue', 1, 'Seed', 24);
add('n-step sarsa maxsteps', 'cliff', cliff, 'Target', 'sarsa', 'Mode', 'nstep', 'N', 3, 'StepSize', 0.3, 'Gamma', 0.9, 'Exploration', 0.3, 'Episodes', 60, 'MaxSteps', 5, 'InitialValue', 1, 'Seed', 25);
add('sarsa(lambda) maxsteps', 'cliff', cliff, 'Target', 'sarsa', 'Mode', 'trace', 'Lambda', 0.7, 'StepSize', 0.3, 'Gamma', 0.9, 'Exploration', 0.3, 'Episodes', 60, 'MaxSteps', 6, 'InitialValue', 1, 'Seed', 26);
add('double q harmonic', 'maxbias', maxb, 'Target', 'qlearning', 'Double', true, 'StepSize', StepSize.HarmonicStepSize('Power', 0.7, 'Scale', 0.9), 'Exploration', 0.2, 'Episodes', 150, 'Seed', 27);
add('sarsa(lambda) accumulating', 'maxbias', maxb, 'Target', 'sarsa', 'Mode', 'trace', 'Lambda', 0.9, 'StepSize', 0.1, 'Exploration', 0.2, 'Episodes', 60, 'Seed', 21);

out = cell(1, numel(cases));
for k = 1:numel(cases)
    c = cases(k).cfg;
    opts = c.opts;
    L = TDLearner('Env', c.envObj, opts{:}, 'Record', true);
    [V, Q, pol, h] = L.run();
    ep = h.Episodes;
    eps = cellfun(@(e) struct('S', e.S(:)', 'A', e.A(:)', 'R', e.R(:)', 'Coin', e.Coin(:)', ...
        'Terminated', logical(e.Terminated), 'Truncated', logical(e.Truncated)), ep, 'UniformOutput', false);
    base = c.envObj.unwrapped();  S = base.NumStates;  A = base.NumActions;
    valid = false(S, A);
    for s = 1:S, for a = base.get_valid_actions(s), valid(s, a) = true; end, end
    out{k} = struct('name', cases(k).name, 'env', c.env, 'cfg', cfgToStruct_(opts), 'valid', double(valid), ...
        'episodes', {eps}, 'V', V, 'Q', Q, 'pi', pol, 'returns', h.EpisodeReturns, 'lengths', h.EpisodeLengths, 'totalSteps', h.TotalSteps);
end
fid = fopen(outFile, 'w');
fwrite(fid, jsonencode(out), 'char');
fclose(fid);
fprintf('wrote td_replay_cases.json with %d cases\n', numel(out));
end

function s = cfgToStruct_(opts)
% flatten name-value options into a plain struct (strategy objects become {kind, params})
s = struct('Task', "control", 'Mode', "onestep", 'Target', "", 'N', 1, 'Lambda', 0.9, 'Gamma', 1, 'Double', false, ...
    'Offline', false, 'Trace', "accumulating", 'InitialValue', 0, 'Exploration', 0.1, ...
    'StepSize', struct('kind', "constant", 'alpha', 0.1, 'power', 1, 'scale', 1));
for i = 1:2:numel(opts)
    k = opts{i};  v = opts{i + 1};
    switch k
        case 'StepSize'
            if isnumeric(v)
                s.StepSize = struct('kind', "constant", 'alpha', v, 'power', 1, 'scale', 1);
            else
                s.StepSize = struct('kind', "harmonic", 'alpha', 0, 'power', v.Power, 'scale', v.Scale);
            end
        case {'Seed', 'Episodes'}
            s.(k) = v;
        otherwise
            s.(k) = v;
    end
end
if strcmp(s.Task, "prediction") && s.Target == "", s.Target = "td0"; end
if s.Target == "", s.Target = "qlearning"; end
end
