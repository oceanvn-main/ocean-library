classdef WorldSolverParityTest < OceanTest.BaseTestCase
    % WORLDSOLVERPARITYTEST  L2c (W2 contract section 7): the existing solvers give the same results on the
    % door-built worlds (through OceanAgents.Env.WorldAsEnvironment) as on the native environments.
    %   (a) ValueIterationSolver on the bridge equals the native result exactly (values, policy, sweep history);
    %       PolicyIterationSolver (fixed 2026-10-01) gives the same values, policy and history on the bridge as on
    %       the native environment, and agrees with value iteration on the optimal values; the converged action values
    %       (the solver's compute_q_value over every valid (s, a)) are compared as well.
    %   (b) every td_replay_golden.json case on cliff, walk, walkCut and windy (24 of 27): TDLearner is built from the
    %       case's recorded cfg, run on the door environment (walkCut = TimeLimit(bridge(randomWalk), 4)), and the
    %       JSON of every recorded field (valid mask, each episode's S/A/R/Coin/Terminated/Truncated, V, Q, pi,
    %       returns, lengths, totalSteps) must be BYTE-IDENTICAL to the frozen golden text of that case.
    %       The same harness is first run on the NATIVE environment, which proves the cfg reconstruction is exact.
    % The golden file was produced by the pre-refactor learner and is only read here.

    methods (Static)
        function env = doorEnv(name)
            G = @(f, varargin) feval(['OceanAgents.Devices.DeviceWorld.' f], varargin{:});   % S1: device worlds
            B = @(env) env;
            switch string(name)
                case "cliff", env = B(G('cliffWalking', 'Rows', 3, 'Cols', 5));
                case "walk", env = B(G('randomWalk'));
                case "walkCut", env = OceanStats.RL.Env.Wrappers.TimeLimit(B(G('randomWalk')), 4);
                case "windy", env = B(G('windy', 'StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8]));
            end
        end

        function env = nativeEnv(name)
            % Exactly the environments of export_td_replay.m
            switch string(name)
                case "cliff", env = OceanStats.RL.Env.Classic.CliffWalkingEnv('Rows', 3, 'Cols', 5);
                case "walk", env = OceanStats.RL.Env.Classic.RandomWalkEnv();
                case "walkCut", env = OceanStats.RL.Env.Wrappers.TimeLimit(OceanStats.RL.Env.Classic.RandomWalkEnv(), 4);
                case "windy", env = OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true, 'Rows', 5, ...
                        'Start', [3 1], 'Goal', [3 8]);
            end
        end

        function [cases, text] = goldenCases()
            % Each case: name, env, cfg (decoded) and Tail = its exact golden text from ,"valid": to the closing brace.
            here = fileparts(which('OceanTest.OceanStats.RL.TDRefactorGoldenTest'));
            text = fileread(fullfile(here, 'td_replay_golden.json'));
            starts = strfind(text, '{"name":');
            ends = [starts(2:end) - 2, numel(text) - 1];            % drop the separating comma / the final bracket
            cases = struct('name', {}, 'env', {}, 'cfg', {}, 'Tail', {});
            for k = 1:numel(starts)
                seg = text(starts(k):ends(k));
                d = jsondecode(seg);
                at = strfind(seg, ',"valid":');
                cases(end + 1) = struct('name', string(d.name), 'env', string(d.env), 'cfg', d.cfg, ...
                    'Tail', seg(at(1):end)); %#ok<AGROW>
            end
        end

        function opts = optionsOf(cfg)
            % TDLearner options from the recorded cfg (export_td_replay.cfgToStruct_ stores every option, with the
            % learner's own defaults for the ones a case did not set).
            if string(cfg.StepSize.kind) == "constant"
                step = cfg.StepSize.alpha;
            else
                step = OceanStats.RL.TD.StepSize.HarmonicStepSize('Power', cfg.StepSize.power, 'Scale', cfg.StepSize.scale);
            end
            opts = {'Task', string(cfg.Task), 'Mode', char(cfg.Mode), 'Target', char(cfg.Target), 'N', cfg.N, ...
                'Lambda', cfg.Lambda, 'Gamma', cfg.Gamma, 'Offline', cfg.Offline, 'Trace', char(cfg.Trace), ...
                'InitialValue', cfg.InitialValue, 'Exploration', cfg.Exploration, 'StepSize', step, ...
                'Episodes', cfg.Episodes, 'Seed', cfg.Seed};
            if isfield(cfg, 'MaxSteps'), opts = [opts, {'MaxSteps', cfg.MaxSteps}]; end
        end

        function id = errorOf(fn)
            id = "";
            try
                fn();
            catch err
                id = string(err.identifier);
            end
        end

        function tail = runCase(env, cfg)
            % The recorded fields of one run, encoded exactly as export_td_replay.m encodes them.
            opts = OceanTest.OceanAgents.Env.WorldSolverParityTest.optionsOf(cfg);
            L = OceanStats.RL.TD.TDLearner('Env', env, opts{:}, 'Record', true);
            [V, Q, pol, h] = L.run();
            eps = cellfun(@(e) struct('S', e.S(:)', 'A', e.A(:)', 'R', e.R(:)', 'Coin', e.Coin(:)', ...
                'Terminated', logical(e.Terminated), 'Truncated', logical(e.Truncated)), h.Episodes, 'UniformOutput', false);
            base = env.unwrapped();  S = base.NumStates;  A = base.NumActions;
            valid = false(S, A);
            for s = 1:S, for a = base.get_valid_actions(s), valid(s, a) = true; end, end
            out = struct('valid', double(valid), 'episodes', {eps}, 'V', V, 'Q', Q, 'pi', pol, ...
                'returns', h.EpisodeReturns, 'lengths', h.EpisodeLengths, 'totalSteps', h.TotalSteps);
            tail = jsonencode(out);
            tail = [',' tail(2:end)];                                % {"valid":...}  ->  ,"valid":...}
        end
    end

    methods (Test)
        function exactSolversAgreeOnTheBridgeL2c(tc)
            C = 'OceanStats.RL.Env.Classic.';
            pairs = { ...
                "cliff", @() feval([C 'CliffWalkingEnv']), @() OceanAgents.Devices.DeviceWorld.cliffWalking(); ...
                "cliff3x5", @() feval([C 'CliffWalkingEnv'], 'Rows', 3, 'Cols', 5), @() OceanAgents.Devices.DeviceWorld.cliffWalking('Rows', 3, 'Cols', 5); ...
                "walk19", @() feval([C 'RandomWalkEnv'], 'NumInner', 19, 'LeftReward', -1, 'RightReward', 1), ...
                          @() OceanAgents.Devices.DeviceWorld.randomWalk('NumInner', 19, 'LeftReward', -1, 'RightReward', 1); ...
                "windy_king", @() feval([C 'WindyGridworldEnv'], 'KingMoves', true), @() OceanAgents.Devices.DeviceWorld.windy('KingMoves', true); ...
                "windy_small", @() feval([C 'WindyGridworldEnv'], 'StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8]), ...
                               @() OceanAgents.Devices.DeviceWorld.windy('StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8])};
            for k = 1:size(pairs, 1)
                name = pairs{k, 1};
                native = pairs{k, 2}();
                bridge = pairs{k, 3}();
                for gamma = [0.9 1]
                    sn = OceanStats.RL.DP.ValueIterationSolver(native, 'DiscountFactor', gamma);
                    sb = OceanStats.RL.DP.ValueIterationSolver(bridge, 'DiscountFactor', gamma);
                    [Vn, pn, hn] = sn.run();
                    [Vb, pb, hb] = sb.run();
                    tc.verifyEqual(Vb, Vn, sprintf('VI values %s gamma %g', name, gamma));
                    tc.verifyEqual(pb, pn, sprintf('VI policy %s gamma %g', name, gamma));
                    tc.verifyEqual(hb, hn, sprintf('VI history %s gamma %g', name, gamma));
                    % the converged action values (one Bellman backup from each model) agree for every valid (s, a)
                    [ss, aa] = find(native.ValidMask);
                    qn = arrayfun(@(s, a) sn.compute_q_value(s, a), ss, aa);
                    qb = arrayfun(@(s, a) sb.compute_q_value(s, a), ss, aa);
                    tc.verifyEqual(qb, qn, sprintf('Q backups %s gamma %g', name, gamma));
                end
                % PolicyIterationSolver (fixed 2026-10-01: legal starting actions, terminal states worth 0 and never
                % evaluated) now solves the grid worlds: the bridge gives exactly the native result, and policy
                % iteration agrees with value iteration on the optimal values
                pn = OceanStats.RL.DP.PolicyIterationSolver(native, 'DiscountFactor', 0.9, 'Theta', 1e-10);
                pb = OceanStats.RL.DP.PolicyIterationSolver(bridge, 'DiscountFactor', 0.9, 'Theta', 1e-10);
                [Vpn, pipn, hpn] = pn.run();
                [Vpb, pipb, hpb] = pb.run();
                tc.verifyEqual(Vpb, Vpn, sprintf('PI values %s', name));
                tc.verifyEqual(pipb, pipn, sprintf('PI policy %s', name));
                tc.verifyEqual(hpb, hpn, sprintf('PI history %s', name));
                vi = OceanStats.RL.DP.ValueIterationSolver(native, 'DiscountFactor', 0.9, 'Theta', 1e-12);
                Vvi = vi.run();
                tc.verifyEqual(Vpn(:), Vvi(:), 'AbsTol', 1e-6, sprintf('PI vs VI optimal values %s', name));
            end
            % a wrapped bridge is refused by the exact solvers exactly like a wrapped native environment
            wrapped = OceanStats.RL.Env.Wrappers.TimeLimit(OceanAgents.Devices.DeviceWorld.randomWalk(), 4);
            tc.verifyError(@() OceanStats.RL.DP.ValueIterationSolver(wrapped), ...
                'OceanStats:RL:Compat:LegacyEnvironment:WrapperChangesModel');
        end

        function tdReplayCasesAreByteIdenticalL2c(tc)
            [cases, ~] = tc.goldenCases();
            tc.assertEqual(numel(cases), 27);
            inScope = cases(ismember([cases.env], ["cliff" "walk" "walkCut" "windy"]));
            tc.assertEqual(numel(inScope), 24);
            tc.verifyEqual(unique([cases(~ismember([cases.env], [inScope.env])).env]), "maxbias");
            for c = inScope
                native = tc.runCase(tc.nativeEnv(c.env), c.cfg);
                tc.assertTrue(strcmp(native, c.Tail), sprintf('harness check: native run of "%s" differs from the golden', c.name));
                door = tc.runCase(tc.doorEnv(c.env), c.cfg);
                tc.verifyTrue(strcmp(door, c.Tail), sprintf('door world differs from the golden in case "%s" (%s)', c.name, c.env));
            end
        end
    end
end
