classdef GridWorldsModelTest < OceanTest.BaseTestCase
    % GRIDWORLDSMODELTEST  OceanAgents.World.Doors.GridWorlds factories (W2_Grid_Doors_Contract.md 3.3, 7 G1):
    % for every in-scope configuration of section 1, the factory's exact model (Fn), terminal mask and valid mask
    % equal the native TabularEnvironment tables entry by entry (bit for bit), the Agency configuration is well formed,
    % and - once the W2 kernel additions exist - the doors driven by a World reproduce native episodes step by step.

    properties (TestParameter)
        config = OceanTest.OceanAgents.World.Doors.GridWorldsModelTest.configurations()
    end

    methods (Static)
        function c = configurations()
            % name -> {factory, native builder, time limit}; the 13 in-scope configurations of section 1.
            G = @OceanAgents.World.Doors.GridWorlds.randomWalk;
            C = @OceanAgents.World.Doors.GridWorlds.cliffWalking;
            Wf = @OceanAgents.World.Doors.GridWorlds.windy;
            RW = @(varargin) OceanStats.RL.Env.Classic.RandomWalkEnv(varargin{:});
            CW = @(varargin) OceanStats.RL.Env.Classic.CliffWalkingEnv(varargin{:});
            WW = @(varargin) OceanStats.RL.Env.Classic.WindyGridworldEnv(varargin{:});
            c = struct( ...
                'walk5',           {{@() G(), @() RW(), Inf}}, ...
                'walk19',          {{@() G('NumInner', 19, 'LeftReward', -1, 'RightReward', 1), ...
                                     @() RW('NumInner', 19, 'LeftReward', -1, 'RightReward', 1), Inf}}, ...
                'walk4even',       {{@() G('NumInner', 4), @() RW('NumInner', 4), Inf}}, ...
                'walk5_limit4',    {{@() G('TimeLimit', 4), @() RW(), 4}}, ...
                'walk1_limit1',    {{@() G('NumInner', 1, 'TimeLimit', 1), @() RW('NumInner', 1), 1}}, ...
                'cliff',           {{@() C(), @() CW(), Inf}}, ...
                'cliff3x5',        {{@() C('Rows', 3, 'Cols', 5), @() CW('Rows', 3, 'Cols', 5), Inf}}, ...
                'cliff_limit20',   {{@() C('TimeLimit', 20), @() CW(), 20}}, ...
                'windy',           {{@() Wf(), @() WW(), Inf}}, ...
                'windy_king',      {{@() Wf('KingMoves', true), @() WW('KingMoves', true), Inf}}, ...
                'windy_king_stay', {{@() Wf('KingMoves', true, 'StayAction', true), ...
                                     @() WW('KingMoves', true, 'StayAction', true), Inf}}, ...
                'windy_stoch',     {{@() Wf('StochasticWind', true), @() WW('StochasticWind', true), Inf}}, ...
                'windy_small',     {{@() Wf('StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8]), ...
                                     @() WW('StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8]), Inf}});
        end
    end

    methods (Test)
        function modelEqualsNative(tc, config)
            gw = config{1}();
            env = config{2}();
            S = env.NumStates;  A = env.NumActions;
            m = gw.Model;
            tc.verifyEqual(m.States, (1:S)');
            tc.verifyEqual(m.Terminal, env.TerminalMask);
            tc.verifyEqual(m.ValidMask, env.ValidMask);
            tc.verifyEqual(gw.ActionLabels, 1:A);
            for s = 1:S
                for a = 1:A
                    if env.ValidMask(s, a)
                        [n1, p1, r1] = env.get_transitions(s, a);
                        [n2, p2, r2] = m.Fn(s, a);
                        tc.assertTrue(isequal(n1, n2) && isequal(p1, p2) && isequal(r1, r2), ...
                            sprintf('Fn(%d, %d): native [%s | %s | %s], door [%s | %s | %s]', s, a, ...
                            mat2str(n1'), mat2str(p1'), mat2str(r1'), mat2str(n2'), mat2str(p2'), mat2str(r2')));
                    else
                        tc.verifyError(@() m.Fn(s, a), 'OceanStats:RL:Env:TabularEnvironment:InvalidAction');
                    end
                end
            end
            % start cell and door configuration agree with the native start distribution
            g = gw.GridMotion;
            tc.verifyEqual(find(env.StartProbs), (g.Start(1) - 1) * g.Cols + g.Start(2));
            tc.verifyEqual(gw.RewardTermination.TimeLimit, config{3});
        end

        function agencyConfiguration(tc, config)
            gw = config{1}();
            a = gw.Agency;
            A = numel(gw.ActionLabels);
            tc.verifyEqual(a.ActionLabels, 1:A);
            tc.verifyEqual([a.Routing.Label], 1:A);
            tc.verifyEqual(unique([a.Routing.Type]), "Move");
            tc.verifyEqual(arrayfun(@(r) r.Payload.Label, a.Routing), 1:A);
            tc.verifyEqual(a.ActionSpace.N, A);
            tc.verifyEqual(a.ObservationSpace.N, numel(gw.Model.States));
            tc.verifyEqual(a.Observe(struct('GridMotion_Cell', struct('Index', 17))), 17);
            tc.verifyEqual([gw.Doors.Name], [OceanAgents.World.Doors.Agency.Name "GridMotion" "RewardTermination"]);
        end

        function doorsReproduceNativeEpisodes(tc, config)
            % World-driven G1: the same seed, the same actions -> the same observations, rewards and endings as the
            % native environment (time limits: the native TimeLimit wrapper against the world's RewardTermination).
            tc.assumeTrue(OceanTest.OceanAgents.World.Doors.GridMotionTest.kernelReady(), ...
                'needs Agency.m and the legacy stream / outbox in World (B1, W2 phase 1)');
            gw = config{1}();
            native = config{2}();
            if isfinite(config{3}), native = OceanStats.RL.Env.Wrappers.TimeLimit(native, config{3}); end
            w = OceanAgents.World.World(gw.Doors, 'SkipIdle', false, 'Injectable', "Move", ...
                'Outbox', ["Reward" "Terminated" "Truncated"]);
            for seed = [1 7 42]
                w.reset(seed);
                [obs, info] = native.reset('Seed', seed);
                id = w.Registry.idsOf(w.Store.rowsWith("Position"));
                v = w.project(id);
                tc.assertEqual(v.GridMotion_Cell.Index, obs, sprintf('seed %d reset', seed));
                for t = 1:200
                    valid = find(info.action_mask);
                    a = valid(1 + mod(t * 5 + seed, numel(valid)));
                    [obs, r, term, trunc, info] = native.step(a);
                    w.inject(OceanAgents.World.Event.make("Move", struct('Label', a), 'SourceEntity', id));
                    w.advance();
                    ev = w.takeOutbox();
                    types = [ev.Type];
                    v = w.project(id);
                    tc.assertEqual(v.GridMotion_Cell.Index, obs, sprintf('seed %d step %d', seed, t));
                    tc.assertEqual(ev(types == "Reward").Payload.Amount, r, sprintf('seed %d step %d reward', seed, t));
                    tc.assertEqual(any(types == "Terminated"), logical(term));
                    tc.assertEqual(any(types == "Truncated"), logical(trunc));
                    if term || trunc, break; end
                end
            end
        end
    end
end
