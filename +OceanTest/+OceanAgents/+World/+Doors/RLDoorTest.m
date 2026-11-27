classdef RLDoorTest < OceanTest.BaseTestCase
    % RLDOORTEST  K16 on NATIVE environments (W2 contract section 6; design section 9A.5): the RL door admits every
    % classic environment by probing RL-1..RL-8 (RL-9 assumed), and refuses one deliberately broken environment per
    % axiom with exactly that axiom's ID. Admission never depends on class names or action labels.

    properties (Constant)
        Fix = 'OceanTest.OceanAgents.Fixtures.World.RL.'
        Axioms = "RL-" + (1:9)
    end

    methods (Static, Access = private)
        function envs = natives()
            % The W2 in-scope configurations (env_golden.mat names) plus MaximizationBias, Blackjack and Racetrack.
            C = 'OceanStats.RL.Env.Classic.';
            TL = @(e, k) OceanStats.RL.Env.Wrappers.TimeLimit(e, k);
            walk = @(n) feval([C 'RandomWalkEnv'], 'NumInner', n);
            cliff = @(varargin) feval([C 'CliffWalkingEnv'], varargin{:});
            windy = @(varargin) feval([C 'WindyGridworldEnv'], varargin{:});
            envs = { ...
                "walk5", walk(5); "walk19", walk(19); "walk4even", walk(4); "walk5_limit4", TL(walk(5), 4); ...
                "walk1_limit1", TL(walk(1), 1); ...
                "cliff", cliff(); "cliff3x5", cliff('Rows', 3, 'Cols', 5); "cliff_limit20", TL(cliff(), 20); ...
                "windy", windy(); "windy_king", windy('KingMoves', true); ...
                "windy_king_stay", windy('KingMoves', true, 'StayAction', true); ...
                "windy_stoch", windy('KingMoves', true, 'StochasticWind', true); ...
                "windy_small", windy('Wind', [0 1 1 0], 'Rows', 4, 'Start', [2 1], 'Goal', [2 4]); ...
                "maxbias", feval([C 'MaximizationBiasEnv']); "blackjack", feval([C 'BlackjackEnv']); ...
                "racetrack", feval([C 'RacetrackEnv'])};
        end
    end

    methods
        function cert = admit(~, env, varargin)
            g = RandStream.getGlobalStream();
            saved = g.State;
            restore = onCleanup(@() set(g, 'State', saved));    % a broken double may draw globally (RL-7)
            cert = OceanAgents.World.Guard.GuardEngine.admitModule(env, OceanAgents.World.Doors.RLDoor.spec(), ...
                'ModuleVersion', "1", varargin{:});
            clear restore
        end

        function verifyTrueStatuses(tc, cert, label)
            tc.verifyEqual([cert.Axioms.ID], tc.Axioms, label);
            expected = [repmat("probed", 1, 8) "assumed"];
            if label == "racetrack"
                % sample-only and a random probe never finishes the track: the post-termination half of RL-5 is
                % never exercised, and the certificate says so instead of claiming "probed"
                expected(5) = "partial";
                tc.verifyEqual(cert.Axioms(5).Coverage.afterTermination, 0, label);
            end
            tc.verifyEqual([cert.Axioms.Status], expected, label);
            tc.verifyEqual([cert.Axioms.Probes], [repmat(16, 1, 8) 0], label);
            tc.verifyEmpty(cert.Axioms(9).Seeds, label);                   % RL-9 never run, never "verified"
        end
    end

    methods (Test)
        function doorSpec(tc)
            door = OceanAgents.World.Doors.RLDoor.spec();
            tc.verifyEqual(door.Admits, "OceanStats.RL.Env.IEnvironment");
            tc.verifyEqual([door.Axioms.ID], tc.Axioms);
            tc.verifyEqual([door.Axioms.Scope], repmat("module", 1, 9));
            tc.verifyEqual([door.Axioms.Mode], [repmat("probe", 1, 8) "assumed"]);
            tc.verifyEqual([door.Axioms(1:8).Probes], repmat(16, 1, 8));
            OceanAgents.World.Guard.GuardEngine.validateAssembly(door, "act", strings(1, 0));   % a valid assembly
            tc.verifyEqual([OceanAgents.World.Doors.RLDoor.spec('Probes', 4).Axioms(1:8).Probes], repmat(4, 1, 8));
            meta = OceanAgents.World.Doors.RLDoor.getMetadata();
            tc.verifyFalse(meta.IsSymbolic);
            tc.verifyEqual(OceanAgents.World.Doors.RLDoor.getPipelineSpec().Kind, "NONVIRTUAL");
        end

        function nativeEnvironmentsAdmittedK16(tc)
            envs = tc.natives();
            for k = 1:size(envs, 1)
                cert = tc.admit(envs{k, 2});
                why = strjoin(string({cert.Axioms([cert.Axioms.Status] == "failed").Detail}), ' | ');
                tc.verifyTrue(cert.Admitted, sprintf('%s refused by [%s]: %s', envs{k, 1}, strjoin(cert.failed(), ', '), why));
                tc.verifyTrueStatuses(cert, envs{k, 1});
                tc.verifyTrue(cert.validFor("1", "1"));
            end
        end

        function admittedWithAnotherProbeSeed(tc)
            envs = tc.natives();
            for k = [1 6 12 15 16]                                           % walk5, cliff, windy_stoch, blackjack, racetrack
                cert = tc.admit(envs{k, 2}, 'Seed', 12345);
                tc.verifyTrue(cert.Admitted, envs{k, 1});
            end
        end

        function eachBrokenEnvironmentRefusedWithExactlyItsID(tc)
            walk = OceanStats.RL.Env.Classic.RandomWalkEnv();
            cases = {'BrokenResetMaskEnv', "RL-1"; 'BrokenStepTypesEnv', "RL-2"; 'BrokenTerminalMaskEnv', "RL-3"; ...
                'BrokenRewardSeedEnv', "RL-4"; 'BrokenIllegalActionEnv', "RL-5"; 'GlobalRngEnv', "RL-7"; ...
                'BrokenStateTableEnv', "RL-8"; 'BrokenModelEnv', "RL-8"};
            for k = 1:size(cases, 1)
                cert = tc.admit(feval([tc.Fix cases{k, 1}]));
                tc.verifyFalse(cert.Admitted, cases{k, 1});
                tc.verifyEqual(cert.failed(), cases{k, 2}, cases{k, 1});
                tc.verifyEqual(cert.Axioms(9).Status, "assumed");
            end
            cert = tc.admit(feval([tc.Fix 'AutoResetLimitEnv'], walk, 3));   % RL-6: wrapper that auto-resets
            tc.verifyEqual(cert.failed(), "RL-6");
            tc.verifySubstring(char(cert.Axioms(6).Detail), 'truncated at step 3');
        end

        function doorBuiltWorldsAdmittedK16(tc)
            % K16 on the reverse bridge: WorldAsEnvironment over each GridWorlds factory passes the same axioms
            % (4 probes per axiom here: a world step costs ~250x a native one; the native suite runs the full 16).
            G = OceanAgents.World.Doors.GridWorlds;
            B = @(gw) OceanAgents.Env.WorldAsEnvironment(gw);
            envs = {"walk", B(G.randomWalk()); "cliff", B(G.cliffWalking()); ...
                "windy stochastic", B(G.windy('StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8])); ...
                "walk, world TimeLimit 4", B(G.randomWalk('TimeLimit', 4)); ...
                "cliff, TimeLimit wrapper 20", OceanStats.RL.Env.Wrappers.TimeLimit(B(G.cliffWalking()), 20)};
            door = OceanAgents.World.Doors.RLDoor.spec('Probes', 4);
            g = RandStream.getGlobalStream();
            before = g.State;
            for k = 1:size(envs, 1)
                cert = OceanAgents.World.Guard.GuardEngine.admitModule(envs{k, 2}, door, 'ModuleVersion', "1");
                why = strjoin(string({cert.Axioms([cert.Axioms.Status] == "failed").Detail}), ' | ');
                tc.verifyTrue(cert.Admitted, sprintf('%s refused by [%s]: %s', envs{k, 1}, strjoin(cert.failed(), ', '), why));
                tc.verifyEqual([cert.Axioms.Status], [repmat("probed", 1, 8) "assumed"], envs{k, 1});
                tc.verifyEqual([cert.Axioms.Probes], [repmat(4, 1, 8) 0], envs{k, 1});
            end
            tc.verifyEqual(g.State, before);
        end

        function namesAndLabelsDoNotMatter(tc)
            % renaming: the same dynamics under another class name get the same verdicts, probes and seeds
            a = tc.admit(OceanStats.RL.Env.Classic.RandomWalkEnv(), 'Seed', 7);
            b = tc.admit(feval([tc.Fix 'RenamedWalkEnv']), 'Seed', 7);
            tc.verifyNotEqual(b.Subject, a.Subject);
            tc.verifyEqual(rmfield(b.Axioms, 'Detail'), rmfield(a.Axioms, 'Detail'));
            tc.verifyEqual([b.Axioms.Detail], [a.Axioms.Detail]);
            % relabelling: actions labelled 0..2 (Discrete(3, 'Start', 0)) with a changing mask are admitted
            cert = tc.admit(OceanTest.OceanAgents.Fixtures.Env.LabelledCounterEnv('Horizon', 4));
            tc.verifyTrue(cert.Admitted, strjoin(cert.failed(), ', '));
            tc.verifyTrueStatuses(cert, "labelled 0..2");
        end

        function probesUseOnlyTheirStreams(tc)
            % RL predicates draw seeds and actions only from the probe stream: the same probe stream reproduces the
            % same episode, and the global stream is untouched by admitting a native environment.
            g = RandStream.getGlobalStream();
            before = g.State;
            c1 = tc.admit(OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true), 'Seed', 3);
            tc.verifyEqual(g.State, before);
            c2 = tc.admit(OceanStats.RL.Env.Classic.WindyGridworldEnv('StochasticWind', true), 'Seed', 3);
            tc.verifyEqual(c2.Axioms, c1.Axioms);
            s1 = RandStream('mrg32k3a', 'Seed', 5);
            s2 = RandStream('mrg32k3a', 'Seed', 5);
            env = OceanStats.RL.Env.Classic.CliffWalkingEnv();
            [ok1, d1] = OceanAgents.World.Doors.RLDoor.rl4(env, s1);
            [ok2, d2] = OceanAgents.World.Doors.RLDoor.rl4(env, s2);
            tc.verifyTrue(ok1 && ok2);
            tc.verifyEqual(d2, d1);
            tc.verifyEqual(s2.State, s1.State);                              % same number of draws
        end

        function nonDiscreteActionSpaceFailsRL1(tc)
            cert = tc.admit(OceanTest.OceanAgents.Fixtures.Env.BoxActionEnv());
            tc.verifyFalse(cert.Admitted);
            tc.verifySubstring(char(cert.Axioms(1).Detail), 'not Discrete');
        end
    end
end
