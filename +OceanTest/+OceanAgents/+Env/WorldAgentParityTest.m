classdef WorldAgentParityTest < OceanTest.BaseTestCase
    % WORLDAGENTPARITYTEST  L3 (W2 contract section 7): the A1 agent stack cannot tell a door-built world from the
    % native environment.
    %   1. SingleAgentRunner: epsilon-greedy + OneStepQLearner over NativeSingleAgentBridge(native) and over
    %      NativeSingleAgentBridge(WorldAsEnvironment(factory)), same seeds -> identical EpisodeReturns,
    %      EpisodeLengths, step counts and Q tables (bit for bit).
    %   2. JointRunner over the world's own agent face (AgentWorld, one agent): the run completes, and every episode's
    %      observations, rewards, endings and return equal an independent replay of the agent's recorded actions on
    %      the NATIVE environment with the same per-episode seeds.

    properties (Constant)
        Alpha = 0.5
        Gamma = 1
        Epsilon = 0.1
    end

    methods (Static)
        function c = configs()
            % name, native builder, factory builder, time limit (wrapper), episodes
            C = 'OceanStats.RL.Env.Classic.';
            c = { ...
                "cliff", @() feval([C 'CliffWalkingEnv']), @(varargin) OceanAgents.Devices.DeviceWorld.cliffWalking(varargin{:}), 60, 12; ...
                "walk5", @() feval([C 'RandomWalkEnv']), @(varargin) OceanAgents.Devices.DeviceWorld.randomWalk(varargin{:}), 4, 30; ...
                "windy_small", @() feval([C 'WindyGridworldEnv'], 'StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8]), ...
                    @(varargin) OceanAgents.Devices.DeviceWorld.windy('StochasticWind', true, 'Rows', 5, 'Start', [3 1], 'Goal', [3 8], varargin{:}), 40, 12};
        end

        function [host, q, rec] = agent(validMask)
            C = OceanTest.OceanAgents.Env.WorldAgentParityTest;
            [S, A] = size(validMask);
            codec = OceanAgents.Artifacts.DiscreteStateCodec(S);
            q = OceanStats.RL.TD.Values.TabularQ(validMask, 0);
            pol = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:A, 'Epsilon', C.Epsilon, 'StateCodec', codec);
            lrn = OceanAgents.Learners.OneStepQLearner(q, 1:A, 'Alpha', C.Alpha, 'Gamma', C.Gamma, 'StateCodec', codec);
            rec = OceanTest.OceanAgents.Fixtures.Integration.RecordingLearner(lrn);
            host = OceanAgents.Runtime.AgentHost("agent1", OceanAgents.Minds.PolicyMind(pol), 'Learner', rec, ...
                'Gamma', C.Gamma);
        end
    end

    methods (Test)
        function singleAgentRunnerIdenticalOnNativeAndWorld(tc)
            cfgs = tc.configs();
            for k = 1:size(cfgs, 1)
                [name, mkNative, mkWorld, limit, episodes] = cfgs{k, :};
                TL = @(env) OceanStats.RL.Env.Wrappers.TimeLimit(env, limit);
                native = mkNative();
                envs = {TL(native), TL(mkWorld()), mkWorld('TimeLimit', limit)};   % device worlds (S1): wrapper and world-native limit   % wrapper and world-native limit
                labels = ["native" "world + TimeLimit wrapper" "world TimeLimit"];
                results = cell(1, 3);
                for j = 1:3
                    [host, q, rec] = tc.agent(native.ValidMask);
                    runner = OceanAgents.Runtime.SingleAgentRunner(OceanAgents.Env.NativeSingleAgentBridge(envs{j}), ...
                        host, 'Seed', 11 + k);
                    r = runner.run(episodes);
                    tc.assertEqual(r.Status, "Completed", sprintf('%s on %s: %s', name, labels(j), r.FailureStage));
                    results{j} = struct('Returns', r.EpisodeReturns, 'Lengths', r.EpisodeLengths, 'Steps', r.StepCalls, ...
                        'Updates', r.UpdateCounts, 'Q', q.Q, 'Counts', q.Counts, ...
                        'Actions', cellfun(@(t) t.Action, rec.Seen), 'Obs', cellfun(@(t) t.NextObservation, rec.Seen));
                end
                for j = 2:3
                    tc.verifyTrue(isequal(results{j}, results{1}), sprintf('%s: %s differs from native', name, labels(j)));
                end
                tc.verifyGreaterThan(results{1}.Steps, episodes);      % a real run, not a trivial one
            end
        end

        function jointRunnerOverAgentWorldMatchesNativeReplay(tc)
            cfgs = tc.configs();
            for k = 1:size(cfgs, 1)
                [name, mkNative, mkWorld, limit, episodes] = cfgs{k, :};
                runSeed = 40 + k;
                % AgentWorld is the door kernel's joint face; it still needs the door-built definitions until S2
                % (System: several bodies) rebuilds joint worlds on devices
                doorWorld = func2str(mkWorld);
                doorWorld = str2func(strrep(doorWorld, 'OceanAgents.Devices.DeviceWorld.', 'OceanAgents.World.Doors.GridWorlds.'));
                gw = doorWorld('TimeLimit', limit);
                aw = OceanAgents.World.AgentWorld(gw.Doors, gw.Agency, 'Name', name);
                [host, ~, rec] = tc.agent(gw.Model.ValidMask);
                r = OceanAgents.Runtime.JointRunner(aw, {host}, 'Seed', runSeed).run(episodes);
                tc.assertEqual(r.Status, "Completed", sprintf('%s: %s', name, r.FailureStage));
                tc.verifyEqual(r.NumEpisodes, episodes);
                tc.verifyEqual(aw.NumAdvances, r.StepCalls);           % exactly one World.advance per joint step
                % independent reference: the same per-episode seeds and the agent's recorded actions on the native env
                reg = OceanAgents.Runtime.StreamRegistry(runSeed);
                native = OceanStats.RL.Env.Wrappers.TimeLimit(mkNative(), limit);
                seen = rec.Seen;
                at = 0;
                returns = zeros(episodes, 1);
                for e = 1:episodes
                    obs = native.reset('Seed', reg.nextSeed("env"));
                    done = false;
                    while ~done
                        at = at + 1;
                        t = seen{at};
                        tc.assertEqual(t.Observation, obs, sprintf('%s episode %d: observation before step %d', name, e, at));
                        [obs, reward, term, trunc] = native.step(t.Action);
                        tc.assertEqual([t.NextObservation t.Reward t.Terminated t.Truncated], ...
                            [obs reward term trunc], sprintf('%s episode %d step %d', name, e, at));
                        returns(e) = returns(e) + reward;
                        done = term || trunc;
                    end
                end
                tc.verifyEqual(at, numel(seen));
                tc.verifyEqual(r.EpisodeReturns, returns, sprintf('%s: returns', name));
            end
        end
    end
end
