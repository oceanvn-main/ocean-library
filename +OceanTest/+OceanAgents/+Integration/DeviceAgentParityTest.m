classdef DeviceAgentParityTest < OceanTest.BaseTestCase
    % DEVICEAGENTPARITYTEST  DV1 V3 and V5: the AB2 agent (Q mind, grid robot body with sensors) runs identically on
    % cliff walking made of devices and on the native environment. In the device world the body's OWN GridPosition is
    % the position the terrain moves (design D9), so the runner is told that this one component is shared on purpose:
    % the body is the boundary between agent and world, and the mind never reads it (it sees only observations).
    % The cost per step is logged against the native environment.

    methods (Static)
        function [rec, q, seconds, robot] = runAgent(kind, episodes)
            codec = OceanAgents.Artifacts.DiscreteStateCodec(48);
            native = OceanStats.RL.Env.Classic.CliffWalkingEnv();
            q = OceanStats.RL.TD.Values.TabularQ(native.ValidMask, 0);
            mind = OceanAgents.Minds.QSignalMind(q, 1:4, 'Epsilon', 0.1, 'StateCodec', codec);
            mind.addLearner(OceanAgents.Learners.OneStepQLearner(q, 1:4, 'Alpha', 0.5, 'Gamma', 1, 'StateCodec', codec), "q");
            position = OceanAgents.Components.GridPosition(48);
            robot = OceanAgents.Body.GridRobotBody(string(1:4), 'Parts', {position});
            OceanAgents.Embodied.Dualism.form(mind, robot);
            if kind == "devices"
                env = OceanAgents.Devices.DeviceWorld.cliffWalking('Position', robot.part("position"));
            else
                env = native;
            end
            host = OceanAgents.Runtime.AgentHost("agent1", mind, 'Gamma', 1);
            runner = OceanAgents.Runtime.SingleAgentRunner(OceanAgents.Env.NativeSingleAgentBridge( ...
                OceanStats.RL.Env.Wrappers.TimeLimit(env, 200)), host, 'Seed', 7, ...
                'IsolationAllow', ["OceanStats.RL.Env.Spaces.ISpace" "OceanAgents.Components.GridPosition"]);
            tic;
            rec = runner.run(episodes);
            seconds = toc;
        end
    end

    methods (Test)
        function theAgentRunsIdenticallyOnDevicesV3(tc)
            [ref, qRef, tRef] = tc.runAgent("native", 60);
            [rec, q, t, robot] = tc.runAgent("devices", 60);
            tc.verifyEqual(rec.Status, "Completed");
            tc.verifyEqual(rec.EpisodeReturns, ref.EpisodeReturns);
            tc.verifyEqual(rec.EpisodeLengths, ref.EpisodeLengths);
            tc.verifyEqual(rec.StepCalls, ref.StepCalls);
            tc.verifyEqual(q.Q, qRef.Q);
            tc.verifyEqual(mean(rec.EpisodeReturns(1:10)), -290.8, 'AbsTol', 1e-9);   % recipe 1
            tc.verifyEqual(robot.part("position").Moves, rec.StepCalls + rec.NumEpisodes, ...
                'the body''s own position was moved by the terrain at every step and every reset');
            tc.log(matlab.unittest.Verbosity.Terse, sprintf(['V5 cost: agent on devices %.3f ms/step, agent on the native ' ...
                'environment %.3f ms/step (x%.2f), %d steps'], 1e3 * t / rec.StepCalls, 1e3 * tRef / ref.StepCalls, ...
                t / tRef, rec.StepCalls));
        end

        function bareWorldCostV5(tc)
            % the world alone, random legal moves: devices vs native vs door-built (reported, loose bound)
            envs = {OceanStats.RL.Env.Classic.CliffWalkingEnv(), OceanAgents.Devices.DeviceWorld.cliffWalking(), ...
                OceanAgents.Env.WorldAsEnvironment(OceanAgents.World.Doors.GridWorlds.cliffWalking())};
            names = ["native" "devices" "doors"];
            ms = zeros(1, 3);
            for k = 1:3
                env = envs{k};  pick = RandStream('mt19937ar', 'Seed', 1);  env.reset('Seed', 7);  n = 1500;
                tic;
                for i = 1:n
                    [~, ~, term] = env.step(randi(pick, 4));
                    if term, env.reset(); end
                end
                ms(k) = 1e3 * toc / n;
            end
            tc.log(matlab.unittest.Verbosity.Terse, sprintf('V5 bare world per step: %s', ...
                strjoin(compose("%s %.3f ms", names', ms'), ", ")));
            tc.verifyLessThan(ms(2), ms(3), 'devices should be cheaper than the door-built world');
        end
    end
end
