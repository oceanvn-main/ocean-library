classdef GymAdmissionTest < OceanTest.BaseTestCase
    % GYMADMISSIONTEST  Foreign environments at the RL door: a Gymnasium environment that follows the standard is
    % admitted, raw Gymnasium behaviour (steps accepted after termination) is refused by RL-5, the exact solver gives
    % the known FrozenLake optimum, and the runner accepts an environment holding a Python object (isolation check).
    % Skipped (assumption) when MATLAB has no Python with gymnasium.

    methods (TestClassSetup)
        function needsGymnasium(tc)
            ok = false;
            try
                py.importlib.import_module('gymnasium');
                ok = true;
            catch
            end
            tc.assumeTrue(ok, 'needs Python with the gymnasium package');
        end
    end

    methods (Test)
        function conformingEnvironmentsAreAdmitted(tc)
            door = OceanAgents.World.Doors.RLDoor.spec();
            envs = {OceanStats.RL.Env.External.GymTabularEnv("FrozenLake-v1"), ...
                    OceanStats.RL.Env.External.GymTabularEnv("Taxi-v4"), OceanStats.RL.Env.External.GymEnv("CartPole-v1")};
            for k = 1:numel(envs)
                cert = OceanAgents.World.Guard.GuardEngine.admitModule(envs{k}, door, 'ModuleVersion', "1");
                tc.verifyTrue(cert.Admitted, sprintf('%s: failed [%s]', envs{k}.Name, strjoin(cert.failed(), ' ')));
            end
        end

        function rawGymnasiumBehaviourIsRefused(tc)
            door = OceanAgents.World.Doors.RLDoor.spec();
            for name = ["FrozenLake-v1" "Taxi-v4" "CartPole-v1"]          % Taxi: only a directed probe reaches a terminal
                if startsWith(name, "CartPole"), env = OceanStats.RL.Env.External.GymEnv(name, 'Strict', false);
                else, env = OceanStats.RL.Env.External.GymTabularEnv(name, 'Strict', false); end
                cert = OceanAgents.World.Guard.GuardEngine.admitModule(env, door, 'ModuleVersion', "1");
                tc.verifyFalse(cert.Admitted);
                tc.verifyEqual(cert.failed(), "RL-5");
            end
        end

        function exactSolverGivesTheKnownOptimum(tc)
            fl = OceanStats.RL.Env.External.GymTabularEnv("FrozenLake-v1");
            vi = OceanStats.RL.DP.ValueIterationSolver(fl, 'DiscountFactor', 1);
            V = vi.run('MaxIter', 5000);
            tc.verifyEqual(V(1), 14 / 17, 'AbsTol', 1e-6);              % slippery 4x4 map, success probability
        end

        function runnerAcceptsAnEnvironmentHoldingAPythonObject(tc)
            fl = OceanStats.RL.Env.External.GymTabularEnv("FrozenLake-v1");
            bridge = OceanAgents.Env.NativeSingleAgentBridge(fl);
            codec = OceanAgents.Artifacts.DiscreteStateCodec(16);
            q = OceanStats.RL.TD.Values.TabularQ(fl.ValidMask, 0);
            policy = OceanAgents.Policies.EpsilonGreedyQPolicy(q, 1:4, 'Epsilon', 0.1, 'StateCodec', codec);
            learner = OceanAgents.Learners.OneStepQLearner(q, 1:4, 'Alpha', 0.1, 'Gamma', 0.99, 'StateCodec', codec);
            host = OceanAgents.Runtime.AgentHost("agent1", OceanAgents.Minds.PolicyMind(policy), 'Learner', learner, 'Gamma', 0.99);
            runner = OceanAgents.Runtime.SingleAgentRunner(bridge, host, 'Seed', 7);
            rec = runner.run(20);
            tc.verifyEqual(rec.Status, "Completed");
            tc.verifyEqual(rec.NumEpisodes, 20);
        end
    end
end
