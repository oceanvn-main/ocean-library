classdef RLDoorCoverageTest < OceanTest.BaseTestCase
    % RLDOORCOVERAGETEST  A check that never ran is not a pass (Gymnasium import finding; design decision D9):
    % certificates record per-condition coverage, an axiom with an unreached condition is "partial" and never
    % "probed", and RL-5 probes are directed to a terminal state when the module declares an exact model.

    methods (Test)
        function directedProbeReachesATerminalOnModelEnvironments(tc)
            door = OceanAgents.World.Doors.RLDoor.spec('Probes', 6);
            envs = {OceanStats.RL.Env.Classic.CliffWalkingEnv(), OceanStats.RL.Env.Classic.WindyGridworldEnv(), ...
                    OceanAgents.Env.WorldAsEnvironment(OceanAgents.World.Doors.GridWorlds.cliffWalking())};
            for k = 1:numel(envs)
                cert = OceanAgents.World.Guard.GuardEngine.admitModule(envs{k}, door, 'ModuleVersion', "1");
                rl5 = cert.Axioms(5);
                tc.verifyTrue(cert.Admitted, class(envs{k}));
                tc.verifyEqual(rl5.Status, "probed", class(envs{k}));
                tc.verifyEqual(rl5.Coverage.afterTermination, 6, class(envs{k}));   % every probe reached a terminal
                tc.verifyEqual(rl5.Coverage.outsideAction, 6);
            end
        end

        function aModelEnvironmentThatAcceptsStepsAfterTheEndIsRefused(tc)
            % Random 50-step episodes almost never reach the cliff-walking goal: before directed probes this
            % violator was admitted.
            env = OceanTest.OceanAgents.Fixtures.World.RL.AcceptsAfterEndCliffEnv();
            cert = OceanAgents.World.Guard.GuardEngine.admitModule(env, OceanAgents.World.Doors.RLDoor.spec('Probes', 6), ...
                'ModuleVersion', "1");
            tc.verifyFalse(cert.Admitted);
            tc.verifyEqual(cert.failed(), "RL-5");
            tc.verifySubstring(cert.Axioms(5).Detail, "after termination was accepted");
        end

        function anUnreachedConditionIsPartialNeverProbed(tc)
            env = OceanTest.OceanAgents.Fixtures.World.RL.NeverEndingEnv();
            door = OceanAgents.World.Doors.RLDoor.spec('Probes', 6);
            cert = OceanAgents.World.Guard.GuardEngine.admitModule(env, door, 'ModuleVersion', "1");
            tc.verifyTrue(cert.Admitted);                                         % nothing failed
            tc.verifyEqual(cert.Axioms(5).Status, "partial");
            tc.verifyEqual(cert.Axioms(5).Coverage.afterTermination, 0);
            tc.verifyEqual(cert.Axioms(5).Coverage.outsideAction, 6);
            tc.verifySubstring(cert.Axioms(5).Detail, "never reached in any probe: afterTermination");
            tc.verifyEqual(cert.unverified(), ["RL-5" "RL-9"]);
            strict = OceanAgents.World.Guard.GuardEngine.admitModule(env, door, 'ModuleVersion', "1", 'RequireCoverage', true);
            tc.verifyFalse(strict.Admitted);
            tc.verifyEqual(strict.failed(), "RL-5");
        end

        function certificatesWithoutCoverageStillWork(tc)
            ax = struct('ID', "M-1", 'Mode', "probe", 'Status', "partial", 'Probes', 4, 'Seeds', 1:4, 'Detail', "");
            c = OceanAgents.World.Guard.Certificate('Subject', "Toy", 'ModuleVersion', "1", 'Door', "rl", ...
                'DoorVersion', "1", 'Axioms', ax);
            tc.verifyTrue(c.Admitted);
            tc.verifyEqual(c.unverified(), "M-1");
            tc.verifyTrue(isstruct(c.Axioms(1).Coverage) && isempty(fieldnames(c.Axioms(1).Coverage)));
        end
    end
end
