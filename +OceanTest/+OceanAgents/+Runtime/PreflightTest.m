classdef PreflightTest < OceanTest.BaseTestCase
    % PREFLIGHTTEST  Runner construction rejects an unfit agent BEFORE the environment is reset (critic round 1:
    % F1 role, F5 shared components, F6 exclusive host binding, F7 import rejection with real artifacts/policies).
    % Every rejection asserts the environment saw zero resets.

    methods (Static, Access = private)
        function b = walkBridge(role)
            % RandomWalk(NumInner = 5) through the real bridge: observations Discrete(7), labels [1 2].
            b = OceanAgents.Env.NativeSingleAgentBridge(OceanStats.RL.Env.Classic.RandomWalkEnv('NumInner', 5), ...
                'AgentID', "a", 'Role', role);
        end
        function h = tableHost(id, labels, role, numStates, stateStart, hostRole)
            % A TablePolicy agent whose artifact always plays its first label.
            art = OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Representation', "Label", ...
                'Table', repmat(labels(1), numStates, 1), 'ActionLabels', labels, 'NumStates', numStates, ...
                'StateStart', stateStart, 'Role', role);
            h = OceanAgents.Runtime.AgentHost(id, OceanAgents.Minds.PolicyMind(OceanAgents.Policies.TablePolicy(art)), ...
                'Role', hostRole);
        end
        function tf = neverReset(bridge)
            tf = bridge.EpisodeID == 0 && bridge.Status == "Idle" && bridge.NativeStepCalls == 0;
        end
    end

    methods (Test)
        function singleAgentImportRejection(tc)
            R = @OceanAgents.Runtime.SingleAgentRunner;
            % positive control: a fitting artifact runs
            b = tc.walkBridge("defender");
            rec = R(b, tc.tableHost("a", [1 2], "defender", 7, 1, "default")).run(1);
            tc.verifyEqual(rec.Status, "Completed");
            % wrong role (critic probe: artifact for "default" on a "defender" slot, host left at "default")
            b = tc.walkBridge("defender");
            tc.verifyError(@() R(b, tc.tableHost("a", [1 2], "default", 7, 1, "default")), 'OceanAgents:Policies:Incompatible');
            tc.verifyTrue(tc.neverReset(b));
            % host explicitly declaring another role
            b = tc.walkBridge("defender");
            tc.verifyError(@() R(b, tc.tableHost("a", [1 2], "defender", 7, 1, "attacker")), 'OceanAgents:Runtime:RosterMismatch');
            tc.verifyTrue(tc.neverReset(b));
            % wrong action count
            b = tc.walkBridge("default");
            tc.verifyError(@() R(b, tc.tableHost("a", [1 2 3], "default", 7, 1, "default")), 'OceanAgents:Policies:Incompatible');
            tc.verifyTrue(tc.neverReset(b));
            % wrong label convention (0-based export [0 1] against labels [1 2])
            b = tc.walkBridge("default");
            tc.verifyError(@() R(b, tc.tableHost("a", [0 1], "default", 7, 1, "default")), 'OceanAgents:Policies:Incompatible');
            tc.verifyTrue(tc.neverReset(b));
            % wrong state indexing (states declared from 0)
            b = tc.walkBridge("default");
            tc.verifyError(@() R(b, tc.tableHost("a", [1 2], "default", 7, 0, "default")), 'OceanAgents:Policies:Incompatible');
            tc.verifyTrue(tc.neverReset(b));
            % wrong representation: a ColumnIndex table holding a label (0) is refused when the policy is built,
            % i.e. before any runner or environment exists (A0 section 9.6)
            art = OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Representation', "ColumnIndex", ...
                'Table', zeros(7, 1), 'ActionLabels', [0 1], 'NumStates', 7);
            tc.verifyError(@() OceanAgents.Policies.TablePolicy(art), 'OceanAgents:Artifacts:Incompatible');
        end

        function jointImportRejection(tc)
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            J = @OceanAgents.Runtime.JointRunner;
            mk = @() ToyJointEnv('Roster', ["p1" "p2"], 'Roles', ["row" "col"], 'Rounds', 3, 'DiscreteObs', true);
            % positive control: observations Discrete(4, Start = 0), labels [1 2]
            env = mk();
            rec = J(env, {tc.tableHost("p1", [1 2], "row", 4, 0, "row"), tc.tableHost("p2", [1 2], "col", 4, 0, "default")}).run(1);
            tc.verifyEqual(rec.Status, "Completed");
            cases = {
                {"row", [1 2], "default", 'OceanAgents:Policies:Incompatible'}    % p2 artifact for the other role
                {"col", [1 2], "row", 'OceanAgents:Runtime:RosterMismatch'}       % p2 host declares the other role
                {"col", [1 2 3], "default", 'OceanAgents:Policies:Incompatible'}  % wrong action count
                {"col", [0 1], "default", 'OceanAgents:Policies:Incompatible'}    % wrong label convention
                };
            for c = 1:numel(cases)
                [artRole, labels, hostRole, id] = cases{c}{:};
                env = mk();
                h1 = tc.tableHost("p1", [1 2], "row", 4, 0, "row");
                h2 = tc.tableHost("p2", labels, artRole, 4, 0, hostRole);
                tc.verifyError(@() J(env, {h1, h2}), id, sprintf('case %d', c));
                tc.verifyEqual(env.ResetCalls, 0, sprintf('case %d: no reset before rejection', c));
                tc.verifyFalse(h1.isBound() || h2.isBound(), sprintf('case %d: a rejected runner binds nobody', c));
            end
        end

        function jointRejectsSharedComponents(tc)
            % F5: two hosts that share a mutable object (here: one containers.Map inside both policies' minds).
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            env = ToyJointEnv();
            q = containers.Map();
            mA = ScriptedMind(1);
            mB = ScriptedMind(1);
            mA.Payload = {q};
            mB.Payload = {struct('table', q)};
            hA = OceanAgents.Runtime.AgentHost("p1", mA);
            hB = OceanAgents.Runtime.AgentHost("p2", mB);
            try
                OceanAgents.Runtime.JointRunner(env, {hA, hB});
                tc.verifyFail('shared handle was accepted');
            catch err
                tc.verifyEqual(err.identifier, 'OceanAgents:Runtime:SharedComponent');
                tc.verifySubstring(err.message, 'p1.Mind.Payload{1} == p2.Mind.Payload{1}.table');
            end
            tc.verifyEqual(env.ResetCalls, 0);
            % allowlisting the class accepts it (a deliberate, documented exception)
            r = OceanAgents.Runtime.JointRunner(env, {hA, hB}, 'IsolationAllow', "containers.Map");
            tc.verifyEqual(r.Status, "Ready");
        end

        function agentMustNotReachTheEnvironment(tc)
            % F5 (host <-> env): a mind holding a handle that the environment also holds.
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            log = EventLog();
            env = SpySingleEnv('Log', log);
            mind = ScriptedMind(2);
            mind.Payload = {log};
            host = OceanAgents.Runtime.AgentHost("agent1", mind);
            tc.verifyError(@() OceanAgents.Runtime.SingleAgentRunner(env, host), 'OceanAgents:Runtime:SharedComponent');
            tc.verifyEqual(env.ResetCalls, 0);
            jenv = ToyJointEnv('Log', log);
            h2 = OceanAgents.Runtime.AgentHost("p2", ScriptedMind(1));
            host1 = OceanAgents.Runtime.AgentHost("p1", ScriptedMind(1, 'Log', log));
            tc.verifyError(@() OceanAgents.Runtime.JointRunner(jenv, {host1, h2}), 'OceanAgents:Runtime:SharedComponent');
            tc.verifyEqual(jenv.ResetCalls, 0);
        end

        function hostIsBoundToOneRunner(tc)
            % F6: a second runner cannot capture a host (it would replace the first runner's stream and change
            % that runner's draws); after release/delete of the first runner the host is free again.
            import OceanTest.OceanAgents.Fixtures.Runtime.*
            mind = FixedProbabilityMind([0.5 0.5]);
            host = OceanAgents.Runtime.AgentHost("agent1", mind);
            r1 = OceanAgents.Runtime.SingleAgentRunner(SpySingleEnv(), host, 'Seed', 1);
            env2 = SpySingleEnv();
            tc.verifyError(@() OceanAgents.Runtime.SingleAgentRunner(env2, host, 'Seed', 2), ...
                'OceanAgents:Runtime:SharedComponent');
            tc.verifyEqual(env2.ResetCalls, 0);
            % r1's draws are exactly those of an untouched reference run
            recRef = OceanAgents.Runtime.SingleAgentRunner(SpySingleEnv(), ...
                OceanAgents.Runtime.AgentHost("agent1", FixedProbabilityMind([0.5 0.5])), 'Seed', 1).run(3);
            rec1 = r1.run(3);
            tc.verifyEqual(rec1.EpisodeLengths, recRef.EpisodeLengths);
            % joint runner cannot take it either
            jh = OceanAgents.Runtime.AgentHost("p2", ScriptedMind(1));
            tc.verifyError(@() OceanAgents.Runtime.JointRunner(ToyJointEnv('Roster', ["agent1" "p2"]), {host, jh}), ...
                'OceanAgents:Runtime:SharedComponent');
            tc.verifyFalse(jh.isBound(), 'a rejected joint runner must not keep the other host');
            % release frees it
            r1.release();
            tc.verifyEqual(r1.Status, "Released");
            tc.verifyError(@() r1.run(1), 'OceanAgents:Runtime:InvalidState');
            r2 = OceanAgents.Runtime.SingleAgentRunner(SpySingleEnv(), host, 'Seed', 2);
            tc.verifyTrue(host.isBound());
            % deleting a runner frees it too
            delete(r2);
            tc.verifyFalse(host.isBound());
            r3 = OceanAgents.Runtime.SingleAgentRunner(SpySingleEnv(), host, 'Seed', 3);
            tc.verifyEqual(r3.run(1).Status, "Completed");
        end
    end
end
