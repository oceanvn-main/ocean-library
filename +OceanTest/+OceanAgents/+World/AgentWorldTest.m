classdef AgentWorldTest < OceanTest.BaseTestCase
    % AGENTWORLDTEST  OceanAgents.World.AgentWorld and Doors.Agency (W2_Grid_Doors_Contract.md section 4; K-W2):
    % EnvSpec, first frame, exactly one World.advance per step, reward sums from the outbox, endings with termination
    % winning, all-false masks (and successor_action_mask on truncation) after an agent's end, EpisodeEnded and
    % NotReady, roster checks, reset([]) continuation, and one run through the existing JointRunner.
    % Uses the test-only LineDoors fixture (agents on a line; Goal, Limit, Bonus).

    methods (Static, Access = private)
        function L = L()
            L = OceanTest.OceanAgents.Fixtures.World.Agent.LineDoors;
        end

        function b = batch(frame, actions)
            % One decision per requested slot, actions(k) for slot k.
            ds = OceanAgents.Core.Decision.empty(1, 0);
            for k = find(frame.Requested)
                ds(end + 1) = OceanAgents.Core.Decision('AgentID', frame.Roster(k), 'EpisodeID', frame.EpisodeID, ...
                    'FrameID', frame.FrameID, 'Action', actions(k)); %#ok<AGROW>
            end
            b = OceanAgents.Core.ActionBatch('EpisodeID', frame.EpisodeID, 'FrameID', frame.FrameID, 'Decisions', ds);
        end
    end

    methods (Test)
        function agencyDoor(tc)
            d = OceanAgents.World.Doors.Agency.spec();
            tc.verifyEqual(d.Membership, "AgentID");
            tc.verifyEmpty(d.Rules);
            tc.verifyEqual(d.Owns.Type.Class, "string");
            tc.verifyEqual(d.Owns.Type.Visibility, "hidden");
        end

        function describeAndFirstFrame(tc)
            L = tc.L();
            aw = OceanAgents.World.AgentWorld(L.doors(["a1" "a2"]), L.cfg(["a1" "a2"]), 'Name', "line");
            spec = aw.describe();
            tc.verifyEqual(spec.Timing, "simultaneous");
            tc.verifyEqual(spec.Roster, ["a1" "a2"]);
            tc.verifyEqual(spec.Horizon, Inf);
            tc.verifyEmpty(spec.Capabilities.list());
            f = aw.reset(3);
            tc.verifyEqual(f.Roster, ["a1" "a2"]);
            tc.verifyEqual([f.Requested f.Live], true(1, 4));
            tc.verifyFalse(any(f.RewardValid));
            tc.verifyEqual(f.Views{1}.ActionMask, [true true]);
            tc.verifyEqual([f.Views{1}.Observation f.Views{2}.Observation], [0 0], 'own Pos only (self)');
            tc.verifyEqual(aw.NumAdvances, 0);
        end

        function oneAdvancePerStepAndRewardSums(tc)
            % a1 moves right (+1 each step), a2 left: rewards -1 per step; a1 reaches Goal 3 at step 3: -1 + 10 = 9.
            L = tc.L();
            aw = OceanAgents.World.AgentWorld(L.doors(["a1" "a2"], 'Goal', 3), L.cfg(["a1" "a2"]));
            f = aw.reset(1);
            rewards = zeros(3, 2);
            for s = 1:3
                f = aw.step(tc.batch(f, [2 1]));
                tc.verifyEqual(aw.NumAdvances, s, 'exactly one World.advance per step');
                tc.verifyEqual(f.Tick, s - 1, 'the world clock moved by exactly one tick per step (independent of the counter)');
                rewards(s, :) = f.Rewards;
                tc.verifyEqual(f.RewardValid, [true true]);
            end
            tc.verifyEqual(rewards, [-1 -1; -1 -1; 9 -1]);
            tc.verifyEqual([f.Terminated(1) f.Truncated(1)], [true false]);
            tc.verifyEqual(f.Views{1}.ActionMask, [false false], 'ended: all-false mask');
            tc.verifyEqual(f.Views{1}.Observation, 3);
            tc.verifyEqual([f.Live f.Requested], [false true false true]);
            tc.verifyFalse(f.EpisodeEnded);
            % a1 is out: the next frame has no view for it and no reward
            f = aw.step(tc.batch(f, [0 1]));
            tc.verifyEqual(f.Rewards, [0 -1]);
            tc.verifyEmpty(f.Views{1});
        end

        function terminationWinsAndTruncationMask(tc)
            % Limit 3 and Goal 3: a1 reaches the goal exactly at the limit (both events) -> Terminated only;
            % a2 is truncated -> successor_action_mask all true; then EpisodeEnded ("terminal": the first ending).
            L = tc.L();
            aw = OceanAgents.World.AgentWorld(L.doors(["a1" "a2"], 'Goal', 3, 'Limit', 3), L.cfg(["a1" "a2"]));
            f = aw.reset(0);
            for s = 1:3
                f = aw.step(tc.batch(f, [2 1]));
            end
            tc.verifyEqual([f.Terminated; f.Truncated], [true false; false true]);
            tc.verifyTrue(f.EpisodeEnded);
            tc.verifyEqual(f.EndReason, "terminal");
            tc.verifyEqual(f.Views{2}.ActionMask, [false false]);
            tc.verifyEqual(f.Views{2}.Info.successor_action_mask, [true true]);
            tc.verifyFalse(isfield(f.Views{1}.Info, 'successor_action_mask'));
            tc.verifyError(@() aw.step(tc.batch(f, [1 1])), 'OceanAgents:Env:NotReady');
            % feedback records can be built from the frame (as JointRunner does)
            fb = f.feedbackFor(2, f.FrameID - 1);
            tc.verifyTrue(fb.Truncated);
        end

        function rosterAndResetSemantics(tc)
            L = tc.L();
            tc.verifyError(@() OceanAgents.World.AgentWorld(L.line("a1"), L.cfg("a1")), ...
                'OceanAgents:World:BadAssembly');                    % no Agency door
            aw = OceanAgents.World.AgentWorld(L.doors(["a1" "a2"]), L.cfg(["a2" "a1"]));
            tc.verifyError(@() aw.reset(0), 'OceanAgents:Runtime:RosterMismatch');
            ok = OceanAgents.World.AgentWorld(L.doors("a1"), L.cfg("a1"));
            tc.verifyError(@() ok.reset([]), 'OceanAgents:World:NoSeed');
            f1 = ok.reset(4);
            f2 = ok.reset([]);                               % continuing streams: a new episode
            tc.verifyEqual([f1.EpisodeID f2.EpisodeID], [1 2]);
            tc.verifyTrue(f2.FrameID > f1.FrameID);
            tc.verifyError(@() ok.step(tc.batch(f1, 1)), 'OceanAgents:Core:StaleFrame');   % answers the old frame
        end

        function runsUnderJointRunner(tc)
            % The existing JointRunner drives an AgentWorld with PolicyMind(RandomLegalPolicy) hosts.
            L = tc.L();
            aw = OceanAgents.World.AgentWorld(L.doors(["a1" "a2"], 'Goal', 2, 'Limit', 6), L.cfg(["a1" "a2"]));
            hosts = {OceanAgents.Runtime.AgentHost("a1", OceanAgents.Minds.PolicyMind(OceanAgents.Policies.RandomLegalPolicy())), ...
                OceanAgents.Runtime.AgentHost("a2", OceanAgents.Minds.PolicyMind(OceanAgents.Policies.RandomLegalPolicy()))};
            runner = OceanAgents.Runtime.JointRunner(aw, hosts, 'Seed', 11);
            rec = runner.run(5);
            tc.verifyEqual(rec.Status, "Completed");
            tc.verifyEqual(rec.NumEpisodes, 5);
            tc.verifyTrue(all(rec.EpisodeLengths >= 2 & rec.EpisodeLengths <= 6));
            tc.verifyEqual(rec.StepCalls, sum(rec.EpisodeLengths));
            tc.verifyEqual(aw.NumAdvances, rec.StepCalls, 'one World.advance per joint step');
            % every step costs -1 per acting agent, +10 on reaching the goal: returns are integers <= 10 - 2
            tc.verifyTrue(all(rec.EpisodeReturns(:) <= 8));
        end

        function pillars(tc)
            L = tc.L();
            aw = OceanAgents.World.AgentWorld(L.doors("a1"), L.cfg("a1"));
            tc.verifyFalse(aw.getMetadata().IsSymbolic);
            tc.verifyEqual(aw.getPipelineSpec().ComponentName, "AgentWorld");
            tc.verifyFalse(OceanAgents.World.Doors.Agency.getMetadata().IsSymbolic);
        end
    end
end
