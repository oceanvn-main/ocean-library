classdef RewardTerminationTest < OceanTest.BaseTestCase
    % REWARDTERMINATIONTEST  G1 for OceanAgents.World.Doors.RewardTermination (W2_Grid_Doors_Contract.md 3.2, 7):
    % rewards (step, cell, cliff), Steps counting, termination on a goal, truncation at the time limit, termination
    % winning when both happen on the same move, and the door driven by a World with the host outbox.

    methods (Static, Access = private)
        function cfg = cliffCfg(limit)
            gw = OceanAgents.World.Doors.GridWorlds.cliffWalking('TimeLimit', limit);
            cfg = gw.RewardTermination;
            cfg.Cols = gw.GridMotion.Cols;                        % the rule reads GridMotion's Cols
        end
        function ctx = ctx(read, steps, ids, events)
            read.Steps = steps;
            read.Steps_Has = true(numel(steps), 1);
            ctx = struct('Now', 1, 'Phase', "settle", 'Door', "RewardTermination", 'Rows', (1:numel(steps))', ...
                'IDs', ids(:), 'Read', read, 'Events', events, 'Stream', []);
        end
        function e = ev(type, to, id)
            payload = struct('From', 1, 'To', to);
            if type == "CliffFall", payload.Fell = 40; end
            e = OceanAgents.World.Event('Type', type, 'Time', 1, 'Phase', "act", 'SourceEntity', id, 'Payload', payload);
        end
    end

    methods (Test)
        function stepCliffAndGoal(tc)
            read = tc.cliffCfg(Inf);
            out = OceanAgents.World.Doors.RewardTermination.settleRule(tc.ctx(read, 0, 5, ...
                [tc.ev("Moved", 25, 5), tc.ev("CliffFall", 37, 5), tc.ev("Moved", 48, 5)]));
            tc.verifyEqual([out.Events.Type], ["Reward" "Reward" "Reward" "Terminated"]);
            amounts = arrayfun(@(e) e.Payload.Amount, out.Events(1:3));
            tc.verifyEqual(amounts, [-1 -100 -1]);
            tc.verifyEqual(out.Events(4).Payload.Reason, "terminal");
            tc.verifyEqual(unique([out.Events.SourceEntity]), 5);
            tc.verifyEqual(out.Writes.Values, 3);
            tc.verifyEqual(out.Writes.Rows, 1);
        end

        function cellRewardsOfTheRandomWalk(tc)
            % walk19 with rewards -1 / 1: entering cell 1 gives -1, cell 21 gives 1, inner cells 0 (StepReward).
            gw = OceanAgents.World.Doors.GridWorlds.randomWalk('NumInner', 19, 'LeftReward', -1, 'RightReward', 1);
            read = gw.RewardTermination;  read.Cols = gw.GridMotion.Cols;
            out = OceanAgents.World.Doors.RewardTermination.settleRule(tc.ctx(read, 0, 1, ...
                [tc.ev("Moved", 10, 1), tc.ev("Moved", 1, 1)]));
            tc.verifyEqual([out.Events.Type], ["Reward" "Reward" "Terminated"]);
            tc.verifyEqual([out.Events(1).Payload.Amount out.Events(2).Payload.Amount], [0 -1]);
            out = OceanAgents.World.Doors.RewardTermination.settleRule(tc.ctx(read, 4, 1, tc.ev("Moved", 21, 1)));
            tc.verifyEqual(out.Events(1).Payload.Amount, 1);
            tc.verifyEqual(out.Writes.Values, 5);
        end

        function truncationAtTheLimitAndTerminationWins(tc)
            read = tc.cliffCfg(3);
            out = OceanAgents.World.Doors.RewardTermination.settleRule(tc.ctx(read, 1, 1, tc.ev("Moved", 25, 1)));
            tc.verifyEqual([out.Events.Type], "Reward");                       % Steps 2 < 3
            out = OceanAgents.World.Doors.RewardTermination.settleRule(tc.ctx(read, 2, 1, tc.ev("Moved", 25, 1)));
            tc.verifyEqual([out.Events.Type], ["Reward" "Truncated"]);        % Steps 3 >= 3
            tc.verifyEqual(out.Events(2).Payload.Reason, "time-limit");
            out = OceanAgents.World.Doors.RewardTermination.settleRule(tc.ctx(read, 2, 1, tc.ev("Moved", 48, 1)));
            tc.verifyEqual([out.Events.Type], ["Reward" "Terminated"], 'termination wins at the limit');
            out = OceanAgents.World.Doors.RewardTermination.settleRule(tc.ctx(read, 2, 1, tc.ev("CliffFall", 37, 1)));
            tc.verifyEqual([out.Events.Type], ["Reward" "Truncated"]);
        end

        function eventsForSeveralEntities(tc)
            read = tc.cliffCfg(Inf);
            out = OceanAgents.World.Doors.RewardTermination.settleRule(tc.ctx(read, [0; 5], [3; 4], ...
                [tc.ev("Moved", 25, 4), tc.ev("Moved", 26, 3), tc.ev("Moved", 27, 4)]));
            tc.verifyEqual(out.Writes.Rows, [1; 2]);
            tc.verifyEqual(out.Writes.Values, [1; 7]);
            tc.verifyEqual([out.Events.SourceEntity], [4 3 4]);
            tc.verifyError(@() OceanAgents.World.Doors.RewardTermination.settleRule(tc.ctx(read, 0, 3, tc.ev("Moved", 1, 9))), ...
                'OceanAgents:World:BadValue');
        end

        function specAndValidation(tc)
            gw = OceanAgents.World.Doors.GridWorlds.cliffWalking();
            d = OceanAgents.World.Doors.RewardTermination.spec(gw.RewardTermination);
            tc.verifyEqual([d.Name d.Drive], ["RewardTermination" "on-event"]);
            t = d.Owns.Type;
            tc.verifyEqual([t.Visibility t.Kind], ["self" "value"]);
            tc.verifyTrue(t.Dimension == OceanAgents.World.Guard.Dimension.N());
            tc.verifyEqual(d.Rules.Consumes, ["Moved" "CliffFall"]);
            tc.verifyEqual(d.Rules.Emits, ["Reward" "Terminated" "Truncated"]);
            c = gw.RewardTermination;
            id = 'OceanAgents:World:BadValue';
            b = c;  b.TimeLimit = 0;       tc.verifyError(@() OceanAgents.World.Doors.RewardTermination.spec(b), id);
            b = c;  b.TimeLimit = 2.5;     tc.verifyError(@() OceanAgents.World.Doors.RewardTermination.spec(b), id);
            b = c;  b.StepReward = NaN;    tc.verifyError(@() OceanAgents.World.Doors.RewardTermination.spec(b), id);
            b = c;  b.Goal = true(2);      tc.verifyError(@() OceanAgents.World.Doors.RewardTermination.spec(b), id);
            b = rmfield(c, 'Goal');        tc.verifyError(@() OceanAgents.World.Doors.RewardTermination.spec(b), id);
        end

        function worldDrivenEndings(tc)
            % cliff3x5 with TimeLimit 3 in a World: up, right, right -> truncation after the third move; a second
            % seeded episode: right (cliff: -100, back to the start) ...; walking down the right edge terminates.
            tc.assumeTrue(OceanTest.OceanAgents.World.Doors.GridMotionTest.kernelReady(), ...
                'needs Agency.m and the legacy stream / outbox in World (B1, W2 phase 1)');
            gw = OceanAgents.World.Doors.GridWorlds.cliffWalking('Rows', 3, 'Cols', 5, 'TimeLimit', 3);
            w = OceanAgents.World.World(gw.Doors, 'SkipIdle', false, 'Injectable', "Move", ...
                'Outbox', ["Reward" "Terminated" "Truncated"]);
            w.reset(1);
            id = w.Registry.idsOf(w.Store.rowsWith("Position"));
            got = strings(0, 1);  amounts = [];
            for a = [1 2 2]
                w.inject(OceanAgents.World.Event.make("Move", struct('Label', a), 'SourceEntity', id));
                w.advance();
                ev = w.takeOutbox();
                got = [got; [ev.Type]']; %#ok<AGROW>
                amounts(end + 1) = ev(1).Payload.Amount; %#ok<AGROW>
            end
            tc.verifyEqual(got, ["Reward"; "Reward"; "Reward"; "Truncated"]);
            tc.verifyEqual(amounts, [-1 -1 -1]);
            w.reset(2);
            id = w.Registry.idsOf(w.Store.rowsWith("Position"));
            seq = {};
            for a = [2 1 2 2 2]                                      % cliff fall, then up and along row 2
                w.inject(OceanAgents.World.Event.make("Move", struct('Label', a), 'SourceEntity', id));
                w.advance();
                ev = w.takeOutbox();
                seq{end + 1} = [ev.Type]; %#ok<AGROW>
                if a == 2 && numel(seq) == 1, tc.verifyEqual(ev(1).Payload.Amount, -100); end
            end
            tc.verifyEqual(seq{1}, "Reward");
            tc.verifyEqual(seq{3}, ["Reward" "Truncated"], 'the limit counts the cliff fall as a step');
            w.reset(3);
            id = w.Registry.idsOf(w.Store.rowsWith("Position"));
            wg = OceanAgents.World.Doors.GridWorlds.cliffWalking('Rows', 3, 'Cols', 5);
            w2 = OceanAgents.World.World(wg.Doors, 'SkipIdle', false, 'Injectable', "Move", ...
                'Outbox', ["Reward" "Terminated" "Truncated"]);
            w2.reset(3);
            id2 = w2.Registry.idsOf(w2.Store.rowsWith("Position"));
            last = strings(1, 0);
            for a = [1 2 2 2 2 3]                                    % up, right x4, down into the goal (3,5)
                w2.inject(OceanAgents.World.Event.make("Move", struct('Label', a), 'SourceEntity', id2));
                w2.advance();
                ev = w2.takeOutbox();
                last = [ev.Type];
            end
            tc.verifyEqual(last, ["Reward" "Terminated"]);
            tc.verifyNotEmpty(id);
        end
    end
end
