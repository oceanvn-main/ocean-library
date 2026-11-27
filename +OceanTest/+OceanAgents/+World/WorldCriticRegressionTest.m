classdef WorldCriticRegressionTest < OceanTest.BaseTestCase
    % WORLDCRITICREGRESSIONTEST  Regression tests for the W1 critic round 1 findings owned by the World loop
    % (W1_Kernel_Contract.md section 12, "Critic round 1 decisions"). Each test reproduces the critic's probe
    % scenario (criticW1/probe_bugs.m, B1-B9) and checks the decided behaviour.

    methods (Static, Access = private)
        function out = onlyAt(c, at, cmd)
            out = struct();
            if c.Now == at, out.Commands = cmd; end
        end
    end

    methods (Test)
        function f1SplitAttachCannotBypassAdmission(tc)
            % B2: gate (members X and Y) has G-1: X in [0, 1]. Attaching X = 5 then Y = 0 to one entity in ONE sync:
            % the second attach is evaluated on the entity as left by the first, so it completes gate membership with
            % X = 5 and is refused by G-1 - exactly like the single attach of both.
            CT = @OceanAgents.World.Guard.ComponentType;
            tX = CT();  tY = CT();  tZ = CT();
            % (the owning door issues both attaches: attach may only name the issuing door's components, N1)
            two = @(c) tc.onlyAt(c, 0, [OceanAgents.World.Command.attach(1, struct('X', struct('Value', 5, 'Type', tX))), ...
                OceanAgents.World.Command.attach(1, struct('Y', struct('Value', 0, 'Type', tY)))]);
            gate = OceanAgents.World.DoorSpec('Name', "gate", 'Owns', struct('Name', {"X", "Y"}, 'Type', {tX, tY}), ...
                'Membership', ["X" "Y"], 'Axioms', OceanAgents.World.Guard.AxiomSpec('ID', "G-1", 'Mode', "static", ...
                'Component', "X", 'Range', [0 1]), 'Drive', "every-k", 'DriveK', 100, ...
                'Rules', OceanAgents.World.RuleSpec('Name', "Att", 'Phase', "act", 'Commands', "attach", 'Fn', two));
            host = OceanAgents.World.DoorSpec('Name', "host", 'Owns', struct('Name', "Z", 'Type', tZ), 'Drive', "never", ...
                'Init', @(s) OceanAgents.World.Command.spawn(struct('Z', struct('Value', 0, 'Type', tZ))));
            w = OceanAgents.World.World([gate host]);
            w.reset(0);
            r = w.run(0);
            act = r.Phases(2);
            tc.verifyEqual(act.CommandsApplied, 1);
            tc.verifyEqual(numel(act.CommandsRefused), 1);
            tc.verifyEqual(act.CommandsRefused(1).AxiomIDs, "G-1");
            row = w.Registry.rowsOf(1);
            tc.verifyTrue(w.Store.has("X", row));
            tc.verifyFalse(w.Store.has("Y", row), 'the entity never joins gate with X = 5');
        end

        function f2AttachOfAnExistingComponentIsRefused(tc)
            % B3: the owner writes V + 1 every act; at t = 2 it also attaches V = 999 to the entity that already has V.
            % Commands never change values: refused with Reason "already-has", V stays rule-driven (4 after t = 0..2).
            % (A NON-owner's attach is refused earlier, Reason "not-owner": WorldCriticRound2Test.)
            CT = @OceanAgents.World.Guard.ComponentType;
            tV = CT();
            meddle = @(c) tc.onlyAt(c, 2, OceanAgents.World.Command.attach(1, struct('V', struct('Value', 999, 'Type', tV))));
            own = OceanAgents.World.DoorSpec('Name', "own", 'Owns', struct('Name', "V", 'Type', tV), 'Membership', "V", ...
                'Drive', "every-tick", 'Init', @(s) OceanAgents.World.Command.spawn(struct('V', struct('Value', 1, 'Type', tV))), ...
                'Rules', [OceanAgents.World.RuleSpec('Name', "Inc", 'Phase', "act", 'Requires', "V", 'Reads', "V", ...
                'Writes', "V", 'Fn', @(c) struct('Writes', struct('Component', "V", 'Rows', c.Rows, 'Values', c.Read.V + 1))), ...
                OceanAgents.World.RuleSpec('Name', "Over", 'Phase', "act", 'Commands', "attach", 'Fn', meddle)]);
            w = OceanAgents.World.World(own);
            w.reset(0);
            r = w.run(2);
            tc.verifyEqual(w.Store.valueAt("V", w.Registry.rowsOf(1), 2), 4);
            tc.verifyEqual(r(end).Phases(2).CommandsApplied, 0);
            tc.verifyEqual(r(end).Phases(2).CommandsRefused(1).Reason, "already-has");
            % the same component attached twice in ONE sync: the second is refused (the first already added it)
            twice = @(c) tc.onlyAt(c, 0, [OceanAgents.World.Command.attach(1, struct('W', struct('Value', 1, 'Type', tV))), ...
                OceanAgents.World.Command.attach(1, struct('W', struct('Value', 2, 'Type', tV)))]);
            wd = OceanAgents.World.DoorSpec('Name', "wdoor", 'Owns', struct('Name', "W", 'Type', tV), 'Drive', "every-tick", ...
                'Init', @(s) OceanAgents.World.Command.spawn(struct('W0', struct('Value', 0, 'Type', tV))), ...
                'Rules', OceanAgents.World.RuleSpec('Name', "Twice", 'Phase', "act", 'Commands', "attach", 'Fn', twice));
            w0 = OceanAgents.World.DoorSpec('Name', "w0door", 'Owns', struct('Name', "W0", 'Type', tV), 'Drive', "never");
            w2 = OceanAgents.World.World([wd w0]);
            w2.reset(0);
            r2 = w2.run(0);
            tc.verifyEqual(r2.Phases(2).CommandsApplied, 1);
            tc.verifyEqual(r2.Phases(2).CommandsRefused(1).Reason, "already-has");
            tc.verifyEqual(w2.Store.valueAt("W", w2.Registry.rowsOf(1), 0), 1);
        end

        function f3NoReservedNamesForMonitorChecks(tc)
            % B1: a battery door renamed "monitor" behaves exactly like "battery" (names are addresses).
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            init = @(s) L.initSpawns({L.offer('Charge', struct('Value', 10, 'Rate', -0.3))});
            b = L.battery('Init', init);
            ren = OceanAgents.World.DoorSpec('Name', "monitor", 'Owns', b.Owns, 'Membership', b.Membership, ...
                'Rules', b.Rules, 'Axioms', b.Axioms, 'Drive', b.Drive, 'Init', init);
            out = cell(1, 2);
            for k = 1:2
                d = [b ren];
                w = OceanAgents.World.World(d(k));
                w.reset(0);
                reps = w.run(100);
                out{k} = struct('Times', [reps.Time], 'Alive', w.Registry.isAlive(1), 'Status', w.Status, ...
                    'Store', w.Store.snapshot(), 'Count', w.Calendar.count());
            end
            tc.verifyEqual(out{2}, out{1});
            tc.verifyEqual(out{1}.Times, 34);
            tc.verifyFalse(out{1}.Alive);
            % monitor checks are flagged Calendar items, not named ones
            w = OceanAgents.World.World(ren);
            w.reset(0);
            items = w.Calendar.peek();
            tc.verifyEqual(sort([items.Kind]), ["monitor" "prediction"]);
        end

        function f4PredicateMonitorRefusedWhenSkipping(tc)
            % B5: predicate monitor AG-9 (Age <= 7) with a predictor that predicts something else (birthday at 3).
            CT = @OceanAgents.World.Guard.ComponentType;
            D = OceanAgents.World.Guard.Dimension;
            tA = CT('Kind', "clock", 'Dimension', D.T());
            ag = OceanAgents.World.DoorSpec('Name', "ag", 'Owns', struct('Name', "Age", 'Type', tA), 'Membership', "Age", ...
                'Drive', "never", 'Init', @(s) OceanAgents.World.Command.spawn(struct('Age', struct('Value', 0, 'Type', tA))), ...
                'Axioms', OceanAgents.World.Guard.AxiomSpec('ID', "AG-9", 'Mode', "monitor", 'Reads', "Age", ...
                'Predicate', @(v) v.Age <= 7), 'Rules', [ ...
                OceanAgents.World.RuleSpec('Name', "Birthday", 'Phase', "settle", 'Kind', "predictor", 'Requires', "Age", ...
                'Reads', "Age", 'Predicts', "Age", 'DueEvent', "Bday", ...
                'Fn', @(c) struct('Predictions', struct('Rows', c.Rows, 'Times', c.Now + 3 - c.Read.Age))), ...
                OceanAgents.World.RuleSpec('Name', "Party", 'Phase', "resolve", 'Consumes', "Bday", 'Fn', @(c) struct())]);
            tc.verifyError(@() OceanAgents.World.World(ag, 'SkipIdle', true), 'OceanAgents:World:BadAssembly');
            w = OceanAgents.World.World(ag, 'SkipIdle', false);
            w.reset(0);
            tc.verifyError(@() w.run(20), 'OceanAgents:Guard:AxiomViolated');
            tc.verifyEqual(w.Now, 8, 'stepping catches it at the first instant with Age > 7');
        end

        function f5InjectAfterRunSameInstantBothWays(tc)
            % B6: after run(10) an injected Ping is delivered at 11 whether the world skipped (processed 0, 7) or
            % stepped (0..10).
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            tick = OceanAgents.World.DoorSpec('Name', "tick7", 'Drive', "every-k", 'DriveK', 7, 'Rules', ...
                OceanAgents.World.RuleSpec('Name', "Nop", 'Phase', "act", 'Fn', @(c) struct()));
            for skip = [false true]
                w = OceanAgents.World.World([L.counter() tick], 'SkipIdle', skip, 'Injectable', "Ping");
                w.reset(0);
                w.run(10);
                w.inject(OceanAgents.World.Event.make("Ping"));
                r = w.advance();
                tc.verifyEqual(r.Time, 11, sprintf('SkipIdle %d', skip));
                tc.verifyEqual(w.Resources.Pings, 1);
            end
        end

        function f9InjectNeverBeyondTheHorizon(tc)
            % B8: with Horizon 5, after run(100) the next instant would be 6: inject raises InvalidState and the
            % world keeps running (it does not fail).
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            w = OceanAgents.World.World(L.counter(), 'Horizon', 5, 'Injectable', "Ping");
            w.reset(0);
            w.run(100);
            tc.verifyError(@() w.inject(OceanAgents.World.Event.make("Ping")), 'OceanAgents:World:InvalidState');
            tc.verifyEqual(w.Status, "Running");
            tc.verifyEqual(w.Resources.Pings, 0);
        end

        function f6PredictorRowsThatNewlySatisfyRequires(tc)
            % B4: predictor P requires [Q Tag]; Q = 5 at rate -1 (zero at 5); Tag is attached at t = 1 -> the row now
            % satisfies Requires and gets a prediction: removed at exactly 5. Detaching Tag cancels it.
            CT = @OceanAgents.World.Guard.ComponentType;
            D = OceanAgents.World.Guard.Dimension;
            tE = CT('Kind', "linear", 'Dimension', D.E());
            tTag = CT();
            mk = @(detachAt) OceanAgents.World.DoorSpec('Name', "bat", 'Owns', struct('Name', {"Q", "Tag"}, 'Type', {tE, tTag}), ...
                'Drive', "every-tick", 'Init', @(s) OceanAgents.World.Command.spawn(struct('Q', struct('Value', ...
                struct('Value', 5, 'Rate', -1), 'Type', tE))), 'Rules', [ ...
                OceanAgents.World.RuleSpec('Name', "P", 'Phase', "settle", 'Kind', "predictor", 'Requires', ["Q" "Tag"], ...
                'Reads', "Q", 'Predicts', "Q", 'DueEvent', "Zero", 'Fn', @(c) struct('Predictions', ...
                struct('Rows', c.Rows, 'Times', ceil(c.Now + c.Read.Q ./ -c.Read.Q_Rate)))), ...
                OceanAgents.World.RuleSpec('Name', "H", 'Phase', "resolve", 'Consumes', "Zero", 'Commands', "remove", ...
                'Fn', @(c) struct('Commands', arrayfun(@(e) OceanAgents.World.Command.remove(e.SourceEntity), c.Events))), ...
                OceanAgents.World.RuleSpec('Name', "Tagger", 'Phase', "act", 'Commands', ["attach" "detach"], ...
                'Fn', @(c) tagFn(c, tTag, detachAt))]);
            w = OceanAgents.World.World(mk(Inf));
            w.reset(0);
            w.run(4);
            tc.verifyTrue(w.Registry.isAlive(1));
            w.run(5);
            tc.verifyFalse(w.Registry.isAlive(1), 'removed at exactly 5');
            % detached at t = 2: the prediction is cancelled, the entity lives on
            w2 = OceanAgents.World.World(mk(2));
            w2.reset(0);
            w2.run(10);
            tc.verifyTrue(w2.Registry.isAlive(1));
            tc.verifyEqual(w2.Calendar.count(), 0, 'the prediction was cancelled');

            function out = tagFn(c, tTag, detachAt)
                out = struct();
                if c.Now == 1
                    out.Commands = OceanAgents.World.Command.attach(1, struct('Tag', struct('Value', 1, 'Type', tTag)));
                elseif c.Now == detachAt
                    out.Commands = OceanAgents.World.Command.detach(1, "Tag");
                end
            end
        end

        function f7OwnDoorProjectionHonoursVisibility(tc)
            % B7a: the vault's own projection may not read its hidden Secret (refused at construction, V6); its
            % "self" Mood shows only on the viewer's row; its public Pub on every row; linear reads get _Rate (F17).
            CT = @OceanAgents.World.Guard.ComponentType;
            tS = CT('Visibility', "hidden");  tM = CT('Visibility', "self");  tP = CT();
            tL = CT('Kind', "linear");
            spawn = @(sec, mood) OceanAgents.World.Command.spawn(struct('Secret', struct('Value', sec, 'Type', tS), ...
                'Mood', struct('Value', mood, 'Type', tM), 'Pub', struct('Value', 7, 'Type', tP), ...
                'Lvl', struct('Value', struct('Value', 3, 'Rate', -0.5), 'Type', tL)));
            mk = @(reads) OceanAgents.World.DoorSpec('Name', "vault", 'Owns', struct('Name', {"Secret", "Mood", "Pub", "Lvl"}, ...
                'Type', {tS, tM, tP, tL}), 'Drive', "never", 'Init', @(s) [spawn(111, 10), spawn(222, 20)], ...
                'Rules', OceanAgents.World.RuleSpec('Name', "Peek", 'Phase', "project", 'Kind', "projection", ...
                'Requires', "Pub", 'Reads', reads, 'Fn', @(c) struct('View', c.Read)));
            % B7a: the vault's own projection reading its hidden Secret is refused (it showed [111 222])
            tc.verifyError(@() OceanAgents.World.World(mk(["Secret" "Pub"])), 'OceanAgents:Guard:HiddenLeak');
            w = OceanAgents.World.World(mk(["Mood" "Pub" "Lvl"]));
            w.reset(0);
            v = w.project(1).vault_Peek;
            tc.verifyEqual(v.Mood, [10; 0], 'self: only the viewer''s own row');
            tc.verifyEqual(v.Pub, [7; 7]);
            tc.verifyEqual(v.Lvl_Rate, [-0.5; -0.5]);
        end

        function f10InitSpawnWithFutureAt(tc)
            % B9: an Init spawn requested At 50 is scheduled, not applied at reset.
            tI = OceanAgents.World.Guard.ComponentType();
            d = OceanAgents.World.DoorSpec('Name', "late", 'Owns', struct('Name', "I", 'Type', tI), 'Drive', "never", ...
                'Init', @(s) OceanAgents.World.Command.spawn(struct('I', struct('Value', 1, 'Type', tI)), 'At', 50));
            w = OceanAgents.World.World(d);
            w.reset(0);
            tc.verifyEqual(w.Registry.numAlive(), 0);
            tc.verifyEqual(w.NextTime, 50);
            r = w.run(100);
            tc.verifyEqual([r.Time], 50);
            tc.verifyEqual(r.Phases(1).CommandsApplied, 1);
            tc.verifyEqual(w.Registry.numAlive(), 1);
        end

        function f16CalendarStorageDoesNotGrowWithHistory(tc)
            % Released and cancelled commands are compacted out of the Calendar's storage.
            c = OceanAgents.World.Calendar();
            for k = 1:200
                cmd = OceanAgents.World.Command.remove(k).stamp("d/r", 0, k, [], "");
                c.scheduleCommand(cmd, k, 1, 1);
            end
            c.cancelEntity(1:50);
            for t = 51:150
                c.due(t, 1);
            end
            snap = c.snapshot();
            tc.verifyEqual(c.count(), 50);
            tc.verifyEqual(numel(snap.Cmds), 50, 'only pending commands are stored');
            items = c.peek();
            tc.verifyEqual([items.EntityID], 151:200);
            tc.verifyEqual(arrayfun(@(it) it.Command.Target, items), (151:200)');
        end
    end
end
