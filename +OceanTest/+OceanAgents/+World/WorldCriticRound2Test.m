classdef WorldCriticRound2Test < OceanTest.BaseTestCase
    % WORLDCRITICROUND2TEST  Regression tests for the W1 critic round 2 findings owned by the World loop
    % (W1_Kernel_Contract.md section 12, "Critic round 2 decisions"), reproducing the critic's scenarios
    % (criticW1r2/probe_new.m N1, N5, N6; probe_k4.m), plus the post-critic items F-R3-3..5 (criticW1r3).
    %
    % K4 as narrowed: permuting rules within a phase changes nothing about what rules READ or which EVENTS they see;
    % the ORDER in which commands are admitted and events delivered follows the declared door and rule rank, which is
    % semantic (like door rank) - the last two tests show both halves.

    methods (Static, Access = private)
        function out = onlyAt(c, at, cmd)
            out = struct();
            if c.Now == at, out.Commands = cmd; end
        end

        function d = ownerDoor(tV)
            % Owns V (value 1 at reset); every act it writes V + 1.
            d = OceanAgents.World.DoorSpec('Name', "own", 'Owns', struct('Name', "V", 'Type', tV), 'Membership', "V", ...
                'Drive', "every-tick", 'Init', @(s) OceanAgents.World.Command.spawn(struct('V', struct('Value', 1, 'Type', tV))), ...
                'Rules', OceanAgents.World.RuleSpec('Name', "Inc", 'Phase', "act", 'Requires', "V", 'Reads', "V", ...
                'Writes', "V", 'Fn', @(c) struct('Writes', struct('Component', "V", 'Rows', c.Rows, 'Values', c.Read.V + 1))));
        end
    end

    methods (Test)
        function n1OnlyTheOwnerAttachesOrDetaches(tc)
            % N1: a non-owner's detach(1, "V") + attach(1, V = 999) in one sync, in either order: both refused
            % ("not-owner"); V stays owner-driven (4 after t = 0..2).
            tV = OceanAgents.World.Guard.ComponentType();
            orders = {@(d, a) [d a], @(d, a) [a d]};
            for k = 1:2
                cmds = orders{k}(OceanAgents.World.Command.detach(1, "V"), ...
                    OceanAgents.World.Command.attach(1, struct('V', struct('Value', 999, 'Type', tV))));
                med = OceanAgents.World.DoorSpec('Name', "meddler", 'Drive', "every-tick", 'Rules', ...
                    OceanAgents.World.RuleSpec('Name', "Reset", 'Phase', "act", 'Commands', ["attach" "detach"], ...
                    'Fn', @(c) tc.onlyAt(c, 2, cmds)));
                w = OceanAgents.World.World([tc.ownerDoor(tV) med]);
                w.reset(0);
                r = w.run(2);
                act = r(end).Phases(2);
                tc.verifyEqual(act.CommandsApplied, 0, sprintf('order %d', k));
                tc.verifyEqual([act.CommandsRefused.Reason], ["not-owner" "not-owner"], sprintf('order %d', k));
                tc.verifyEqual(w.Store.valueAt("V", w.Registry.rowsOf(1), 2), 4, sprintf('order %d', k));
            end
            % spawn may offer any door's components (a new entity has no values to overwrite)
            sp = OceanAgents.World.DoorSpec('Name', "maker", 'Drive', "every-k", 'DriveK', 100, 'Rules', ...
                OceanAgents.World.RuleSpec('Name', "Make", 'Phase', "act", 'Commands', "spawn", 'Fn', @(c) ...
                struct('Commands', OceanAgents.World.Command.spawn(struct('V', struct('Value', 7, 'Type', tV))))));
            w = OceanAgents.World.World([tc.ownerDoor(tV) sp]);
            w.reset(0);
            r = w.run(0);
            tc.verifyEqual(r.Phases(2).CommandsApplied, 1);
            % the owner's own detach is allowed
            det = OceanAgents.World.DoorSpec('Name', "own", 'Owns', struct('Name', "V", 'Type', tV), 'Drive', "every-tick", ...
                'Init', @(s) OceanAgents.World.Command.spawn(struct('V', struct('Value', 1, 'Type', tV))), ...
                'Rules', OceanAgents.World.RuleSpec('Name', "Drop", 'Phase', "act", 'Commands', "detach", ...
                'Fn', @(c) tc.onlyAt(c, 1, OceanAgents.World.Command.detach(1, "V"))));
            w = OceanAgents.World.World(det);
            w.reset(0);
            w.run(1);
            tc.verifyFalse(w.Store.has("V", w.Registry.rowsOf(1)));
        end

        function n5DeliveryOrderFollowsRuleRank(tc)
            % N5: rules First (rank 1, SourceEntity 9) and Second (rank 2, SourceEntity 3) emit E in "act"; the
            % settle consumer receives them in rule-rank order [1 2] (the key has rule rank before source entity).
            R = @OceanAgents.World.RuleSpec;
            em = OceanAgents.World.DoorSpec('Name', "em", 'Resources', struct('Order', zeros(1, 0)), 'Drive', "every-k", ...
                'DriveK', 100, 'Rules', [ ...
                R('Name', "First", 'Phase', "act", 'Emits', "E", 'Fn', @(c) struct('Events', ...
                OceanAgents.World.Event.make("E", struct('who', 1), 'SourceEntity', 9))), ...
                R('Name', "Second", 'Phase', "act", 'Emits', "E", 'Fn', @(c) struct('Events', ...
                OceanAgents.World.Event.make("E", struct('who', 2), 'SourceEntity', 3))), ...
                R('Name', "Rec", 'Phase', "settle", 'Consumes', "E", 'Reads', "Order", 'Writes', "Order", ...
                'Fn', @(c) struct('Writes', struct('Resource', "Order", 'Values', ...
                [c.Read.Order arrayfun(@(e) e.Payload.who, c.Events)])))]);
            w = OceanAgents.World.World(em);
            w.reset(0);
            w.run(0);
            tc.verifyEqual(w.Resources.Order, [1 2]);
        end

        function n6aInjectIsAtomic(tc)
            % N6a: inject([Ping Pong]) with Pong not injectable: error, nothing queued, NextTime unchanged, no sequence
            % number consumed; the world then runs normally.
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            tk = OceanAgents.World.DoorSpec('Name', "tk", 'Drive', "every-k", 'DriveK', 7, 'Rules', ...
                OceanAgents.World.RuleSpec('Name', "Nop", 'Phase', "act", 'Fn', @(c) struct()));
            w = OceanAgents.World.World([L.counter() tk], 'Injectable', "Ping");
            w.reset(0);
            w.run(3);
            next = w.NextTime;
            tc.verifyError(@() w.inject([OceanAgents.World.Event.make("Ping"), OceanAgents.World.Event.make("Pong")]), ...
                'OceanAgents:World:UnknownEvent');
            tc.verifyEqual(w.NextTime, next);
            tc.verifyEqual(w.Status, "Running");
            w.run(20);
            tc.verifyEqual(w.Resources.Pings, 0, 'the valid Ping of the refused call was not queued');
            % a reference world that never saw the refused call is identical
            ref = OceanAgents.World.World([L.counter() tk], 'Injectable', "Ping");
            ref.reset(0);
            ref.run(3);
            ref.run(20);
            tc.verifyEqual(w.Store.snapshot(), ref.Store.snapshot());
            tc.verifyEqual(w.Resources, ref.Resources);
            w.inject(OceanAgents.World.Event.make("Ping"));
            w.run(30);
            tc.verifyEqual(w.Resources.Pings, 1);
        end

        function n6cRunInfAndTEndValidation(tc)
            % N6c: run(Inf) stops at no future work (skipping) and a later inject still works; run(tEnd) needs an
            % integer or Inf; a stepping world without a horizon cannot run(Inf).
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            w = OceanAgents.World.World(L.counter(), 'Injectable', "Ping");
            w.reset(0);
            r = w.run(Inf);                                  % quiescent: returns at once
            tc.verifyEmpty(r);
            w.inject(OceanAgents.World.Event.make("Ping"));
            r = w.run(Inf);                                  % processes the delivery, then stops
            tc.verifyEqual([r.Time], 0);
            tc.verifyEqual(w.Resources.Pings, 1);
            w.inject(OceanAgents.World.Event.make("Ping"));
            tc.verifyEqual(w.NextTime, 1, 'after run(Inf) the next instant follows the last processed one');
            tc.verifyError(@() w.run(10.5), 'OceanAgents:World:BadValue');
            tc.verifyError(@() w.run(-1), 'OceanAgents:World:BadValue');
            tc.verifyError(@() w.run(NaN), 'OceanAgents:World:BadValue');
            step = OceanAgents.World.World(L.counter(), 'SkipIdle', false, 'Injectable', "Ping");
            step.reset(0);
            tc.verifyError(@() step.run(Inf), 'OceanAgents:World:BadValue');
            hz = OceanAgents.World.World(L.counter(), 'SkipIdle', false, 'Horizon', 4, 'Injectable', "Ping");
            hz.reset(0);
            r = hz.run(Inf);
            tc.verifyEqual([r.Time], 0:4, 'with a horizon, run(Inf) stops there');
        end

        function notOwnerAppliesToCalendarReleasedCommands(tc)
            % Post-critic F-R3-5 (probe_r3 R1): door a schedules at t = 0, for t = 3: attach its own X, attach b's Y,
            % detach b's V, remove b's entity 2. At t = 3 (released from the Calendar, skipping or stepping): the own
            % attach and the remove are applied; the Y attach and the V detach are refused "not-owner".
            R = @OceanAgents.World.RuleSpec;
            tV = OceanAgents.World.Guard.ComponentType();
            b = OceanAgents.World.DoorSpec('Name', "b", 'Owns', struct('Name', {"V", "Y"}, 'Type', {tV, tV}), ...
                'Membership', "V", 'Drive', "never", 'Init', @(s) [ ...
                OceanAgents.World.Command.spawn(struct('V', struct('Value', 1, 'Type', tV))), ...
                OceanAgents.World.Command.spawn(struct('V', struct('Value', 2, 'Type', tV)))]);
            plan = [OceanAgents.World.Command.attach(1, struct('X', struct('Value', 5, 'Type', tV)), 'At', 3), ...
                OceanAgents.World.Command.attach(1, struct('Y', struct('Value', 5, 'Type', tV)), 'At', 3), ...
                OceanAgents.World.Command.detach(1, "V", 'At', 3), OceanAgents.World.Command.remove(2, 'At', 3)];
            a = OceanAgents.World.DoorSpec('Name', "a", 'Owns', struct('Name', "X", 'Type', tV), 'Drive', "every-k", ...
                'DriveK', 100, 'Rules', R('Name', "Plan", 'Phase', "act", 'Commands', ["attach" "detach" "remove"], ...
                'Fn', @(c) tc.onlyAt(c, 0, plan)));
            for skip = [true false]
                w = OceanAgents.World.World([b a], 'SkipIdle', skip);
                w.reset(0);
                r = w.run(5);
                a3 = r([r.Time] == 3).Phases(2);
                label = sprintf('SkipIdle %d', skip);
                tc.verifyEqual(a3.CommandsApplied, 2, label);
                tc.verifyEqual([a3.CommandsRefused.Reason], ["not-owner" "not-owner"], label);
                tc.verifyEqual(unique([a3.CommandsRefused.Issuer]), "a/Plan", label);
                tc.verifyEqual([w.Store.has("X", 1) w.Store.has("Y", 1) w.Store.has("V", 1)], [true false true], label);
                tc.verifyEqual(w.Store.valueAt("V", 1, 5), 1, label);
                tc.verifyFalse(w.Registry.isAlive(2), label);
            end
        end

        function injectEmptyIsANoOp(tc)
            % Post-critic F-R3-3: inject([]) changes nothing (also at the horizon, where a real inject is refused).
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            w = OceanAgents.World.World(L.counter(), 'Injectable', "Ping", 'Horizon', 5);
            w.reset(0);
            next = w.NextTime;
            w.inject(OceanAgents.World.Event.empty(1, 0));
            w.inject([]);
            tc.verifyEqual(w.NextTime, next);
            w.run(10);
            tc.verifyEqual(w.Resources.Pings, 0);
            w.inject([]);                                    % at the horizon: still a no-op, not an error
            tc.verifyEqual(w.Status, "Running");
        end

        function horizonMustBeAnIntegerTick(tc)
            % Post-critic F-R3-4 (probe_r6): a fractional Horizon is refused at construction.
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            tc.verifyError(@() OceanAgents.World.World(L.mover(), 'Horizon', 10.5), 'OceanAgents:World:BadValue');
            w = OceanAgents.World.World(L.mover(), 'Horizon', 10);
            tc.verifyEqual(w.Horizon, 10);
        end

        function k4CommandAdmissionFollowsRuleRank(tc)
            % probe_k4: door g owns A, B, C (members need all three; static G-1: B + C <= 5); entity has A only. Rules
            % X (attach B = 3) and Y (attach C = 3) run in the same phase: whichever is ranked first is admitted, the
            % second completes membership and is refused by G-1. Rule rank is declared operator order (semantic).
            t = OceanAgents.World.Guard.ComponentType();
            att = @(nm) OceanAgents.World.Command.attach(1, struct(nm, struct('Value', 3, 'Type', t)));
            X = OceanAgents.World.RuleSpec('Name', "X", 'Phase', "act", 'Commands', "attach", ...
                'Fn', @(c) tc.onlyAt(c, 0, att("B")));
            Y = OceanAgents.World.RuleSpec('Name', "Y", 'Phase', "act", 'Commands', "attach", ...
                'Fn', @(c) tc.onlyAt(c, 0, att("C")));
            has = zeros(2, 2);
            orders = {[X Y], [Y X]};
            for k = 1:2
                g = OceanAgents.World.DoorSpec('Name', "g", 'Owns', struct('Name', {"A", "B", "C"}, 'Type', {t, t, t}), ...
                    'Membership', ["A" "B" "C"], 'Drive', "every-k", 'DriveK', 100, 'Rules', orders{k}, ...
                    'Axioms', OceanAgents.World.Guard.AxiomSpec('ID', "G-1", 'Mode', "static", 'Reads', ["B" "C"], ...
                    'Predicate', @(v) v.B + v.C <= 5), ...
                    'Init', @(s) OceanAgents.World.Command.spawn(struct('A', struct('Value', 1, 'Type', t))));
                w = OceanAgents.World.World(g);
                w.reset(0);
                r = w.run(0);
                has(k, :) = [w.Store.has("B", 1) w.Store.has("C", 1)];
                tc.verifyEqual(r.Phases(2).CommandsRefused(1).AxiomIDs, "G-1");
            end
            tc.verifyEqual(has, [1 0; 0 1], 'the first-ranked rule''s attach is admitted');
        end

        function k4RuleOrderChangesNoReadsAndNoEventVisibility(tc)
            % Swapping two emitting rules of one phase: every consumer sees the same set of events (only their
            % delivery order, by rule rank, differs), and the reads-only door computes the same values.
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            R = @OceanAgents.World.RuleSpec;
            e1 = R('Name', "E1", 'Phase', "act", 'Emits', "E", 'Fn', @(c) struct('Events', ...
                OceanAgents.World.Event.make("E", struct('who', 1))));
            e2 = R('Name', "E2", 'Phase', "act", 'Emits', "E", 'Fn', @(c) struct('Events', ...
                OceanAgents.World.Event.make("E", struct('who', 2))));
            rec = R('Name', "Rec", 'Phase', "settle", 'Consumes', "E", 'Reads', "Seen", 'Writes', "Seen", ...
                'Fn', @(c) struct('Writes', struct('Resource', "Seen", 'Values', ...
                [c.Read.Seen sort(arrayfun(@(e) e.Payload.who, c.Events))])));
            seen = cell(1, 2);
            orders = {[e1 e2 rec], [e2 e1 rec]};
            for k = 1:2
                d = OceanAgents.World.DoorSpec('Name', "em", 'Resources', struct('Seen', zeros(1, 0)), ...
                    'Drive', "every-tick", 'Rules', orders{k});
                w = OceanAgents.World.World([d L.swap([1 2])]);
                w.reset(0);
                w.run(4);
                seen{k} = w.Resources;
            end
            tc.verifyEqual(seen{2}, seen{1});
            tc.verifyEqual(seen{1}.Seen, repmat([1 2], 1, 5));
        end
    end
end
