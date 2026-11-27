classdef WorldLoopTest < OceanTest.BaseTestCase
    % WORLDLOOPTEST  OceanAgents.World.World, the instant algorithm (W1_Kernel_Contract.md sections 5.5 and 7):
    % the hand-worked battery (skip straight to the predicted instant, removal at exactly t0 + 20, same as stepping),
    % K2 declared reads only, K4 event delivery and in-phase order independence, K6 commands (guards, IDs never
    % reused), K7 linear/clock values and rebasing, K8 predictor rounding, K10 no retroactive scheduling and
    % issue-time randomness, K13 per-door stream isolation, InstantReport fields, run/horizon semantics,
    % Undeclared outputs, pillars, and a (reported, not asserted) per-instant overhead measurement.

    methods (Static, Access = private)
        function L = L()
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
        end
        function w = world(doors, varargin)
            w = OceanAgents.World.World(doors, varargin{:});
        end
    end

    methods (Test)
        function batteryHandExample(tc)
            % Charge 10, Rate -0.5 from t0 = 0: 10 - 0.5 t = 0 at t = 20. With nothing else driven, the skipping
            % world processes the single instant 20 (instants 0..19 are skipped); the stepping world walks 0..20.
            L = tc.L();
            init = @(s) L.initSpawns({L.offer('Charge', struct('Value', 10, 'Rate', -0.5))});
            ws = tc.world(L.battery('Init', init), 'SkipIdle', true);
            ws.reset(1);
            tc.verifyEqual(ws.NextTime, 20, 'the only future work is the predicted Empty at 20');
            items = ws.Calendar.peek();
            empty = items([items.DueEvent] == "Empty");
            tc.verifyEqual([empty.Time empty.EntityID], [20 1]);
            check = items([items.DueEvent] ~= "Empty");        % skip-safe BT-1 check: first instant with Charge < 0
            tc.verifyEqual([check.Time check.EntityID], [21 1]);
            tc.verifyEqual(ws.Store.valueAt("Charge", 1, 13), 10 - 0.5 * 13);    % a function of time, never written
            rs = ws.run(100);
            tc.verifyEqual([rs.Time], 20, 'skipped straight from 0 to 20');
            tc.verifyEqual(rs(1).Phases(1).DueItems, 1);
            tc.verifyEqual(rs(1).Phases(1).RulesRun, "battery/Drain");
            tc.verifyEqual(rs(1).Phases(1).CommandsApplied, 1);
            tc.verifyEqual(rs(1).NextTime, Inf);
            tc.verifyFalse(ws.Registry.isAlive(1));
            % stepping: alive through 19, removed at exactly 20
            wt = tc.world(L.battery('Init', init), 'SkipIdle', false);
            wt.reset(1);
            rt = wt.run(19);
            tc.verifyEqual([rt.Time], 0:19);
            tc.verifyTrue(wt.Registry.isAlive(1));
            r20 = wt.advance();
            tc.verifyEqual([r20.Time r20.Phases(1).CommandsApplied], [20 1]);
            tc.verifyFalse(wt.Registry.isAlive(1));
            tc.verifyEqual(wt.Store.snapshot(), ws.Store.snapshot());
            tc.verifyEqual(wt.Registry.snapshot(), ws.Registry.snapshot());
        end

        function instantReportFields(tc)
            L = tc.L();
            w = tc.world([L.mover(), L.spawner(2, ["Position" "Velocity"])]);
            w.reset(3);
            r = w.advance();
            tc.verifyEqual(sort(string(fieldnames(r)))', ["NextTime" "Phases" "Time"]);
            tc.verifyEqual([r.Phases.Phase], ["resolve" "act" "settle" "project"]);
            tc.verifyEqual(sort(string(fieldnames(r.Phases)))', sort(["Phase" "DueItems" "RulesRun" "EventsEmitted" ...
                "CommandsApplied" "CommandsRefused" "GuardNoOps" "PredictionsUpdated"]));
            tc.verifyTrue(all(isfield(r.Phases(1).CommandsRefused, {'Kind', 'Reason', 'AxiomIDs'})));
            tc.verifyEqual(r.Time, 0);
            tc.verifyEqual(r.NextTime, 1);                 % mover is every-tick
        end

        function declaredReadsOnly(tc)
            % K2: ctx has exactly the documented fields and ctx.Read exactly the declared reads (+ _Has / _Rate).
            L = tc.L();
            probe = OceanAgents.World.RuleSpec('Name', "Probe", 'Phase', "act", 'Requires', "Position", ...
                'Reads', ["Position" "Charge" "Log"], 'Writes', "Log", 'Fn', @(ctx) struct('Writes', struct( ...
                'Resource', "Log", 'Values', {{string(fieldnames(ctx))', string(fieldnames(ctx.Read))'}})));
            spy = OceanAgents.World.DoorSpec('Name', "spy", 'Resources', struct('Log', {{}}), 'Rules', probe, ...
                'Drive', "every-tick");
            init = @(s) L.initSpawns({L.offer('Position', [1 2], 'Velocity', [0 0])});
            w = tc.world([L.mover('Init', init), L.battery(), spy]);
            w.reset(0);
            w.advance();
            log = w.Resources.Log;
            tc.verifyEqual(sort(log{1}), sort(["Now" "Phase" "Door" "Rows" "IDs" "Read" "Events" "Stream"]));
            tc.verifyEqual(sort(log{2}), sort(["Position" "Position_Has" "Charge" "Charge_Has" "Charge_Rate" "Log"]));
        end

        function eventDeliveryAcrossPhases(tc)
            % K4: an event emitted in "act" at t reaches a "settle" consumer at t, and "act" / "resolve" consumers at
            % t + 1 - never its own phase at the same instant.
            L = tc.L();
            w = tc.world([L.echo(), L.listener()]);
            w.reset(0);
            w.run(4);
            R = w.Resources;
            tc.verifyEqual(R.LaterLog, [(0:4)' (0:4)'], 'settle sees act events at the same instant');
            tc.verifyEqual(R.SameLog, [(1:4)' (0:3)'], 'act sees act events one instant later');
            tc.verifyEqual(R.EarlierLog, [(1:4)' (0:3)'], 'resolve sees them at the next instant');
        end

        function inPhaseRuleOrderIrrelevant(tc)
            % K4: rules read the state as at the start of their phase, so their order within the phase is irrelevant.
            L = tc.L();
            a = tc.world(L.swap([1 2]));
            b = tc.world(L.swap([2 1]));
            a.reset(0); b.reset(0);
            a.run(5); b.run(5);
            tc.verifyEqual(a.Resources, b.Resources);
            tc.verifyEqual([a.Resources.X a.Resources.Y], [6 5]);   % Y copies X from the start of the phase
        end

        function commandsGuardsAndIDs(tc)
            % K6: a guarded no-op is recorded; a removed entity's ID is never reused (not even after reset).
            L = tc.L();
            T = L.types();
            issued = OceanAgents.World.RuleSpec('Name', "Plan", 'Phase', "resolve", 'Commands', ["spawn" "remove"], ...
                'Fn', @(ctx) planFn(ctx, T));
            planner = OceanAgents.World.DoorSpec('Name', "planner", 'Rules', issued, 'Drive', "every-tick");
            w = tc.world([L.mover(), planner]);
            w.reset(0);
            r0 = w.advance();                                % t=0: spawn one entity
            tc.verifyEqual(r0.Phases(1).CommandsApplied, 1);
            tc.verifyEqual(w.Registry.LastID, 1);
            r1 = w.advance();                                % t=1: remove 1, and a guarded remove of 999 (no-op)
            tc.verifyEqual(r1.Phases(1).CommandsApplied, 1);
            tc.verifyEqual(r1.Phases(1).GuardNoOps, 1);
            tc.verifyFalse(w.Registry.isAlive(1));
            r2 = w.advance();                                % t=2: spawn again -> ID 2, never 1
            tc.verifyEqual(r2.Phases(1).CommandsApplied, 1);
            tc.verifyTrue(w.Registry.isAlive(2));
            r3 = w.advance();                                % t=3: unguarded remove of a dead entity -> refused
            tc.verifyEqual(r3.Phases(1).CommandsRefused(1).Reason, "not-alive");
            w.reset(0);
            w.advance();
            tc.verifyEqual(w.Registry.LastID, 3, 'IDs continue across episodes');

            function out = planFn(ctx, T)
                out = struct('Commands', OceanAgents.World.Command.empty(1, 0));
                switch ctx.Now
                    case {0, 2}
                        out.Commands = OceanAgents.World.Command.spawn(struct( ...
                            'Position', struct('Value', [0 0], 'Type', T.Position), ...
                            'Velocity', struct('Value', [1 0], 'Type', T.Velocity)));
                    case 1
                        out.Commands = [OceanAgents.World.Command.remove(1), ...
                            OceanAgents.World.Command.remove(999, 'Guard', @(v) v.TargetAlive)];
                    case 3
                        out.Commands = OceanAgents.World.Command.remove(1);
                end
            end
        end

        function linearAndClockValuesAndRebase(tc)
            % K7: Charge 10 at rate -0.5 from 0; at t = 4 the rate is rebased to -1 (value kept: 8).
            % Hand values: Charge(3) = 8.5, Charge(4) = 8, Charge(6) = 6, empty at 4 + 8 = 12. Born = -3: age(t) = t + 3.
            L = tc.L();
            init = @(s) L.initSpawns({L.offer('Charge', struct('Value', 10, 'Rate', -0.5), 'Born', -3)});
            w = tc.world([L.battery('RebaseAt', 4, 'Init', init), L.ager('Init', [])]);
            w.reset(0);
            w.run(3);
            tc.verifyEqual(w.Store.valueAt("Charge", 1, w.Now), 8.5);
            tc.verifyEqual(w.Store.valueAt("Born", 1, w.Now), 6);
            w.run(4);
            raw = w.Store.raw("Charge");
            tc.verifyEqual([raw.Base(1) raw.Rate(1) raw.T0(1)], [8 -1 4], 'rebased at 4');
            w.run(6);
            tc.verifyEqual(w.Store.valueAt("Charge", 1, 6), 6);
            items = w.Calendar.peek();
            tc.verifyEqual(items([items.DueEvent] == "Empty").Time, 12, 're-predicted after the rebase');
            w.run(11);
            tc.verifyTrue(w.Registry.isAlive(1));
            w.run(12);
            tc.verifyFalse(w.Registry.isAlive(1));
        end

        function predictorRounding(tc)
            % K8: Charge 10 at rate -0.3 crosses zero at 33.33...; the convention is ceil, so Empty is due at 34
            % (charge there -0.2) and the entity is removed at exactly 34. BT-1 (Charge >= 0) is not violated: the
            % monitor sees the post-command state, in which the entity is gone.
            L = tc.L();
            init = @(s) L.initSpawns({L.offer('Charge', struct('Value', 10, 'Rate', -0.3))});
            w = tc.world(L.battery('Init', init));
            w.reset(0);
            tc.verifyEqual(w.NextTime, 34);
            r = w.run(100);
            tc.verifyEqual([r.Time], 34);
            tc.verifyFalse(w.Registry.isAlive(1));
            tc.verifyEqual(ceil(10 / 0.3), 34);
        end

        function noRetroactiveCommands(tc)
            % K10: a rule scheduling a command before the current point is a fail-stop (World:Retroactive).
            L = tc.L();
            bad = OceanAgents.World.RuleSpec('Name', "Past", 'Phase', "act", 'Commands', "remove", ...
                'Fn', @(ctx) struct('Commands', OceanAgents.World.Command.remove(1, 'At', max(ctx.Now - 1, 0), ...
                'Phase', "resolve")));
            door = OceanAgents.World.DoorSpec('Name', "timetraveller", 'Rules', bad, 'Drive', "every-tick");
            w = tc.world([L.mover(), door]);
            w.reset(0);
            tc.verifyError(@() w.advance(), 'OceanAgents:World:Retroactive');
            tc.verifyEqual(w.Status, "Failed");
        end

        function randomnessDrawnAtIssue(tc)
            % K10: future spawns issued at t = 0 (for t = 9, 5, 7, in that issue order) carry positions drawn at
            % issue time from the planner's stream; release order (5, 7, 9) does not change them.
            L = tc.L();
            T = L.types();
            plan = OceanAgents.World.RuleSpec('Name', "Plan", 'Phase', "settle", 'Commands', "spawn", ...
                'Fn', @(ctx) planFn(ctx, T));
            planner = OceanAgents.World.DoorSpec('Name', "planner", 'Rules', plan, 'Drive', "every-k", 'DriveK', 1000);
            w = tc.world([L.mover(), planner]);
            w.reset(42);
            w.run(10);
            ref = OceanAgents.Runtime.StreamRegistry(42).stream("door:planner");
            drawn = [randi(ref, [0 100], 1, 2); randi(ref, [0 100], 1, 2); randi(ref, [0 100], 1, 2)];  % issue order 9,5,7
            % entity k spawned at t_k then moved with velocity 0: positions unchanged
            ids = [3 1 2];                                   % issued for 9, 5, 7; released 5, 7, 9 -> IDs 1, 2, 3
            for k = 1:3
                tc.verifyEqual(w.Store.valueAt("Position", w.Registry.rowsOf(ids(k)), w.Now), drawn(k, :));
            end

            function out = planFn(ctx, T)
                cmds = OceanAgents.World.Command.empty(1, 0);
                for at = [9 5 7]
                    o = struct('Position', struct('Value', randi(ctx.Stream, [0 100], 1, 2), 'Type', T.Position), ...
                        'Velocity', struct('Value', [0 0], 'Type', T.Velocity));
                    cmds(end + 1) = OceanAgents.World.Command.spawn(o, 'At', at, 'Phase', "resolve"); %#ok<AGROW>
                end
                out = struct('Commands', cmds);
            end
        end

        function streamIsolationAcrossDoors(tc)
            % K13: adding a door that draws every tick does not change another door's draws.
            L = tc.L();
            noise = OceanAgents.World.RuleSpec('Name', "Noise", 'Phase', "resolve", 'Reads', "N", 'Writes', "N", ...
                'Fn', @(ctx) struct('Writes', struct('Resource', "N", 'Values', ctx.Read.N + rand(ctx.Stream, 1, 3))));
            noisy = OceanAgents.World.DoorSpec('Name', "noisy", 'Resources', struct('N', zeros(1, 3)), 'Rules', noise, ...
                'Drive', "every-tick");
            a = tc.world([L.spawner(3, ["Position" "Velocity"]), L.mover('Kick', true)]);
            b = tc.world([L.spawner(3, ["Position" "Velocity"]), noisy, L.mover('Kick', true)]);
            a.reset(9); b.reset(9);
            a.run(40); b.run(40);
            tc.verifyEqual(b.Store.snapshot(), a.Store.snapshot());
            tc.verifyEqual(b.Registry.snapshot(), a.Registry.snapshot());
            tc.verifyNotEqual(b.Resources.N, zeros(1, 3));
        end

        function injectAndRunSemantics(tc)
            L = tc.L();
            w = tc.world(L.counter(), 'Injectable', "Ping");
            w.reset(0);
            tc.verifyEqual(w.NextTime, Inf, 'nothing to do');
            w.inject([OceanAgents.World.Event.make("Ping") OceanAgents.World.Event.make("Ping")]);
            tc.verifyEqual(w.NextTime, 0, 'nothing processed or covered yet: the first instant');
            r = w.run(50);
            tc.verifyEqual([r.Time], 0);
            tc.verifyEqual(w.Resources.Pings, 2);
            w.inject(OceanAgents.World.Event.make("Ping"));
            r = w.run(80);
            tc.verifyEqual([r.Time], 51, 'run(50) covered time through 50: delivered at 51');
            tc.verifyEqual(w.Resources.Pings, 3);
            tc.verifyError(@() w.inject(OceanAgents.World.Event.make("Pong")), 'OceanAgents:World:UnknownEvent');
            tc.verifyError(@() w.advance(), 'OceanAgents:World:InvalidState');   % no future work
            % horizon: a stepping world stops at the horizon
            h = tc.world(L.mover(), 'Horizon', 5, 'SkipIdle', false);
            h.reset(0);
            r = h.run(100);
            tc.verifyEqual([r.Time], 0:5);
            tc.verifyEqual(h.NextTime, Inf);
            % advance before any reset
            fresh = tc.world(L.mover());
            tc.verifyError(@() fresh.advance(), 'OceanAgents:World:InvalidState');
        end

        function undeclaredOutputs(tc)
            L = tc.L();
            mk = @(fn) OceanAgents.World.DoorSpec('Name', "rogue", 'Resources', struct('R', 0), ...
                'Rules', OceanAgents.World.RuleSpec('Name', "Rogue", 'Phase', "act", 'Requires', "Position", ...
                'Reads', "Position", 'Writes', "R", 'Emits', "Blip", 'Fn', fn), 'Drive', "every-tick");
            sink = OceanAgents.World.DoorSpec('Name', "sink", 'Rules', OceanAgents.World.RuleSpec('Name', "Eat", ...
                'Phase', "settle", 'Consumes', "Blip", 'Fn', @(ctx) struct()), 'Drive', "on-event");
            init = @(s) L.initSpawns({L.offer('Position', [0 0], 'Velocity', [1 1])});
            cases = {
                @(ctx) struct('Writes', struct('Component', "Velocity", 'Rows', ctx.Rows, 'Values', [0 0]))   % not its write
                @(ctx) struct('Events', OceanAgents.World.Event.make("Other"))                             % not its emit
                @(ctx) struct('Commands', OceanAgents.World.Command.remove(1))                              % not its command
                @(ctx) struct('Predictions', struct('Rows', ctx.Rows, 'Times', 5))                         % a rule predicts
                };
            for k = 1:numel(cases)
                w = tc.world([L.mover('Init', init), mk(cases{k}), sink]);
                w.reset(0);
                tc.verifyError(@() w.advance(), 'OceanAgents:World:Undeclared', sprintf('case %d', k));
            end
        end

        function pillars(tc)
            L = tc.L();
            w = tc.world(L.mover());
            tc.verifyFalse(w.getMetadata().IsSymbolic);
            tc.verifyEqual(w.getPipelineSpec().ComponentName, "World");
        end

        function perInstantOverhead(tc)
            % Reported, not asserted: cost of one every-tick instant of the mover door with n entities (Init
            % spawns go through admission one by one, reported separately).
            L = tc.L();
            for n = [1e3 1e4]
                offers = arrayfun(@(k) L.offer('Position', [k 0], 'Velocity', [1 0]), 1:n, 'UniformOutput', false);
                w = tc.world(L.mover('Init', @(s) L.initSpawns(offers)));
                t0 = tic;
                w.reset(0);
                tInit = toc(t0);
                w.advance();                                 % warm-up
                t1 = tic;
                w.run(w.Now + 20);
                perInstant = toc(t1) / 20;
                tc.verifyEqual(w.Registry.numAlive(), n);
                tc.log(1, sprintf('mover, %d entities: reset with %d admitted spawns %.2f s; %.2f ms per instant', ...
                    n, n, tInit, 1e3 * perInstant));
            end
        end
    end
end
