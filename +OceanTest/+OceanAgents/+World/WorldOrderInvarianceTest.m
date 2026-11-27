classdef WorldOrderInvarianceTest < OceanTest.BaseTestCase
    % WORLDORDERINVARIANCETEST  K5 as redefined in W1_Kernel_Contract.md section 12: results do not depend on
    % NON-SEMANTIC orders - the field order of spawn offers, of event payloads and of door resources, and the order in
    % which items are inserted into the Calendar through the loop.
    % Door rank (the order doors are listed) is NOT non-semantic: it is the declared operator order of the world
    % (tie-breaks, ID allocation among same-instant spawns), so permuting doors is allowed to change results. The last
    % test shows exactly that - different entity IDs - while an ID-insensitive comparison of the entities agrees.

    methods (Static, Access = private)
        function L = L()
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
        end
    end

    methods (Test)
        function offerFieldOrder(tc)
            L = tc.L();
            base = {L.offer('Position', [1 2], 'Velocity', [1 -1], 'Charge', struct('Value', 7, 'Rate', -0.25)), ...
                L.offer('Position', [5 5], 'Velocity', [0 1], 'Charge', struct('Value', 3, 'Rate', -0.5))};
            flipped = cellfun(@(o) orderfields(o, {'Charge', 'Velocity', 'Position'}), base, 'UniformOutput', false);
            a = OceanAgents.World.World([L.mover('Init', @(s) L.initSpawns(base)), L.battery()]);
            b = OceanAgents.World.World([L.mover('Init', @(s) L.initSpawns(flipped)), L.battery()]);
            a.reset(1); b.reset(1);
            a.run(30); b.run(30);
            tc.verifyEqual(b.Store.snapshot(), a.Store.snapshot());
            tc.verifyEqual(b.Registry.snapshot(), a.Registry.snapshot());
            tc.verifyEqual(b.Calendar.peek(), a.Calendar.peek());
        end

        function payloadFieldOrder(tc)
            use = OceanAgents.World.RuleSpec('Name', "Use", 'Phase', "resolve", 'Consumes', "Order", 'Reads', "Sum", ...
                'Writes', "Sum", 'Fn', @(ctx) struct('Writes', struct('Resource', "Sum", 'Values', ...
                ctx.Read.Sum + sum(arrayfun(@(e) 10 * e.Payload.a + e.Payload.b, ctx.Events)))));
            door = OceanAgents.World.DoorSpec('Name', "user", 'Resources', struct('Sum', 0), 'Rules', use, 'Drive', "on-event");
            ra = runWith(door, struct('a', 3, 'b', 4));
            rb = runWith(door, struct('b', 4, 'a', 3));
            tc.verifyEqual(rb, ra);
            tc.verifyEqual(ra.Sum, 34);

            function res = runWith(door, payload)
                w = OceanAgents.World.World(door, 'Injectable', "Order");
                w.reset(0);
                w.inject(OceanAgents.World.Event.make("Order", payload));
                w.run(5);
                res = w.Resources;
            end
        end

        function resourceFieldOrder(tc)
            step = OceanAgents.World.RuleSpec('Name', "Step", 'Phase', "act", 'Reads', ["A" "B"], 'Writes', ["A" "B"], ...
                'Fn', @(ctx) struct('Writes', [struct('Resource', "A", 'Values', ctx.Read.A + ctx.Read.B), ...
                struct('Resource', "B", 'Values', ctx.Read.A)]));
            mk = @(res) OceanAgents.World.DoorSpec('Name', "fib", 'Resources', res, 'Rules', step, 'Drive', "every-tick");
            a = OceanAgents.World.World(mk(struct('A', 1, 'B', 0)));
            b = OceanAgents.World.World(mk(struct('B', 0, 'A', 1)));
            a.reset(0); b.reset(0);
            a.run(10); b.run(10);
            tc.verifyEqual(orderfields(b.Resources), orderfields(a.Resources));
            tc.verifyEqual(a.Resources.A, 144);              % Fibonacci after 11 steps
        end

        function calendarInsertionOrderThroughTheLoop(tc)
            % Two doors each schedule a spawn for t = 10; in one world X inserts first (t = 0) and Y later (t = 3),
            % in the other the reverse. The released order follows the tie-break key (door rank), not insertion order.
            L = tc.L();
            T = L.types();
            mk = @(name, issueAt, pos) OceanAgents.World.DoorSpec('Name', name, 'Rules', OceanAgents.World.RuleSpec( ...
                'Name', "Plan", 'Phase', "settle", 'Commands', "spawn", 'Fn', @(ctx) planFn(ctx, issueAt, pos, T)), ...
                'Drive', "every-tick");
            a = OceanAgents.World.World([L.mover(), mk("x", 0, [1 1]), mk("y", 3, [2 2])]);
            b = OceanAgents.World.World([L.mover(), mk("x", 3, [1 1]), mk("y", 0, [2 2])]);
            a.reset(0); b.reset(0);
            a.run(12); b.run(12);
            tc.verifyEqual(b.Store.snapshot(), a.Store.snapshot());
            tc.verifyEqual(b.Registry.snapshot(), a.Registry.snapshot());
            tc.verifyEqual(a.Store.valueAt("Position", a.Registry.rowsOf([1; 2]), 12), [1 1; 2 2], 'x (rank 2) first');

            function out = planFn(ctx, issueAt, pos, T)
                out = struct();
                if ctx.Now == issueAt
                    out.Commands = OceanAgents.World.Command.spawn(struct('Position', struct('Value', pos, ...
                        'Type', T.Position), 'Velocity', struct('Value', [0 0], 'Type', T.Velocity)), 'At', 10, ...
                        'Phase', "resolve");
                end
            end
        end

        function doorRankIsSemanticIDsMayDiffer(tc)
            % Two spawner doors, both spawning at reset: swapping their ranks swaps which entity gets which ID.
            % Allowed (door rank is the declared operator order); the entities themselves are the same.
            L = tc.L();
            mk = @(name, pos) OceanAgents.World.DoorSpec('Name', name, 'Drive', "never", ...
                'Init', @(s) L.initSpawns({L.offer('Position', pos, 'Velocity', [0 0])}));
            a = OceanAgents.World.World([L.mover(), mk("p", [1 1]), mk("q", [9 9])]);
            b = OceanAgents.World.World([L.mover(), mk("q", [9 9]), mk("p", [1 1])]);
            a.reset(0); b.reset(0);
            pa = a.Store.valueAt("Position", a.Registry.rowsOf([1; 2]), 0);
            pb = b.Store.valueAt("Position", b.Registry.rowsOf([1; 2]), 0);
            tc.verifyNotEqual(pa, pb, 'IDs follow door rank');
            tc.verifyEqual(sortrows(pa), sortrows(pb), 'ID-insensitive: the same entities');
        end
    end
end
