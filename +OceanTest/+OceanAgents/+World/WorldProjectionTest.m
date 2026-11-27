classdef WorldProjectionTest < OceanTest.BaseTestCase
    % WORLDPROJECTIONTEST  World.project (W1_Kernel_Contract.md sections 6.3 and 12): K12 hidden components are
    % unreachable by every projection - another door's and the owning door's alike are refused at construction (V6),
    % and a paired perturbation of hidden values leaves every view unchanged; a "self" component shows only on the
    % viewer's own row, in every projection (the owner's too); a projection changes no state and returns only a View.

    methods (Static, Access = private)
        function d = secret(wealth)
            % Owns Badge (public), Mood (self), Wealth (hidden); three members spawned at reset.
            T = @(vis) OceanAgents.World.Guard.ComponentType('Visibility', vis);
            offers = arrayfun(@(k) struct('Badge', struct('Value', k, 'Type', T("public")), ...
                'Mood', struct('Value', 10 * k, 'Type', T("self")), ...
                'Wealth', struct('Value', wealth(k), 'Type', T("hidden"))), 1:3, 'UniformOutput', false);
            init = @(s) OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors.initSpawns(offers);
            % its OWN projection shows Badge and Mood; Mood is "self", so even the owner's view shows it on the
            % viewer's row only (critic F7: visibility applies to every projection)
            inner = OceanAgents.World.RuleSpec('Name', "Inner", 'Phase', "project", 'Kind', "projection", ...
                'Requires', "Badge", 'Reads', ["Badge" "Mood"], 'Fn', @(ctx) struct('View', ctx.Read));
            d = OceanAgents.World.DoorSpec('Name', "secret", 'Owns', struct('Name', {"Badge", "Mood", "Wealth"}, ...
                'Type', {T("public"), T("self"), T("hidden")}), 'Membership', "Badge", 'Drive', "never", 'Init', init, ...
                'Rules', inner);
        end

        function d = observer(reads)
            look = OceanAgents.World.RuleSpec('Name', "Look", 'Phase', "project", 'Kind', "projection", ...
                'Requires', "Badge", 'Reads', reads, 'Fn', @(ctx) struct('View', ...
                struct('IDs', ctx.IDs, 'Read', ctx.Read)));
            d = OceanAgents.World.DoorSpec('Name', "observer", 'Rules', look, 'Drive', "never");
        end
    end

    methods (Test)
        function hiddenOfAnotherDoorIsRefusedAtConstruction(tc)
            tc.verifyError(@() OceanAgents.World.World([tc.secret([1 2 3]), tc.observer(["Badge" "Wealth"])]), ...
                'OceanAgents:Guard:HiddenLeak');
        end

        function ownDoorHiddenIsRefusedAtConstruction(tc)
            % F7: hidden components are unreadable by EVERY projection, the owning door's included (V6).
            T = @(vis) OceanAgents.World.Guard.ComponentType('Visibility', vis);
            peek = OceanAgents.World.RuleSpec('Name', "Peek", 'Phase', "project", 'Kind', "projection", ...
                'Requires', "Secret", 'Reads', "Secret", 'Fn', @(ctx) struct('View', ctx.Read));
            vault = OceanAgents.World.DoorSpec('Name', "vault", 'Owns', struct('Name', "Secret", 'Type', T("hidden")), ...
                'Drive', "never", 'Rules', peek);
            tc.verifyError(@() OceanAgents.World.World(vault), 'OceanAgents:Guard:HiddenLeak');
        end

        function selfOnlyOnTheViewersRow(tc)
            w = OceanAgents.World.World([tc.secret([100 200 300]), tc.observer(["Badge" "Mood"])]);
            w.reset(0);
            v = w.project(2).observer_Look;
            tc.verifyEqual(v.IDs, [1; 2; 3]);
            tc.verifyEqual(v.Read.Badge, [1; 2; 3], 'public: everyone');
            tc.verifyEqual(v.Read.Mood, [0; 20; 0], 'self: only the viewer''s own row');
            tc.verifyEqual(v.Read.Mood_Has, [false; true; false]);
            none = w.project(0).observer_Look;
            tc.verifyEqual(none.Read.Mood, [0; 0; 0], 'no viewer: no self component');
            tc.verifyError(@() w.project(99), 'OceanAgents:World:UnknownEntity');
        end

        function pairedPerturbationOfHiddenValues(tc)
            % K12: two worlds that differ only in the hidden Wealth give identical views for every viewer.
            a = OceanAgents.World.World([tc.secret([100 200 300]), tc.observer(["Badge" "Mood"])]);
            b = OceanAgents.World.World([tc.secret([-5 7e6 42]), tc.observer(["Badge" "Mood"])]);
            a.reset(0); b.reset(0);
            tc.assertNotEqual(a.Store.valueAt("Wealth", [1; 2; 3], 0), b.Store.valueAt("Wealth", [1; 2; 3], 0));
            for viewer = 0:3
                tc.verifyEqual(a.project(viewer), b.project(viewer), sprintf('viewer %d', viewer));
            end
            own = a.project(2).secret_Inner;                 % the owning door's own projection
            tc.verifyEqual(own.Mood, [0; 20; 0], 'self: viewer row only, also for the owning door');
        end

        function projectionChangesNothingAndOnlyReturnsAView(tc)
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            init = @(s) L.initSpawns({L.offer('Position', [1 1], 'Velocity', [1 0])});
            w = OceanAgents.World.World(L.mover('Init', init));
            w.reset(0);
            w.run(2);
            before = w.Store.snapshot();
            v = w.project(1);
            tc.verifyEqual(v.mover_See.Position, [4 1]);
            tc.verifyEqual(w.Store.snapshot(), before);
            rogue = OceanAgents.World.RuleSpec('Name', "Leak", 'Phase', "project", 'Kind', "projection", ...
                'Requires', "Position", 'Reads', "Position", 'Fn', @(ctx) struct('View', struct(), 'Extra', 1));
            w2 = OceanAgents.World.World([L.mover('Init', init), OceanAgents.World.DoorSpec('Name', "rogue", ...
                'Rules', rogue, 'Drive', "never")]);
            w2.reset(0);
            tc.verifyError(@() w2.project(1), 'OceanAgents:World:Undeclared');
        end
    end
end
