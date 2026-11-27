classdef WorldFailStopTest < OceanTest.BaseTestCase
    % WORLDFAILSTOPTEST  Fail-stop of the World loop (W1_Kernel_Contract.md section 5.5): any error while an instant
    % is processed leaves Store, Registry and Calendar exactly as at the last completed synchronisation point
    % (compared against snapshots), sets Status "Failed", records Failure; advance/run/inject/project then raise
    % World:InvalidState; reset is allowed. Covers K3 (write conflict) and K17 (monitor violation naming the axiom ID
    % and the entity IDs).

    methods (Static, Access = private)
        function s = state(w)
            s = struct('Store', w.Store.snapshot(), 'Registry', w.Registry.snapshot(), ...
                'Calendar', w.Calendar.snapshot(), 'Resources', w.Resources);
        end

        function d = twin(conflictAt, phase)
            % Door owning X (dimensionless) with two rules; both write X on every row at instant conflictAt.
            T = OceanAgents.World.Guard.ComponentType();
            w1 = OceanAgents.World.RuleSpec('Name', "W1", 'Phase', phase, 'Requires', "X", 'Reads', "X", 'Writes', "X", ...
                'Fn', @(ctx) struct('Writes', struct('Component', "X", 'Rows', ctx.Rows, 'Values', ctx.Read.X + 1)));
            w2 = OceanAgents.World.RuleSpec('Name', "W2", 'Phase', phase, 'Requires', "X", 'Reads', "X", 'Writes', "X", ...
                'Fn', @(ctx) conflictFn(ctx, conflictAt));
            init = @(s) [OceanAgents.World.Command.spawn(struct('X', struct('Value', 0, 'Type', T))), ...
                OceanAgents.World.Command.spawn(struct('X', struct('Value', 10, 'Type', T)))];
            d = OceanAgents.World.DoorSpec('Name', "twin", 'Owns', struct('Name', "X", 'Type', T), 'Membership', "X", ...
                'Rules', [w1 w2], 'Drive', "every-tick", 'Init', init);

            function out = conflictFn(ctx, at)
                out = struct();
                if ctx.Now == at
                    out.Writes = struct('Component', "X", 'Rows', ctx.Rows(1), 'Values', -1);
                end
            end
        end
    end

    methods (Test)
        function writeConflictInFirstPhase(tc)
            % K3: two rules write (X, row 1) in "resolve" at t = 3: nothing of instant 3 is applied.
            w = OceanAgents.World.World(tc.twin(3, "resolve"));
            w.reset(0);
            w.run(2);
            before = tc.state(w);
            try
                w.advance();
                tc.verifyFail('the conflict must fail the instant');
            catch err
                tc.verifyEqual(err.identifier, 'OceanAgents:World:WriteConflict');
                tc.verifySubstring(err.message, 'twin/W1');
                tc.verifySubstring(err.message, 'twin/W2');
            end
            tc.verifyEqual(tc.state(w), before, 'Store, Registry, Calendar and resources exactly as after instant 2');
            tc.verifyEqual(w.Status, "Failed");
            tc.verifyEqual(w.Failure.identifier, 'OceanAgents:World:WriteConflict');
            tc.verifyEqual(w.Store.valueAt("X", [1; 2], w.Now), [3; 13]);
        end

        function failureInLaterPhaseKeepsEarlierPhases(tc)
            % The conflict is in "settle" at t = 3: the "act" synchronisation of instant 3 (mover) stays applied,
            % the settle outputs do not.
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            init = @(s) L.initSpawns({L.offer('Position', [0 0], 'Velocity', [2 1])});
            w = OceanAgents.World.World([L.mover('Init', init), tc.twin(3, "settle")]);
            w.reset(0);
            w.run(2);
            tc.verifyError(@() w.advance(), 'OceanAgents:World:WriteConflict');
            moverRow = w.Registry.rowsOf(1);
            tc.verifyEqual(w.Store.valueAt("Position", moverRow, 3), [8 4], 'moved at t = 0, 1, 2 and 3 (act)');
            tc.verifyEqual(w.Store.valueAt("X", w.Registry.rowsOf([2; 3]), 3), [3; 13], 'X as after instant 2');
        end

        function monitorViolationNamesAxiomAndEntity(tc)
            % K17: an ager entity admitted at age 95 (AG-1: age <= 100 holds) violates AG-1 at t = 6.
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            init = @(s) L.initSpawns({L.offer('Position', [0 0], 'Velocity', [0 0]), L.offer('Born', -95), ...
                L.offer('Born', -10)});
            w = OceanAgents.World.World([L.mover('Init', init), L.ager('Retire', false)]);
            w.reset(0);
            tc.verifyEqual(w.ResetReport.CommandsApplied, 3, 'admitted: age 95 satisfies the axiom');
            w.run(5);
            before = tc.state(w);
            try
                w.advance();
                tc.verifyFail('AG-1 must be violated at t = 6');
            catch err
                tc.verifyEqual(err.identifier, 'OceanAgents:Guard:AxiomViolated');
                tc.verifySubstring(err.message, 'AG-1');
                tc.verifySubstring(err.message, 'entity 2');
            end
            tc.verifyEqual(tc.state(w), before);
            tc.verifyEqual(w.Now, 6);
        end

        function afterFailureOnlyResetWorks(tc)
            w = OceanAgents.World.World(tc.twin(0, "resolve"));
            w.reset(0);
            tc.verifyError(@() w.advance(), 'OceanAgents:World:WriteConflict');
            ID = 'OceanAgents:World:InvalidState';
            tc.verifyError(@() w.advance(), ID);
            tc.verifyError(@() w.run(10), ID);
            tc.verifyError(@() w.inject(OceanAgents.World.Event.make("Ping")), ID);
            tc.verifyError(@() w.project(0), ID);
            w.reset(0);
            tc.verifyEqual(w.Status, "Running");
            tc.verifyEmpty(w.Failure);
            tc.verifyEqual(w.Registry.LastID, 4, 'the failed episode''s IDs 1-2 are never reused');
        end

        function ruleErrorIsFailStop(tc)
            % An error thrown inside a rule function: same fail-stop, the error recorded as Failure.
            boom = OceanAgents.World.RuleSpec('Name', "Boom", 'Phase', "act", 'Fn', @(ctx) boomFn(ctx));
            d = OceanAgents.World.DoorSpec('Name', "fragile", 'Resources', struct('K', 0), 'Rules', [boom, ...
                OceanAgents.World.RuleSpec('Name', "Tick", 'Phase', "resolve", 'Reads', "K", 'Writes', "K", ...
                'Fn', @(ctx) struct('Writes', struct('Resource', "K", 'Values', ctx.Read.K + 1)))], 'Drive', "every-tick");
            w = OceanAgents.World.World(d);
            w.reset(0);
            w.run(3);
            tc.verifyError(@() w.advance(), 'Test:Boom');
            tc.verifyEqual(w.Failure.identifier, 'Test:Boom');
            tc.verifyEqual(w.Resources.K, 5, 'the resolve sync of instant 4 was completed before act failed');

            function out = boomFn(ctx)
                out = struct();
                if ctx.Now == 4, error('Test:Boom', 'boom at %d', ctx.Now); end
            end
        end
    end
end
