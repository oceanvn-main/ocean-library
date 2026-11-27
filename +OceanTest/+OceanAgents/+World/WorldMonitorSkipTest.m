classdef WorldMonitorSkipTest < OceanTest.BaseTestCase
    % WORLDMONITORSKIPTEST  Monitors and instant skipping (W1_Kernel_Contract.md section 12):
    %   - monitors see the post-command state (an entity removed at its predicted crossing is not monitored there);
    %   - an entity RANGE monitor on a linear or clock component is skip-safe WITHOUT any predictor: the kernel
    %     schedules a check at the first integer instant the value leaves the range, so a skipping world fails at the
    %     same instant, with the same axiom and entity, as a stepping world (previously a hole: the skipping world
    %     never processed that instant);
    %   - a PREDICATE monitor on a linear/clock component is refused at construction when skipping (even if its door
    %     predicts that component - the predictor need not predict the violation); it is allowed when stepping.

    methods (Static, Access = private)
        function L = L()
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
        end

        function [when, err, w] = runToFailure(doors, skip, tEnd)
            % Advance until a failure (returns the instant and the error) or until tEnd (when = Inf).
            w = OceanAgents.World.World(doors, 'SkipIdle', skip);
            w.reset(0);
            when = Inf;
            err = [];
            try
                w.run(tEnd);
            catch err
                when = w.Now;
            end
        end
    end

    methods (Test)
        function clockRangeMonitorWithoutPredictor(tc)
            % AG-1: age <= 100, as a range axiom on the clock Born; no OldAge predictor. Born = -95: age 101 at t = 6.
            L = tc.L();
            doors = L.ager('Retire', false, 'Init', @(s) L.initSpawns({L.offer('Born', -95)}));
            w = OceanAgents.World.World(doors, 'SkipIdle', true);
            w.reset(0);
            tc.verifyEqual(w.NextTime, 6, 'nothing is driven: the scheduled monitor check is the only future work');
            [tSkip, eSkip] = tc.runToFailure(doors, true, 50);
            [tStep, eStep] = tc.runToFailure(doors, false, 50);
            tc.verifyEqual([tSkip tStep], [6 6]);
            tc.verifyEqual(eSkip.identifier, 'OceanAgents:Guard:AxiomViolated');
            tc.verifyEqual(eSkip.message, eStep.message);
            tc.verifySubstring(eSkip.message, 'AG-1');
            tc.verifySubstring(eSkip.message, 'entity 1');
        end

        function linearRangeMonitorWithoutPredictor(tc)
            % TK-1: Level >= 0 with no predictor. Level 10 at rate -0.5: 0 at t = 20 (still in range), -0.5 at 21.
            % Level 10 at rate -0.3: -0.2 at t = 34 (the first integer instant below 0).
            L = tc.L();
            for c = {[-0.5 21], [-0.3 34]}
                rate = c{1}(1);
                expected = c{1}(2);
                doors = L.tank('Init', @(s) L.initSpawns({L.offer('Level', struct('Value', 10, 'Rate', rate))}));
                w = OceanAgents.World.World(doors);
                w.reset(0);
                tc.verifyEqual(w.NextTime, expected);
                [tSkip, eSkip, ws] = tc.runToFailure(doors, true, 100);
                [tStep, eStep] = tc.runToFailure(doors, false, 100);
                tc.verifyEqual([tSkip tStep], [expected expected], sprintf('rate %g', rate));
                tc.verifyEqual(eSkip.message, eStep.message);
                tc.verifySubstring(eSkip.message, 'TK-1');
                tc.verifyTrue(ws.Registry.isAlive(1), 'fail-stop: nothing of the failing phase applied');
            end
        end

        function checkFollowsRebasesAndRemoval(tc)
            % The check is recomputed when the time function changes (battery rebase at t = 4 to rate -1: value 8,
            % below 0 first at 13) and cancelled with the entity (the battery removes it at its crossing, 12).
            L = tc.L();
            init = @(s) L.initSpawns({L.offer('Charge', struct('Value', 10, 'Rate', -0.5))});
            w = OceanAgents.World.World(L.battery('RebaseAt', 4, 'Init', init));
            w.reset(0);
            check = @() w.Calendar.peek();
            items = check();
            tc.verifyEqual(items([items.DueEvent] ~= "Empty").Time, 21);
            w.run(4);
            items = check();
            tc.verifyEqual(items([items.DueEvent] ~= "Empty").Time, 13);
            tc.verifyEqual(items([items.DueEvent] == "Empty").Time, 12);
            w.run(20);
            tc.verifyFalse(w.Registry.isAlive(1));
            tc.verifyEqual(w.Calendar.count(), 0, 'the pending check was cancelled with the entity');
            tc.verifyEqual(w.Status, "Running", 'removed at its crossing: BT-1 (Charge >= 0) not tripped');
        end

        function postCommandStateIsMonitored(tc)
            % Removal exactly at the (ceil) crossing: charge -0.2 at t = 34 is never monitored, the entity is gone.
            L = tc.L();
            init = @(s) L.initSpawns({L.offer('Charge', struct('Value', 10, 'Rate', -0.3))});
            w = OceanAgents.World.World(L.battery('Init', init));
            w.reset(0);
            r = w.run(100);
            tc.verifyEqual([r.Time], 34);
            tc.verifyEqual(w.Status, "Running");
            tc.verifyFalse(w.Registry.isAlive(1));
        end

        function predicateMonitorOnTimeVaryingRefusedWhenSkipping(tc)
            L = tc.L();
            try
                OceanAgents.World.World(L.ager('Retire', false, 'Monitor', "predicate"), 'SkipIdle', true);
                tc.verifyFail('must be refused when skipping');
            catch err
                tc.verifyEqual(err.identifier, 'OceanAgents:World:BadAssembly');
                tc.verifySubstring(err.message, 'AG-1');
                tc.verifySubstring(err.message, 'Born');
            end
            w = OceanAgents.World.World(L.ager('Retire', false, 'Monitor', "predicate"), 'SkipIdle', false);
            tc.verifyEqual(w.Status, "Idle", 'allowed when stepping');
            % a predictor on Born does not exempt it (critic B5: a predictor need not predict the violation)
            tc.verifyError(@() OceanAgents.World.World(L.ager('Retire', true, 'Monitor', "predicate"), 'SkipIdle', true), ...
                'OceanAgents:World:BadAssembly');
        end
    end
end
