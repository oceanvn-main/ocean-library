classdef WorldSkipEquivalenceTest < OceanTest.BaseTestCase
    % WORLDSKIPEQUIVALENCETEST  K9: skipping equals stepping. For 20 seeds, a random assembly of the test doors (mover
    % with random kicks, battery with random rate rebases and zero-crossing predictions, ager with retirement at age
    % 100, one or two spawners with random periods and offers, counter fed by an injected Ping) is run for 200 instants
    % twice: stepping every tick (SkipIdle false) and skipping idle instants (SkipIdle true). At every instant the
    % skipping world processes, its Store, Registry, Calendar and resources equal the stepping world's at that instant;
    % at the end both are equal (the Calendar compared without its release cursor).
    % Second part (section 12, skip-safe monitors): assemblies with an ager WITHOUT its retirement predictor (range
    % monitor AG-1 on the clock Born) and a tank WITHOUT any predictor (range monitor TK-1 on the linear Level). Their
    % monitors are eventually violated; the skipping world must fail at the same instant, with the same error, and be
    % left in the same (restored) state as the stepping world.

    properties (Constant)
        Seeds = 1:20
        Horizon = 200
    end

    methods (Static, Access = private)
        function doors = assembly(seed, withoutPredictors)
            if nargin < 2, withoutPredictors = false; end
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            rs = RandStream('mrg32k3a', 'Seed', 1000 + seed);
            use = rand(rs, 1, 3) < 0.6;                      % mover, battery, ager
            if ~any(use), use(randi(rs, 3)) = true; end
            doors = OceanAgents.World.DoorSpec.empty(1, 0);
            comps = strings(1, 0);
            if use(1), doors(end + 1) = L.mover('Kick', rand(rs) < 0.5); comps = [comps "Position" "Velocity"]; end
            if use(2)
                tweak = rand(rs) < 0.6;
                doors(end + 1) = L.battery('Tweak', tweak, 'DriveK', randi(rs, [2 9]));
                comps = [comps "Charge"];
            end
            if use(3) || withoutPredictors
                doors(end + 1) = L.ager('Retire', ~withoutPredictors);
                comps = [comps "Born"];
            end
            if withoutPredictors && rand(rs) < 0.7
                doors(end + 1) = L.tank();
                comps = [comps "Level"];
            end
            doors(end + 1) = L.spawner(randi(rs, [3 15]), comps);
            if rand(rs) < 0.5
                sub = comps(rand(rs, 1, numel(comps)) < 0.5);
                if isempty(sub), sub = comps(1); end
                doors(end + 1) = L.spawner(randi(rs, [5 25]), sub, "spawner2");
            end
            doors(end + 1) = L.counter();
        end

        function s = state(w, withCursor)
            c = w.Calendar.snapshot();
            if ~withCursor, c = rmfield(c, {'Now', 'NowPhase'}); end
            s = struct('Store', w.Store.snapshot(), 'Registry', w.Registry.snapshot(), 'Calendar', c, ...
                'Resources', w.Resources);
        end
    end

    methods (Test)
        function skippingEqualsStepping(tc)
            H = tc.Horizon;
            totalStep = 0;
            totalSkip = 0;
            spawned = 0;
            removed = 0;
            for seed = tc.Seeds
                doors = tc.assembly(seed);
                step = OceanAgents.World.World(doors, 'SkipIdle', false, 'Injectable', "Ping");
                skip = OceanAgents.World.World(doors, 'SkipIdle', true, 'Injectable', "Ping");
                step.reset(seed);  skip.reset(seed);
                step.inject(OceanAgents.World.Event.make("Ping"));
                skip.inject(OceanAgents.World.Event.make("Ping"));
                states = cell(1, H + 1);
                while step.NextTime <= H
                    r = step.advance();
                    states{r.Time + 1} = tc.state(step, true);
                end
                tc.assertEqual(numel(states), H + 1);
                processed = 0;
                while skip.NextTime <= H
                    r = skip.advance();
                    processed = processed + 1;
                    tc.assertEqual(tc.state(skip, true), states{r.Time + 1}, ...
                        sprintf('seed %d: states differ at instant %d', seed, r.Time));
                end
                tc.verifyEqual(tc.state(skip, false), tc.state(step, false), sprintf('seed %d: final states differ', seed));
                totalStep = totalStep + (H + 1);
                totalSkip = totalSkip + processed;
                spawned = spawned + skip.Registry.LastID;
                removed = removed + skip.Registry.LastID - skip.Registry.numAlive();
            end
            tc.log(1, sprintf(['K9: %d seeds x %d instants; stepping processed %d instants, skipping %d; ' ...
                '%d entities spawned, %d removed (predicted Empty / Old)'], numel(tc.Seeds), H + 1, totalStep, ...
                totalSkip, spawned, removed));
            tc.verifyGreaterThan(removed, 0, 'predicted removals must actually occur');
            tc.verifyLessThan(totalSkip, totalStep, 'some assemblies must have idle instants to skip');
        end

        function runAndInjectDriverPattern(tc)
            % A driver alternates run(t_k) and inject(Ping) (critic B6): after each run the two worlds have covered
            % the same time, the injected event is delivered at the same instant, and the states agree.
            for seed = 1:8
                doors = tc.assembly(seed);
                rs = RandStream('mrg32k3a', 'Seed', 500 + seed);
                stops = unique(sort(randi(rs, [0 tc.Horizon - 1], 1, 6)));
                step = OceanAgents.World.World(doors, 'SkipIdle', false, 'Injectable', "Ping");
                skip = OceanAgents.World.World(doors, 'SkipIdle', true, 'Injectable', "Ping");
                step.reset(seed);  skip.reset(seed);
                for tk = stops
                    step.run(tk);  skip.run(tk);
                    tc.assertEqual(tc.state(skip, false), tc.state(step, false), sprintf('seed %d: after run(%d)', seed, tk));
                    step.inject(OceanAgents.World.Event.make("Ping"));
                    skip.inject(OceanAgents.World.Event.make("Ping"));
                end
                step.run(tc.Horizon);  skip.run(tc.Horizon);
                tc.verifyEqual(tc.state(skip, false), tc.state(step, false), sprintf('seed %d: final', seed));
                tc.verifyEqual(skip.Resources.Pings, numel(stops), sprintf('seed %d: every ping delivered', seed));
            end
        end

        function monitorFailuresAtTheSameInstant(tc)
            H = tc.Horizon;
            failures = 0;
            for seed = 1:10
                doors = tc.assembly(seed, true);
                step = OceanAgents.World.World(doors, 'SkipIdle', false, 'Injectable', "Ping");
                skip = OceanAgents.World.World(doors, 'SkipIdle', true, 'Injectable', "Ping");
                step.reset(seed);  skip.reset(seed);
                states = cell(1, H + 1);
                stepErr = [];
                while step.NextTime <= H
                    try
                        r = step.advance();
                    catch stepErr
                        break
                    end
                    states{r.Time + 1} = tc.state(step, true);
                end
                skipErr = [];
                while skip.NextTime <= H
                    try
                        r = skip.advance();
                    catch skipErr
                        break
                    end
                    tc.assertEqual(tc.state(skip, true), states{r.Time + 1}, ...
                        sprintf('seed %d: states differ at instant %d', seed, r.Time));
                end
                tc.assertEqual(isempty(skipErr), isempty(stepErr), sprintf('seed %d: only one world failed', seed));
                if ~isempty(stepErr)
                    failures = failures + 1;
                    tc.verifyEqual(skip.Now, step.Now, sprintf('seed %d: failure instant', seed));
                    tc.verifyEqual(skipErr.identifier, stepErr.identifier);
                    tc.verifyEqual(skipErr.message, stepErr.message, sprintf('seed %d: failure message', seed));
                    tc.verifyEqual(tc.state(skip, false), tc.state(step, false), sprintf('seed %d: restored states', seed));
                end
            end
            tc.log(1, sprintf('skip-safe monitors: %d of 10 assemblies failed a monitor; skipping matched stepping', failures));
            tc.verifyGreaterThan(failures, 5, 'the monitors must actually be violated in most assemblies');
        end
    end
end
