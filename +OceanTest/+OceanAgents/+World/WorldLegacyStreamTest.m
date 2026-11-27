classdef WorldLegacyStreamTest < OceanTest.BaseTestCase
    % WORLDLEGACYSTREAMTEST  W2 kernel additions (W2_Grid_Doors_Contract.md section 2; K-W2): a "legacy-mt19937ar"
    % door's stream is RandStream('mt19937ar', 'Seed', seed) at reset(seed); "keyed" doors keep their StreamRegistry
    % stream; reset([]) keeps every door stream (twin-stream check: the draws continue exactly where they stopped) and
    % clears the rest (Init runs again, the ID counter continues); reset([]) before any seeded reset raises
    % OceanAgents:World:NoSeed.

    methods (Static, Access = private)
        function d = drawer(name, stream)
            % Init spawns an entity whose <name>Mark is one draw; every instant appends one draw to <name>Draws.
            T = OceanAgents.World.Guard.ComponentType();
            res = char(name + "Draws");
            mark = char(name + "Mark");
            draw = OceanAgents.World.RuleSpec('Name', "Draw", 'Phase', "act", 'Reads', string(res), 'Writes', string(res), ...
                'Fn', @(ctx) struct('Writes', struct('Resource', string(res), 'Values', [ctx.Read.(res) rand(ctx.Stream)])));
            d = OceanAgents.World.DoorSpec('Name', name, 'Stream', stream, 'Resources', struct(res, zeros(1, 0)), ...
                'Owns', struct('Name', string(mark), 'Type', T), 'Rules', draw, 'Drive', "every-tick", ...
                'Init', @(s) OceanAgents.World.Command.spawn(struct(mark, struct('Value', rand(s), 'Type', T))));
        end
    end

    methods (Test)
        function legacyStreamIsMt19937arOfTheSeed(tc)
            w = OceanAgents.World.World([tc.drawer("old", "legacy-mt19937ar"), tc.drawer("new", "keyed")], ...
                'SkipIdle', false);
            w.reset(42);
            w.run(4);
            legacy = rand(RandStream('mt19937ar', 'Seed', 42), 1, 6);   % Init draw, then one per instant 0..4
            tc.verifyEqual([w.Store.valueAt("oldMark", 1, 0) w.Resources.oldDraws], legacy);
            keyed = rand(OceanAgents.Runtime.StreamRegistry(42).stream("door:new"), 1, 6);
            tc.verifyEqual([w.Store.valueAt("newMark", 2, 0) w.Resources.newDraws], keyed);
        end

        function twinStreamContinuesAcrossResetEmpty(tc)
            % reset(9), instants 0..4, reset([]), instants 0..4: each door's 12 draws (2 Init + 10 rule draws) are
            % exactly the first 12 of its stream - legacy mt19937ar(9) and keyed StreamRegistry(9) "door:new".
            w = OceanAgents.World.World([tc.drawer("old", "legacy-mt19937ar"), tc.drawer("new", "keyed")], ...
                'SkipIdle', false);
            w.reset(9);
            w.run(4);
            old1 = [w.Store.valueAt("oldMark", 1, 0) w.Resources.oldDraws];
            new1 = [w.Store.valueAt("newMark", 2, 0) w.Resources.newDraws];
            w.reset([]);
            tc.verifyEqual(w.Registry.numAlive(), 2, 'Store cleared, Init ran again');
            tc.verifyEqual(w.Registry.LastID, 4, 'the ID counter continues');
            tc.verifyEmpty(w.Resources.oldDraws, 'resources back to their initial values');
            w.run(4);
            old2 = [w.Store.valueAt("oldMark", w.Registry.rowsOf(3), 0) w.Resources.oldDraws];
            new2 = [w.Store.valueAt("newMark", w.Registry.rowsOf(4), 0) w.Resources.newDraws];
            tc.verifyEqual([old1 old2], rand(RandStream('mt19937ar', 'Seed', 9), 1, 12));
            tc.verifyEqual([new1 new2], rand(OceanAgents.Runtime.StreamRegistry(9).stream("door:new"), 1, 12));
            tc.verifyEqual(w.Seed, 9, 'Seed is the last SEEDED reset');
        end

        function seededResetRestartsTheStreams(tc)
            w = OceanAgents.World.World(tc.drawer("old", "legacy-mt19937ar"), 'SkipIdle', false);
            w.reset(9);
            w.run(4);
            a = w.Resources.oldDraws;
            w.reset([]);
            w.run(4);
            tc.verifyNotEqual(w.Resources.oldDraws, a, 'continued, not restarted');
            w.reset(9);
            w.run(4);
            tc.verifyEqual(w.Resources.oldDraws, a, 'a seeded reset restarts the stream');
        end

        function resetEmptyNeedsASeededReset(tc)
            w = OceanAgents.World.World(tc.drawer("old", "legacy-mt19937ar"));
            tc.verifyError(@() w.reset([]), 'OceanAgents:World:NoSeed');
            tc.verifyError(@() w.reset(1.5), 'OceanAgents:World:BadValue');
            w.reset(0);
            w.reset([]);
            tc.verifyEqual(w.Status, "Running");
        end
    end
end
