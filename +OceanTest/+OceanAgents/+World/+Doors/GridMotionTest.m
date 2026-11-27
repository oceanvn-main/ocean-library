classdef GridMotionTest < OceanTest.BaseTestCase
    % GRIDMOTIONTEST  G1 for OceanAgents.World.Doors.GridMotion (W2_Grid_Doors_Contract.md sections 3.1, 7):
    % exact model rows against hand-built rows, the Move rule's draw count (exactly one per event, deterministic moves
    % included) and stochastic-wind outcome order/selection against a twin mt19937ar stream, the cliff teleport,
    % blocked moves from terminal cells, the Init start draw, the Cell projection, and the rule driven by a World.

    methods (Static, Access = private)
        function g = cliffCfg()
            gw = OceanAgents.World.Doors.GridWorlds.cliffWalking();
            g = gw.GridMotion;
        end
        function g = windyCfg(stochastic)
            gw = OceanAgents.World.Doors.GridWorlds.windy('StochasticWind', stochastic);
            g = gw.GridMotion;
        end
        function ctx = ctx(g, pos, ids, labels, stream)
            % A rule context as the World builds it: rows 1..n, the door's resources, one Move event per label.
            read = rmfield(g, 'AgentID');
            read.Position = pos;
            read.Position_Has = true(size(pos, 1), 1);
            ev = OceanAgents.World.Event.empty(1, 0);
            for k = 1:numel(labels)
                ev(end + 1) = OceanAgents.World.Event('Type', "Move", 'Time', 1, 'Phase', "act", ...
                    'SourceEntity', ids(min(k, numel(ids))), 'Seq', k, 'Payload', struct('Label', labels(k))); %#ok<AGROW>
            end
            ctx = struct('Now', 1, 'Phase', "act", 'Door', "GridMotion", 'Rows', (1:size(pos, 1))', ...
                'IDs', ids(:), 'Read', read, 'Events', ev, 'Stream', stream);
        end
        function rs = mt(seed)
            rs = RandStream('mt19937ar', 'Seed', seed);
        end
    end

    methods (Test)
        % ---------------- exact model rows ----------------
        function transitionsCliffHandRows(tc)
            g = tc.cliffCfg();
            T = @(s, a) OceanAgents.World.Doors.GridMotion.transitions(g, s, a);
            [n, p] = T(37, 2);  tc.verifyEqual([n p], [37 1]);        % right from the start is the cliff -> start
            [n, p] = T(37, 1);  tc.verifyEqual([n p], [25 1]);
            [n, p] = T(25, 3);  tc.verifyEqual([n p], [37 1]);
            [n, p] = T(36, 3);  tc.verifyEqual([n p], [48 1]);        % into the goal
            [n, p] = T(1, 1);   tc.verifyEqual([n p], [1 1]);         % clipped at the top-left corner
            [n, p] = T(48, 1);  tc.verifyEqual([n p], zeros(0, 2));   % terminal goal: no outcomes
            [n, ~] = T(40, 1);  tc.verifyEmpty(n);                    % cliff cells are terminal-like
            tc.verifyError(@() T(49, 1), 'OceanAgents:World:BadValue');
            tc.verifyError(@() T(37, 5), 'OceanAgents:World:BadValue');
        end

        function transitionsStochasticWindOrder(tc)
            % Windy 7 x 10, cell (4,7) = index 37, wind 2 -> winds [1 2 3] in that order, probs 1/3 each.
            % Right (label 2) -> column 8, rows 4-1=3, 4-2=2, 4-3=1 -> indices 28, 18, 8.
            g = tc.windyCfg(true);
            [n, p] = OceanAgents.World.Doors.GridMotion.transitions(g, 37, 2);
            tc.verifyEqual(n, [28; 18; 8]);
            tc.verifyEqual(p, [1; 1; 1] / 3);
            % up (label 1) from (2,7) = 17: rows 2-1-1=0->1, 2-1-2->1, 2-1-3->1: three outcomes, all cell 7
            [n, p] = OceanAgents.World.Doors.GridMotion.transitions(g, 17, 1);
            tc.verifyEqual(n, [7; 7; 7]);
            tc.verifyEqual(p, [1; 1; 1] / 3);
            % a calm column has one outcome even with stochastic wind
            [n, p] = OceanAgents.World.Doors.GridMotion.transitions(g, 31, 2);    % (4,1) wind 0
            tc.verifyEqual([n p], [32 1]);
            % deterministic wind: one outcome, wind 2
            [n, p] = OceanAgents.World.Doors.GridMotion.transitions(tc.windyCfg(false), 37, 2);
            tc.verifyEqual([n p], [18 1]);
        end

        % ---------------- the Move rule ----------------
        function oneDrawPerMoveDeterministicIncluded(tc)
            g = tc.cliffCfg();
            rs = tc.mt(5);  twin = tc.mt(5);
            out = OceanAgents.World.Doors.GridMotion.moveRule(tc.ctx(g, [4 1], 7, [1 2 2 3], rs));
            rand(twin, 1, 4);
            tc.verifyEqual(rs.State, twin.State, 'four deterministic moves draw exactly four uniforms');
            % (4,1) up -> (3,1); right -> (3,2); right -> (3,3); down -> (4,3) = cliff -> back to (4,1)
            tc.verifyEqual(out.Writes.Values, [4 1]);
            tc.verifyEqual([out.Events.Type], ["Moved" "Moved" "Moved" "CliffFall"]);
            tc.verifyEqual([out.Events(1).Payload.From out.Events(1).Payload.To], [37 25]);
            tc.verifyEqual(out.Events(4).Payload, struct('From', 27, 'To', 37, 'Fell', 39));
            tc.verifyEqual(unique([out.Events.SourceEntity]), 7);
        end

        function stochasticSelectionMatchesTwin(tc)
            % From (4,7) moving right: outcome k = find(u*c(end) <= c) with c = cumsum([1 1 1]/3)'. The twin stream
            % replays the native rule; over 60 seeds every one of the three outcomes occurs.
            g = tc.windyCfg(true);
            cells = [3 8; 2 8; 1 8];
            seen = false(1, 3);
            for seed = 1:60
                rs = tc.mt(seed);  twin = tc.mt(seed);
                out = OceanAgents.World.Doors.GridMotion.moveRule(tc.ctx(g, [4 7], 1, 2, rs));
                u = rand(twin);
                c = cumsum([1 1 1]' / 3);
                k = find(u * c(end) <= c, 1);
                tc.assertEqual(out.Writes.Values, cells(k, :), sprintf('seed %d', seed));
                tc.assertEqual(rs.State, twin.State);
                seen(k) = true;
                % hand check of the selection: u <= 1/3 -> first outcome (wind 1), u > 2/3 -> last (wind 3)
                if u <= 1/3 - 1e-12, tc.assertEqual(k, 1); elseif u > 2/3 + 1e-12, tc.assertEqual(k, 3); end
            end
            tc.verifyTrue(all(seen), 'every stochastic outcome was exercised');
        end

        function terminalBlocksWithoutDrawing(tc)
            g = tc.cliffCfg();
            rs = tc.mt(3);  twin = tc.mt(3);
            out = OceanAgents.World.Doors.GridMotion.moveRule(tc.ctx(g, [4 12], 1, [1 4], rs));   % at the goal
            tc.verifyEqual(rs.State, twin.State);
            tc.verifyEmpty(out.Events);
            tc.verifyEmpty(out.Writes.Rows);
            tc.verifyError(@() OceanAgents.World.Doors.GridMotion.moveRule(tc.ctx(g, [4 1], 1, 5, rs)), ...
                'OceanAgents:World:BadValue');
            tc.verifyError(@() OceanAgents.World.Doors.GridMotion.moveRule(tc.ctx(g, [4 1], 1, 1.5, rs)), ...
                'OceanAgents:World:BadValue');
        end

        function eventsAreMatchedToTheirEntities(tc)
            % Two entities, events in delivery order; each moves from its own current cell.
            g = tc.cliffCfg();
            c = tc.ctx(g, [1 1; 2 2], [11; 12], [2 3], tc.mt(1));
            c.Events(2) = OceanAgents.World.Event('Type', "Move", 'Time', 1, 'Phase', "act", 'SourceEntity', 12, ...
                'Seq', 2, 'Payload', struct('Label', 3));
            out = OceanAgents.World.Doors.GridMotion.moveRule(c);
            tc.verifyEqual(out.Writes.Rows, [1; 2]);
            tc.verifyEqual(out.Writes.Values, [1 2; 3 2]);
            c.Events(2) = OceanAgents.World.Event('Type', "Move", 'Time', 1, 'Phase', "act", 'SourceEntity', 99, ...
                'Seq', 2, 'Payload', struct('Label', 3));
            tc.verifyError(@() OceanAgents.World.Doors.GridMotion.moveRule(c), 'OceanAgents:World:BadValue');
        end

        % ---------------- Init and projection ----------------
        function initDrawsOnceAndSpawnsAtStart(tc)
            g = tc.windyCfg(false);
            rs = tc.mt(9);  twin = tc.mt(9);
            cmd = OceanAgents.World.Doors.GridMotion.initSpawn(g, "agent7", rs);
            rand(twin);
            tc.verifyEqual(rs.State, twin.State, 'reset draws exactly one uniform (native doReset_)');
            tc.verifyEqual(cmd.Kind, "spawn");
            tc.verifyEqual(cmd.Offer.Position.Value, [4 1]);
            tc.verifyEqual(cmd.Offer.AgentID.Value, "agent7");
            tc.verifyEqual(cmd.Offer.Steps.Value, 0);
            tc.verifyTrue(cmd.Offer.Steps.Type.matches(OceanAgents.World.Doors.RewardTermination.stepsType()));
        end

        function cellProjection(tc)
            out = OceanAgents.World.Doors.GridMotion.cellProjection(struct('Read', struct('Position', [4 1; 1 12], 'Cols', 12)));
            tc.verifyEqual(out.View.Index, [37; 12]);
        end

        function specShape(tc)
            d = OceanAgents.World.Doors.GridMotion.spec(tc.cliffCfg());
            tc.verifyEqual([d.Name d.Stream d.Drive], ["GridMotion" "legacy-mt19937ar" "on-event"]);
            tc.verifyEqual(string({d.Owns.Name}), "Position");
            t = d.Owns(1).Type;
            tc.verifyEqual([t.Width, t.Dimension.isDimensionless()], [2 1]);
            tc.verifyEqual([t.Kind t.Class t.Visibility], ["value" "double" "public"]);
            tc.verifyEqual([d.Rules.Name], ["Move" "Cell"]);
            tc.verifyEqual(sort(string(fieldnames(d.Resources)))', ...
                sort(["Rows" "Cols" "Moves" "Wind" "StochasticWind" "Cliff" "Start" "Terminal"]));
            bad = tc.cliffCfg();  bad.Start = [4 2];                    % start on the cliff
            tc.verifyError(@() OceanAgents.World.Doors.GridMotion.spec(bad), 'OceanAgents:World:BadValue');
            bad = tc.cliffCfg();  bad.Wind = [0 0];
            tc.verifyError(@() OceanAgents.World.Doors.GridMotion.spec(bad), 'OceanAgents:World:BadValue');
        end

        % ---------------- driven by the World ----------------
        function worldDrivenMatchesTwinStream(tc)
            % The doors in a World (Agency + GridMotion + RewardTermination), Move events injected one per instant.
            % The door's legacy stream is mt19937ar(seed): a twin replaying the native rule (one draw at reset, one
            % per move) predicts every cell, including stochastic-wind outcomes.
            tc.assumeTrue(OceanTest.OceanAgents.World.Doors.GridMotionTest.kernelReady(), ...
                'needs Agency.m and the legacy stream / outbox in World (B1, W2 phase 1)');
            gw = OceanAgents.World.Doors.GridWorlds.windy('StochasticWind', true);
            g = gw.GridMotion;
            w = OceanAgents.World.World(gw.Doors, 'SkipIdle', false, 'Injectable', "Move", ...
                'Outbox', ["Reward" "Terminated" "Truncated"]);
            for seed = [0 3 17]
                w.reset(seed);
                twin = tc.mt(seed);
                rand(twin);                                         % the start draw
                id = w.Registry.idsOf(w.Store.rowsWith("Position"));
                tc.assertNumElements(id, 1);
                pos = g.Start;
                labels = 1 + mod((1:40) * 7, 4);
                for t = 1:numel(labels)
                    s = (pos(1) - 1) * g.Cols + pos(2);
                    if g.Terminal(pos(1), pos(2)), break; end
                    [n, p] = OceanAgents.World.Doors.GridMotion.transitions(g, s, labels(t));
                    c = cumsum(p);
                    nxt = n(find(rand(twin) * c(end) <= c, 1));
                    w.inject(OceanAgents.World.Event.make("Move", struct('Label', labels(t)), 'SourceEntity', id));
                    w.advance();
                    v = w.project(id);
                    tc.assertEqual(v.GridMotion_Cell.Index, nxt, sprintf('seed %d step %d', seed, t));
                    ev = w.takeOutbox();
                    tc.assertEqual(ev(1).Type, "Reward");
                    tc.assertEqual(ev(1).Payload.Amount, -1);
                    pos = [floor((nxt - 1) / g.Cols) + 1, mod(nxt - 1, g.Cols) + 1];
                end
            end
        end
    end

    methods (Static)
        function tf = kernelReady()
            % True once B1's W2 kernel additions and the Agency door exist.
            tf = exist('OceanAgents.World.Doors.Agency', 'class') == 8 && ...
                any(strcmp(methods('OceanAgents.World.World'), 'takeOutbox'));
        end
    end
end
