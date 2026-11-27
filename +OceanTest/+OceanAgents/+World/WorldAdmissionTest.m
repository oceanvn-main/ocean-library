classdef WorldAdmissionTest < OceanTest.BaseTestCase
    % WORLDADMISSIONTEST  Spawn and attach through the World loop (W1_Kernel_Contract.md sections 7.1 step 5.3 and
    % 9.2; K15): the car scenario is refused with its axiom IDs and recorded while the run continues; without the
    % axioms it is admitted; Init spawns go through the same admission at reset; an entity joining two doors is checked
    % against both; attach sees the entity's current components; a wrong dimension is a type mismatch; attaching a
    % component the entity already has is refused ("already-has").

    methods (Static, Access = private)
        function d = arrivals(offersAtReset, offersPerFiring, k)
            % Door with no components: Init spawns offersAtReset; Arrive (every-k) spawns offersPerFiring.
            arrive = OceanAgents.World.RuleSpec('Name', "Arrive", 'Phase', "settle", 'Commands', "spawn", ...
                'Fn', @(ctx) struct('Commands', OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors.initSpawns(offersPerFiring)));
            d = OceanAgents.World.DoorSpec('Name', "arrivals", 'Rules', arrive, 'Drive', "every-k", 'DriveK', k, ...
                'Init', @(s) OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors.initSpawns(offersAtReset));
        end

        function d = humanAttacher(plan)
            % The human door (GuardDoors.human(true): Speed, Mass, HB-1, HB-2) with a rule that attaches, at instant
            % plan(k).At, plan(k).Offer to entity plan(k).Target - the owning door issues the attach (N1).
            h = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors.human(true);
            fire = OceanAgents.World.RuleSpec('Name', "Attach", 'Phase', "act", 'Commands', "attach", ...
                'Fn', @(ctx) attachFn(ctx, plan));
            d = OceanAgents.World.DoorSpec('Name', h.Name, 'Owns', h.Owns, 'Membership', h.Membership, ...
                'Axioms', h.Axioms, 'Rules', fire, 'Drive', "every-tick");

            function out = attachFn(ctx, plan)
                out = struct();
                k = find([plan.At] == ctx.Now, 1);
                if ~isempty(k)
                    out.Commands = OceanAgents.World.Command.attach(plan(k).Target, plan(k).Offer);
                end
            end
        end
    end

    methods (Test)
        function carRefusedAndRecordedRunContinues(tc)
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            w = OceanAgents.World.World([G.human(true), G.road(), tc.arrivals({G.carOffer(), G.walkerOffer()}, ...
                {G.carOffer()}, 3)]);
            w.reset(0);
            rr = w.ResetReport;
            tc.verifyEqual(rr.CommandsApplied, 1, 'the walker is admitted at reset');
            tc.verifyEqual(numel(rr.CommandsRefused), 1);
            ref = rr.CommandsRefused(1);
            tc.verifyEqual(ref.Kind, "spawn");
            tc.verifyEqual(ref.Reason, "axiom");
            tc.verifyEqual(sort(ref.AxiomIDs), ["HB-1" "HB-2"]);
            tc.verifyEqual(w.Registry.numAlive(), 1);
            reports = w.run(9);                              % Arrive fires at 0, 3, 6, 9: every car refused
            refused = arrayfun(@(r) numel(r.Phases(3).CommandsRefused), reports);
            tc.verifyEqual(sum(refused), 4);
            tc.verifyEqual(w.Status, "Running");
            tc.verifyEqual(w.Registry.numAlive(), 1);
            tc.verifyEqual(w.Registry.LastID, 1, 'a refused spawn allocates no ID');
        end

        function carAdmittedWithoutTheAxioms(tc)
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            w = OceanAgents.World.World([G.human(false), G.road(), tc.arrivals({G.carOffer()}, {}, 1000)]);
            w.reset(0);
            tc.verifyEqual(w.ResetReport.CommandsApplied, 1);
            tc.verifyEmpty(w.ResetReport.CommandsRefused);
            tc.verifyEqual(w.Store.valueAt("Speed", 1, 0), 30);
        end

        function checkedAgainstEveryDoorItJoins(tc)
            % A walker (fine for human) placed off the road segment is refused by the road's RD-1.
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            w = OceanAgents.World.World([G.human(true), G.road(), tc.arrivals({G.offer(1.4, 70, [2000 0])}, {}, 1000)]);
            w.reset(0);
            tc.verifyEqual(w.ResetReport.CommandsRefused(1).AxiomIDs, "RD-1");
        end

        function attachChecksTheJoinedEntity(tc)
            % Entity 1 is on the road only (Position). t = 1: wrong dimension (type-mismatch); t = 2: the car's
            % Speed/Mass (HB-1, HB-2); t = 3: walker values admitted - the entity now belongs to both doors; t = 4:
            % attaching Speed/Mass again is refused (already-has).
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            D = OceanAgents.World.Guard.Dimension;
            T = @(varargin) OceanAgents.World.Guard.ComponentType(varargin{:});
            roadOnly = struct('Position', struct('Value', [5 0], 'Type', T('Width', 2, 'Dimension', D.L())));
            body = @(speed, mass) struct('Speed', struct('Value', speed, 'Type', T('Dimension', D.L() / D.T())), ...
                'Mass', struct('Value', mass, 'Type', T('Dimension', D.M())));
            wrongDim = struct('Speed', struct('Value', 1, 'Type', T('Dimension', D.L())), ...
                'Mass', struct('Value', 70, 'Type', T('Dimension', D.M())));
            plan = struct('At', {1, 2, 3, 4}, 'Target', {1, 1, 1, 1}, ...
                'Offer', {wrongDim, body(30, 1200), body(1.4, 70), body(1.0, 60)});
            w = OceanAgents.World.World([tc.humanAttacher(plan), G.road(), tc.arrivals({roadOnly}, {}, 1000)]);
            w.reset(0);
            r = w.run(4);
            act = arrayfun(@(x) x.Phases(2), r);
            tc.verifyEqual(act(2).CommandsRefused(1).Reason, "type-mismatch", 'right class and width, wrong dimension');
            tc.verifyEqual(sort(act(3).CommandsRefused(1).AxiomIDs), ["HB-1" "HB-2"]);
            tc.verifyEqual(act(4).CommandsApplied, 1);
            tc.verifyEqual(w.Store.valueAt("Speed", 1, 4), 1.4);
            tc.verifyEqual(act(5).CommandsRefused(1).Reason, "already-has", 'commands never change values');
        end
    end
end
