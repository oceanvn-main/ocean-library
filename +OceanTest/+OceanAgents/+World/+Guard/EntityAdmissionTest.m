classdef EntityAdmissionTest < OceanTest.BaseTestCase
    % ENTITYADMISSIONTEST  GuardEngine.admitEntity (layers 1 and 2) and GuardEngine.checkMonitors on the real
    % OceanAgents.World.Store (W1_Kernel_Contract.md sections 9.2 and 7.1 step 5.4; K15, K17 guard half).
    % Car scenario of design section 9A.6: refused with the axiom ID when the axiom exists, admitted when it does not.

    properties (Constant)
        G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors
    end

    methods (Static, Access = private)
        function [ok, reasons] = admit(offer, doors, store, now, varargin)
            if nargin < 4, now = 0; end
            [ok, reasons] = OceanAgents.World.Guard.GuardEngine.admitEntity(offer, doors, store, now, varargin{:});
        end
    end

    methods (Test)
        function carScenarioRefusedByAxiomIDs(tc)
            doors = [tc.G.human(true), tc.G.road()];
            store = tc.G.storeFor(doors);
            [ok, reasons] = tc.admit(tc.G.carOffer(), doors, store);
            tc.verifyFalse(ok);
            tc.verifyEqual([reasons.Reason], ["axiom" "axiom"]);
            tc.verifyEqual([reasons.AxiomID], ["HB-1" "HB-2"]);          % Speed 30 > 12 and Mass 1200 > 300
            tc.verifyEqual([reasons.Door], ["human" "human"]);
            tc.verifySubstring(char(reasons(1).Detail), 'Speed');
            % a human-sized walker is admitted by both doors
            tc.verifyTrue(tc.admit(tc.G.walkerOffer(), doors, store));
        end

        function carAdmittedWhenTheAxiomIsAbsent(tc)
            % No axiom forbids it, so the car is admitted: the honest consequence of the world's rules.
            doors = [tc.G.human(false), tc.G.road()];
            [ok, reasons] = tc.admit(tc.G.carOffer(), doors, tc.G.storeFor(doors));
            tc.verifyTrue(ok);
            tc.verifyEmpty(reasons);
        end

        function entityJoiningTwoDoorsIsCheckedAgainstBoth(tc)
            doors = [tc.G.human(true), tc.G.road()];
            store = tc.G.storeFor(doors);
            [ok, reasons] = tc.admit(tc.G.offer(1.4, 70, [-5 0]), doors, store);    % off the road
            tc.verifyFalse(ok);
            tc.verifyEqual([reasons.AxiomID], "RD-1");
            tc.verifyEqual([reasons.Door], "road");
            [~, reasons] = tc.admit(tc.G.offer(30, 70, [-5 0]), doors, store);      % fails both doors
            tc.verifyEqual(sort([reasons.AxiomID]), ["HB-1" "RD-1"]);
            % a door applies only when its Membership is a subset of the entity's components
            only = rmfield(tc.G.carOffer(), 'Position');
            [~, reasons] = tc.admit(only, doors, store);
            tc.verifyEqual([reasons.AxiomID], ["HB-1" "HB-2"]);
            onlyRoad = rmfield(tc.G.carOffer(), {'Speed', 'Mass'});
            tc.verifyTrue(tc.admit(onlyRoad, doors, store));                          % human axioms not applicable
        end

        function layerOneTypesAndValues(tc)
            doors = [tc.G.human(true), tc.G.road()];
            store = tc.G.storeFor(doors);
            D = tc.G.dim();
            base = tc.G.walkerOffer();
            % no owner
            o = base;  o.Fuel = struct('Value', 40, 'Type', tc.G.type('Dimension', D.E()));
            [ok, r] = tc.admit(o, doors, store);
            tc.verifyFalse(ok);
            tc.verifyEqual([r.Reason], "no-owner");
            tc.verifySubstring(char(r.Detail), 'Fuel');
            % right class and width, wrong dimension (K14): a fuel amount offered as a speed
            o = base;  o.Speed.Type = tc.G.type('Dimension', D.E());
            [ok, r] = tc.admit(o, doors, store);
            tc.verifyFalse(ok);
            tc.verifyEqual([r.Reason r.Door], ["type-mismatch" "human"]);
            tc.verifySubstring(char(r.Detail), 'Dimension');
            tc.verifySubstring(char(r.Detail), 'E^1');
            % width and kind mismatches
            o = base;  o.Position.Type = tc.G.type('Width', 3, 'Dimension', D.L());
            [~, r] = tc.admit(o, doors, store);
            tc.verifySubstring(char(r.Detail), 'Width');
            o = base;  o.Speed.Type = tc.G.type('Kind', "linear", 'Dimension', D.L() / D.T());
            [~, r] = tc.admit(o, doors, store);
            tc.verifySubstring(char(r.Detail), 'Kind');
            % visibility is not part of the match
            o = base;  o.Mass.Type = tc.G.type('Dimension', D.M(), 'Visibility', "hidden");
            tc.verifyTrue(tc.admit(o, doors, store));
            % malformed values
            for bad = {[1 2 3], [NaN 0], [true false], int32([1 2])}
                o = base;  o.Position.Value = bad{1};
                [ok, r] = tc.admit(o, doors, store);
                tc.verifyFalse(ok);
                tc.verifyEqual([r.Reason], "bad-value", mat2str(bad{1}));
            end
            % all layer-1 reasons are reported together; layer 2 is not evaluated on a malformed entity
            o = base;  o.Position.Value = [NaN 0];  o.Speed.Type = tc.G.type();
            [~, r] = tc.admit(o, doors, store);
            tc.verifyEqual(sort([r.Reason]), ["bad-value" "type-mismatch"]);
        end

        function linearAndClockOffers(tc)
            doors = [tc.G.battery(), tc.G.ager()];
            store = tc.G.storeFor(doors);
            D = tc.G.dim();
            charge = @(v, rate) struct('Charge', struct('Value', struct('Value', v, 'Rate', rate), ...
                'Type', tc.G.type('Kind', "linear", 'Dimension', D.E())));
            born = @(t0) struct('Born', struct('Value', t0, 'Type', tc.G.type('Kind', "clock", 'Dimension', D.T())));
            tc.verifyTrue(tc.admit(charge(10, -1), doors, store, 5));
            [ok, r] = tc.admit(charge(-5, 1), doors, store, 5);                     % BT-1: charge never negative
            tc.verifyFalse(ok);
            tc.verifyEqual([r.AxiomID], "BT-1");
            [~, r] = tc.admit(struct('Charge', struct('Value', 10, 'Type', tc.G.type('Kind', "linear", ...
                'Dimension', D.E()))), doors, store, 5);                             % linear needs (Value, Rate)
            tc.verifyEqual([r.Reason], "bad-value");
            tc.verifyTrue(tc.admit(born(-50), doors, store, 0));                     % age 50
            [ok, r] = tc.admit(born(-150), doors, store, 0);                         % AG-1: age 150 > 100
            tc.verifyFalse(ok);
            tc.verifyEqual([r.AxiomID r.Door], ["AG-1" "ager"]);
            tc.verifyTrue(tc.admit(born(0), doors, store, 100));                     % age exactly 100
            tc.verifyFalse(tc.admit(born(0), doors, store, 101));
        end

        function attachUsesTheEntitysComponentsAfterTheCommand(tc)
            doors = [tc.G.human(true), tc.G.road()];
            store = tc.G.storeFor(doors);
            row = store.addRows(1);
            store.setOffer(row, rmfield(tc.G.walkerOffer(), 'Position'), 0);        % Speed 1.4, Mass 70
            onRoad = struct('Position', tc.G.walkerOffer().Position);
            tc.verifyTrue(tc.admit(onRoad, doors, store, 0, 'Row', row));
            off = onRoad;  off.Position.Value = [2000 0];
            [ok, r] = tc.admit(off, doors, store, 0, 'Row', row);
            tc.verifyFalse(ok);
            tc.verifyEqual([r.AxiomID], "RD-1");
            % an existing component violating another door's axiom is found when that door becomes applicable:
            % the entity holds Position only; attaching Speed + Mass 400 joins "human" and fails HB-2
            row2 = store.addRows(1);
            store.setOffer(row2, onRoad, 0);
            [~, r] = tc.admit(rmfield(tc.G.offer(3, 400, [0 0]), 'Position'), doors, store, 0, 'Row', row2);
            tc.verifyEqual([r.AxiomID], "HB-2");
            % post-write values: a proposed write raising row's Speed to 20 makes the attach fail HB-1
            proposed = struct('Speed', struct('Rows', row, 'Values', 20));
            [ok, r] = tc.admit(onRoad, doors, store, 0, 'Row', row, 'Proposed', proposed);
            tc.verifyFalse(ok);
            tc.verifyEqual([r.AxiomID], "HB-1");
        end

        function attachOfAComponentAlreadyHeldIsRefusedF2(tc)
            % Commands never change values: attaching a component the entity already has is refused ("already-has").
            doors = [tc.G.human(true), tc.G.road()];
            store = tc.G.storeFor(doors);
            row = store.addRows(1);
            store.setOffer(row, rmfield(tc.G.walkerOffer(), 'Position'), 0);         % Speed 1.4, Mass 70
            again = struct('Speed', tc.G.walkerOffer().Speed);
            again.Speed.Value = 3;                                                    % a legal value, still refused
            [ok, r] = tc.admit(again, doors, store, 0, 'Row', row);
            tc.verifyFalse(ok);
            tc.verifyEqual([r.Reason r.Component r.Door], ["already-has" "Speed" "human"]);
            tc.verifyEqual(store.valueAt("Speed", row, 0), 1.4);                     % nothing changed
            % a spawn has no current components: the same offer is fine there
            tc.verifyTrue(tc.admit(rmfield(tc.G.walkerOffer(), 'Position'), doors, store, 0));
            % attached EARLIER IN THE SAME SYNC (overlay Has = true): refused as well, and the overlay value is used
            onRoad = struct('Position', tc.G.walkerOffer().Position);
            earlier = struct('Position', struct('Rows', row, 'Values', [3 0], 'Has', true));
            [ok, r] = tc.admit(onRoad, doors, store, 0, 'Row', row, 'Proposed', earlier);
            tc.verifyFalse(ok);
            tc.verifyEqual([r.Reason r.Component], ["already-has" "Position"]);
            % detached earlier in the same sync (Has = false): the component may be attached again
            row2 = store.addRows(1);
            store.setOffer(row2, tc.G.walkerOffer(), 0);
            gone = struct('Position', struct('Rows', row2, 'Values', [0 0], 'Has', false));
            tc.verifyFalse(tc.admit(onRoad, doors, store, 0, 'Row', row2));             % still held in the store
            tc.verifyTrue(tc.admit(onRoad, doors, store, 0, 'Row', row2, 'Proposed', gone));
        end

        function overlayMakesSplitAttachesSeeEachOther(tc)
            % F1 support: an attach sees components attached earlier in the same sync through overlay Has/Values,
            % so splitting Speed and Mass into two attaches cannot bypass the human axioms.
            doors = [tc.G.human(true), tc.G.road()];
            store = tc.G.storeFor(doors);
            row = store.addRows(1);
            store.setOffer(row, struct('Position', tc.G.walkerOffer().Position), 0);
            car = tc.G.carOffer();                                                     % Speed 30, Mass 1200
            tc.verifyTrue(tc.admit(struct('Speed', car.Speed), doors, store, 0, 'Row', row));   % alone: not a member
            earlier = struct('Speed', struct('Rows', row, 'Values', 30, 'Has', true));
            [ok, r] = tc.admit(struct('Mass', car.Mass), doors, store, 0, 'Row', row, 'Proposed', earlier);
            tc.verifyFalse(ok);
            tc.verifyEqual([r.AxiomID], ["HB-1" "HB-2"]);                            % sees Speed 30 from the overlay
        end

        function reasonsIndependentOfOfferFieldOrderF15(tc)
            doors = [tc.G.human(true), tc.G.road()];
            store = tc.G.storeFor(doors);
            D = tc.G.dim();
            o = tc.G.offer(30, 1200, [-5 0]);                                         % three axiom failures
            bad = tc.G.walkerOffer();
            bad.Speed.Type = tc.G.type('Dimension', D.E());
            bad.Position.Value = [NaN 1];
            bad.Fuel = struct('Value', 1, 'Type', tc.G.type('Dimension', D.E()));      % three layer-1 failures
            row = store.addRows(1);
            store.setOffer(row, tc.G.walkerOffer(), 0);
            cases = {{o, {}}, {bad, {}}, {tc.G.walkerOffer(), {'Row', row}}};
            for c = 1:numel(cases)
                offer = cases{c}{1};
                extra = cases{c}{2};
                f = fieldnames(offer);
                [~, r0] = tc.admit(offer, doors, store, 0, extra{:});
                tc.verifyNotEmpty(r0);
                for perm = perms(1:numel(f))'
                    [~, r] = tc.admit(orderfields(offer, f(perm)), doors, store, 0, extra{:});
                    tc.verifyEqual(r, r0, sprintf('case %d, order %s', c, strjoin(f(perm), ',')));
                end
            end
            [~, r] = tc.admit(bad, doors, store, 0);
            tc.verifyEqual([r.Reason], ["bad-value" "no-owner" "type-mismatch"]);
            tc.verifyEqual([r.Component], ["Position" "Fuel" "Speed"]);
            [~, r] = tc.admit(tc.G.walkerOffer(), doors, store, 0, 'Row', row);
            tc.verifyEqual([r.Component], ["Mass" "Position" "Speed"]);               % all already held
        end

        function refusalNeverThrowsAndChangesNothing(tc)
            A = @(varargin) OceanAgents.World.Guard.AxiomSpec(varargin{:});
            broken = A('ID', "HB-7", 'Reads', "Mass", 'Predicate', @(v) v.Mass.NoSuchField > 0);   % throws
            wrongShape = A('ID', "HB-8", 'Reads', "Mass", 'Predicate', @(v) [true true]);
            doors = [tc.G.human(true, [broken wrongShape]), tc.G.road()];
            store = tc.G.storeFor(doors);
            row = store.addRows(1);
            store.setOffer(row, tc.G.walkerOffer(), 0);
            before = store.snapshot();
            [ok, r] = tc.admit(tc.G.walkerOffer(), doors, store);
            tc.verifyFalse(ok);
            tc.verifyEqual([r.AxiomID], ["HB-7" "HB-8"]);
            tc.verifySubstring(char(r(1).Detail), 'predicate error');
            tc.verifySubstring(char(r(2).Detail), '1x2');
            tc.verifyEqual(store.snapshot(), before);                                 % admission is pure
            tc.verifyEqual(sort(fieldnames(r))', {'AxiomID', 'Component', 'Detail', 'Door', 'Reason'});
        end

        function monitorsSeePostWriteValues(tc)
            doors = [tc.G.human(true), tc.G.road(), tc.G.ager(), tc.G.battery()];
            store = tc.G.storeFor(doors);
            rows = store.addRows(3);
            store.setOffer(rows(1), tc.G.offer(5, 70, [1 0]), 0);
            store.setOffer(rows(2), tc.G.offer(8, 400, [2 0]), 0);                   % Mass 400: HB-2 is STATIC
            D = tc.G.dim();
            store.setOffer(rows(3), struct('Born', struct('Value', -85, 'Type', tc.G.type('Kind', "clock", ...
                'Dimension', D.T())), 'Charge', struct('Value', struct('Value', 10, 'Rate', -1), ...
                'Type', tc.G.type('Kind', "linear", 'Dimension', D.E()))), 0);
            chk = @(proposed, now) OceanAgents.World.Guard.GuardEngine.checkMonitors(doors, store, proposed, now);
            tc.verifyEmpty(chk(struct(), 5));                                         % static axioms are not monitored
            v = chk(struct('Speed', struct('Rows', rows(1), 'Values', 20)), 5);       % proposed write breaks HB-1
            tc.verifyEqual([v.AxiomID], "HB-1");
            tc.verifyEqual(v.EntityRows, rows(1));
            tc.verifyEqual(v.Door, "human");
            v = chk(struct(), 20);                     % clock: age 105 > 100; linear: 10 - 20 = -10 < 0
            tc.verifyEqual([v.AxiomID], ["AG-1" "BT-1"]);
            tc.verifyEqual([v.EntityRows], [rows(3) rows(3)]);
            v = chk(struct('Born', struct('Rows', rows(3), 'Values', 15)), 20);       % clock write sets T0: age 5
            tc.verifyEqual([v.AxiomID], "BT-1");
            v = chk(struct(), 11);                                                    % linear: 10 - 11 = -1 < 0
            tc.verifyEqual([v.AxiomID], "BT-1");
            tc.verifyEmpty(chk(struct('Charge', struct('Rows', rows(3), 'Base', 4, 'Rate', 0, 'T0', 11)), 11));
            v = chk(struct('Charge', struct('Rows', rows(3), 'Values', -2)), 3);      % a write rebases at now
            tc.verifyEqual([v.AxiomID], "BT-1");
            % several rows at once; dead rows are ignored
            v = chk(struct('Speed', struct('Rows', rows(1:2), 'Values', [30; 40])), 5);
            tc.verifyEqual(v.EntityRows, rows(1:2)');
            store.killRows(rows(1));
            v = chk(struct('Speed', struct('Rows', rows(1:2), 'Values', [30; 40])), 5);
            tc.verifyEqual(v.EntityRows, rows(2));
        end

        function prebuiltIndexGivesIdenticalVerdicts(tc)
            % buildIndex + 'Index' must change nothing but the cost: same verdicts, reasons and violations.
            GE = OceanAgents.World.Guard.GuardEngine;
            doors = [tc.G.human(true), tc.G.road(), tc.G.ager(), tc.G.battery()];
            idx = GE.buildIndex(doors);
            store = tc.G.storeFor(doors);
            D = tc.G.dim();
            wrongDim = tc.G.walkerOffer();  wrongDim.Speed.Type = tc.G.type('Dimension', D.E());
            fuel = tc.G.walkerOffer();  fuel.Fuel = struct('Value', 1, 'Type', tc.G.type('Dimension', D.E()));
            offers = {tc.G.carOffer(), tc.G.walkerOffer(), tc.G.offer(1.4, 70, [-5 0]), wrongDim, fuel, ...
                rmfield(tc.G.carOffer(), 'Position'), ...
                struct('Born', struct('Value', -150, 'Type', tc.G.type('Kind', "clock", 'Dimension', D.T()))), ...
                struct('Charge', struct('Value', struct('Value', -1, 'Rate', 0), 'Type', ...
                    tc.G.type('Kind', "linear", 'Dimension', D.E())))};
            for k = 1:numel(offers)
                [ok1, r1] = GE.admitEntity(offers{k}, doors, store, 3);
                [ok2, r2] = GE.admitEntity(offers{k}, doors, store, 3, 'Index', idx);
                tc.verifyEqual(ok2, ok1, sprintf('offer %d', k));
                tc.verifyEqual(r2, r1, sprintf('offer %d', k));
            end
            rows = store.addRows(3);
            store.setOffer(rows(1), tc.G.offer(5, 70, [1 0]), 0);
            store.setOffer(rows(2), rmfield(tc.G.walkerOffer(), 'Position'), 0);
            store.setOffer(rows(3), struct('Born', struct('Value', -85), 'Charge', struct('Value', ...
                struct('Value', 10, 'Rate', -1))), 0);
            onRoad = struct('Position', tc.G.walkerOffer().Position);
            proposed = struct('Speed', struct('Rows', rows(2), 'Values', 20));
            [ok1, r1] = GE.admitEntity(onRoad, doors, store, 0, 'Row', rows(2), 'Proposed', proposed);
            [ok2, r2] = GE.admitEntity(onRoad, doors, store, 0, 'Row', rows(2), 'Proposed', proposed, 'Index', idx);
            tc.verifyEqual({ok2, r2}, {ok1, r1});
            tc.verifyEqual([r1.AxiomID], "HB-1");
            for now = [0 11 20]
                p = struct('Speed', struct('Rows', rows(1), 'Values', 30));
                v1 = GE.checkMonitors(doors, store, p, now);
                v2 = GE.checkMonitors(doors, store, p, now, 'Index', idx);
                tc.verifyEqual(v2, v1, sprintf('now %d', now));
                tc.verifyNotEmpty(v1);
            end
            tc.verifyError(@() GE.admitEntity(tc.G.walkerOffer(), doors, store, 0, 'Index', struct('x', 1)), ...
                'OceanAgents:World:BadAssembly');
        end

        function admissionCostWithPrebuiltIndex(tc)
            % 1e4 admissions (the World's Init path) with a prebuilt index; reported, not asserted beyond completing.
            GE = OceanAgents.World.Guard.GuardEngine;
            doors = [tc.G.human(true), tc.G.road(), tc.G.ager(), tc.G.battery()];
            store = tc.G.storeFor(doors);
            offer = tc.G.walkerOffer();
            n = 1e4;
            tic;
            idx = GE.buildIndex(doors);
            tBuild = toc;
            tic;
            okAll = true;
            for k = 1:n
                okAll = GE.admitEntity(offer, doors, store, 0, 'Index', idx) && okAll;
            end
            tIndexed = toc;
            m = 500;
            tic;
            for k = 1:m
                GE.admitEntity(offer, doors, store, 0);
            end
            tPlain = toc * n / m;
            tc.verifyTrue(okAll);
            tc.log(1, sprintf(['admitEntity x %d: %.3f s with a prebuilt index (%.1f us/call; buildIndex %.2f ms); ' ...
                'without the index ~%.3f s (extrapolated from %d calls)'], n, tIndexed, 1e6 * tIndexed / n, ...
                1e3 * tBuild, tPlain, m));
        end

        function unvalidatedMissingReadIsAFaultNotAPass(tc)
            % V11 makes this unreachable for a validated assembly; if the guard ever meets a member lacking a read
            % component (assembly not validated), it raises instead of silently passing the axiom.
            A = @(varargin) OceanAgents.World.Guard.AxiomSpec(varargin{:});
            far = A('ID', "HB-5", 'Mode', "monitor", 'Reads', ["Speed" "Position"], ...
                'Predicate', @(v) v.Speed < 10 | v.Position(:, 1) > 0);
            doors = [tc.G.human(true, far), tc.G.road()];
            GE = OceanAgents.World.Guard.GuardEngine;
            tc.verifyError(@() GE.validateAssembly(doors, tc.G.Phases, strings(1, 0)), 'OceanAgents:Guard:BadAxiom');
            store = tc.G.storeFor(doors);
            row = store.addRows(1);
            store.setOffer(row, rmfield(tc.G.offer(11, 70, [0 0]), 'Position'), 0);    % a human member without Position
            tc.verifyError(@() GE.checkMonitors(doors, store, struct(), 0), 'OceanAgents:Guard:BadAxiom');
            tc.verifyError(@() GE.admitEntity(rmfield(tc.G.walkerOffer(), 'Position'), doors, store, 0), ...
                'OceanAgents:Guard:BadAxiom');
            % with Position in the human Membership (V11 satisfied) the axiom is checked on every member
            fixed = OceanAgents.World.DoorSpec('Name', "human", 'Owns', doors(1).Owns, ...
                'Membership', ["Speed" "Mass" "Position"], 'Axioms', doors(1).Axioms);
            road = OceanAgents.World.DoorSpec('Name', "road", 'Owns', doors(2).Owns, 'Membership', "Position", ...
                'Axioms', doors(2).Axioms);
            GE.validateAssembly([fixed road], tc.G.Phases, strings(1, 0));
            store2 = tc.G.storeFor([fixed road]);
            rows = store2.addRows(2);
            store2.setOffer(rows(1), rmfield(tc.G.offer(11, 70, [0 0]), 'Position'), 0);   % not a human member
            store2.setOffer(rows(2), tc.G.offer(11, 70, [0 0]), 0);                        % member, violates HB-5
            v = GE.checkMonitors([fixed road], store2, struct(), 0);
            tc.verifyEqual([v.AxiomID], "HB-5");
            tc.verifyEqual(v.EntityRows, rows(2));
        end
    end
end
