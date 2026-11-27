classdef DimensionCheckTest < OceanTest.BaseTestCase
    % DIMENSIONCHECKTEST  K14: dimensions by algebra (V8 DimChecks with factors "Name", "Name^p" and the reserved "dt"),
    % the wrong-dimension refusal at entity admission, and the RENAMING test: consistently renaming every component,
    % resource, door and event name in an assembly never changes any guard verdict (names are addresses, never evidence).

    properties (Constant)
        G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors
        R = OceanTest.OceanAgents.Fixtures.World.Guard.Renamer
    end

    methods (Static, Access = private)
        function id = verdict(doors, injectable)
            % "ok" or the error identifier raised by validateAssembly.
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            if nargin < 2, injectable = G.Injectable; end
            id = "ok";
            try
                OceanAgents.World.Guard.GuardEngine.validateAssembly(doors, G.Phases, injectable);
            catch err
                id = string(err.identifier);
            end
        end

        function d = accel(dimTerms)
            % A door with Position (L), Velocity (L/T) and Accel (L/T^2) to exercise powers of dt.
            D = OceanAgents.World.Guard.Dimension;
            T = @(varargin) OceanAgents.World.Guard.ComponentType(varargin{:});
            r = OceanAgents.World.RuleSpec('Name', "Kick", 'Phase', "act", 'Reads', ["Pos" "Vel" "Acc"], ...
                'Writes', "Pos", 'DimChecks', struct('Target', "Pos", 'Terms', {dimTerms}), 'Fn', @(c) struct());
            d = OceanAgents.World.DoorSpec('Name', "ballistic", 'Owns', struct('Name', {"Pos", "Vel", "Acc"}, ...
                'Type', {T('Dimension', D.L()), T('Dimension', D.L() / D.T()), T('Dimension', D.L() / D.T() ^ 2)}), ...
                'Rules', r);
        end
    end

    methods (Test)
        function dimensionalAlgebraOfRules(tc)
            ok = @(terms) tc.verifyEqual(tc.verdict(tc.accel(terms)), "ok");
            bad = @(terms) tc.verifyEqual(tc.verdict(tc.accel(terms)), "OceanAgents:Guard:DimensionMismatch");
            ok({"Pos", ["Vel" "dt"]});                                     % x + v dt
            ok({"Pos", ["Vel" "dt"], ["Acc" "dt^2"]});                     % x + v dt + a dt^2
            ok({["Vel^2" "Acc^-1"]});                                      % v^2 / a  is a length
            ok({["Pos^2" "Pos^-1"]});
            bad({"Pos", "Vel"});                                           % L + L T^-1
            bad({"Pos", ["Acc" "dt"]});                                    % L T^-1 is not L
            bad({["Vel" "dt^-1"]});
            bad({["Acc" "dt^3"]});
            % the mover door of the contract: Position += Velocity*dt
            tc.verifyEqual(tc.verdict(tc.G.mover()), "ok");
            tc.verifyEqual(tc.verdict(tc.G.mover({"Position", ["Velocity" "dt^2"]})), "OceanAgents:Guard:DimensionMismatch");
        end

        function malformedFactorsAreBadRules(tc)
            id = "OceanAgents:World:BadRule";
            tc.verifyEqual(tc.verdict(tc.accel({["Vel" "dt^x"]})), id);
            tc.verifyEqual(tc.verdict(tc.accel({["Vel" "dt^1.5"]})), id);
            tc.verifyEqual(tc.verdict(tc.accel({["Speedo" "dt"]})), id);   % unknown factor
            D = OceanAgents.World.Guard.Dimension;
            r = OceanAgents.World.RuleSpec('Name', "Kick", 'Phase', "act", 'Writes', "Pos", 'Fn', @(c) struct(), ...
                'DimChecks', struct('Target', "Nowhere", 'Terms', {{"Pos"}}));
            d = OceanAgents.World.DoorSpec('Name', "ballistic", 'Owns', struct('Name', "Pos", ...
                'Type', OceanAgents.World.Guard.ComponentType('Dimension', D.L())), 'Rules', r);
            tc.verifyEqual(tc.verdict(d), id);                             % unknown target
            try
                OceanAgents.World.Guard.GuardEngine.validateAssembly(tc.accel({"Pos", "Vel"}), tc.G.Phases, strings(1, 0));
            catch err
                tc.verifySubstring(err.message, 'ballistic');                % door, rule and dimensions named
                tc.verifySubstring(err.message, 'Kick');
                tc.verifySubstring(err.message, 'L^1 T^-1');
            end
        end

        function wrongDimensionRefusedAtAdmission(tc)
            % Right class, right width, wrong dimension: refused (fuel E^1 is not a speed L^1 T^-1).
            doors = tc.G.human(false);
            D = tc.G.dim();
            o = rmfield(tc.G.walkerOffer(), 'Position');
            o.Speed.Type = tc.G.type('Dimension', D.E());
            [ok, r] = OceanAgents.World.Guard.GuardEngine.admitEntity(o, doors, tc.G.storeFor(doors), 0);
            tc.verifyFalse(ok);
            tc.verifyEqual([r.Reason], "type-mismatch");
            o.Speed.Type = tc.G.type('Dimension', D.L() / D.T());
            tc.verifyTrue(OceanAgents.World.Guard.GuardEngine.admitEntity(o, doors, tc.G.storeFor(doors), 0));
        end

        function renamingNeverChangesAValidationVerdict(tc)
            A = @(varargin) OceanAgents.World.Guard.AxiomSpec(varargin{:});
            rule = @(varargin) OceanAgents.World.RuleSpec('Phase', "act", 'Fn', @(c) struct(), varargin{:});
            door = @(name, varargin) OceanAgents.World.DoorSpec('Name', name, varargin{:});
            base = tc.G.kernelAssembly();
            car = [tc.G.human(), tc.G.road()];
            assemblies = { ...
                base, ...                                                                   % ok
                [base, tc.G.human(), tc.G.road()], ...                                       % ok
                [base, base(1)], ...                                                         % V1 duplicate door
                [tc.G.mover(), tc.G.road()], ...                                             % V2 duplicate owner
                [base, door("thief", 'Rules', rule('Name', "Steal", 'Writes', "Charge"))], ...   % V3
                [base, door("loud", 'Rules', rule('Name', "Shout", 'Emits', "Boom"))], ...       % V4
                [base, door("ear", 'Rules', rule('Name', "Hear", 'Consumes', "Whisper"))], ...   % V5
                [base, door("spy", 'Owns', struct('Name', "Gold", 'Type', tc.G.type('Visibility', "hidden")), ...
                    'Rules', rule('Name', "Keep", 'Reads', "Gold")), door("eye", 'Rules', OceanAgents.World.RuleSpec( ...
                    'Name', "Look", 'Phase', "project", 'Kind', "projection", 'Reads', "Gold", 'Fn', @(c) struct()))], ...  % V6
                [base, tc.G.mover({"Position", "Velocity"})], ...                             % V8 (and V1: duplicate)
                tc.G.mover({"Position", "Velocity"}), ...                                     % V8
                [car, tc.G.road(A('ID', "HB-1", 'Component', "Position", 'Range', [0 1]))], ...  % V9 (also V1 dup road)
                [tc.G.human(), door("race", 'Membership', "Speed", ...
                    'Axioms', A('ID', "RC-1", 'Component', "Speed", 'Range', [20 30]))], ...    % V10
                [tc.G.human(), door("race", 'Membership', "Speed", ...
                    'Axioms', A('ID', "RC-1", 'Component', "Speed", 'Range', [10 30]))], ...    % ok
                [tc.G.human(), door("race", 'Membership', "Mass", ...
                    'Axioms', A('ID', "RC-1", 'Component', "Speed", 'Range', [10 30]))], ...    % V11
                [base, door("vault", 'Owns', struct('Name', "Gold", 'Type', tc.G.type('Visibility', "hidden"))), ...
                    door("engine", 'Rules', rule('Name', "Use", 'Reads', "Gold"))], ...          % V12
                [base, door("vault", 'Owns', struct('Name', "Gold", 'Type', tc.G.type('Visibility', "hidden"))), ...
                    door("engine", 'Rules', rule('Name', "Use", 'Reads', "Gold", 'Reveals', "Gold"))], ...   % ok (declared)
                [base, door("vault", 'Owns', struct('Name', "Gold", 'Type', tc.G.type('Visibility', "self"))), ...
                    door("eye", 'Rules', OceanAgents.World.RuleSpec('Name', "Look", 'Phase', "project", ...
                    'Kind', "projection", 'Requires', "Gold", 'Fn', @(c) struct()))], ...          % N3 (V6 Requires)
                [base, door("vault", 'Owns', struct('Name', "Gold", 'Type', tc.G.type('Visibility', "hidden"))), ...
                    door("engine", 'Rules', rule('Name', "Use", 'Requires', "Gold"))], ...       % N3 (V12 Requires)
                [base, door("vault", 'Owns', struct('Name', "Hid", 'Type', tc.G.type('Kind', "linear", ...
                    'Visibility', "hidden"))), door("spy", 'Rules', [OceanAgents.World.RuleSpec('Name', "Sniff", ...
                    'Phase', "act", 'Kind', "predictor", 'Predicts', "Hid", 'DueEvent', "Empty", 'Fn', @(c) struct())])], ...   % N4
                [base, door("vault", 'Owns', struct('Name', "Gold", 'Type', tc.G.type('Visibility', "hidden"))), ...
                    door("gate", 'Owns', struct('Name', "Tag", 'Type', tc.G.type()), 'Membership', ["Tag" "Gold"], ...
                    'Axioms', A('ID', "GT-1", 'Component', "Gold", 'Range', [0 5]))]};       % V13 (R3)
            verdicts = strings(1, numel(assemblies));
            for k = 1:numel(assemblies)
                doors = assemblies{k};
                map = tc.R.mapFor(doors, ["Empty" "Ping" "Boom" "Whisper"]);
                renamed = tc.R.doors(doors, map);
                tc.assertNotEqual([renamed.Name], [doors.Name]);           % the renaming really changes names
                verdicts(k) = tc.verdict(doors, "Ping");
                tc.verifyEqual(tc.verdict(renamed, map('Ping')), verdicts(k), sprintf('assembly %d', k));
            end
            % the set of cases covers passes and every check family used above
            tc.verifyEqual(numel(unique(verdicts)), 10);  % "ok" + 9 distinct error IDs (V11 raises BadAxiom)
            tc.verifyEqual(verdicts(end - 6:end), ["OceanAgents:Guard:BadAxiom" "OceanAgents:Guard:HiddenLeak" "ok" ...
                "OceanAgents:Guard:HiddenLeak" "OceanAgents:Guard:HiddenLeak" "OceanAgents:Guard:HiddenLeak" ...
                "OceanAgents:Guard:HiddenLeak"]);
        end

        function renamingNeverChangesAnAdmissionOrMonitorVerdict(tc)
            doors = [tc.G.human(true), tc.G.road(), tc.G.ager(), tc.G.battery()];
            map = tc.R.mapFor(doors);
            renamed = tc.R.doors(doors, map);
            D = tc.G.dim();
            offers = {tc.G.carOffer(), tc.G.walkerOffer(), tc.G.offer(1.4, 70, [-5 0]), tc.G.offer(30, 70, [-5 0]), ...
                rmfield(tc.G.carOffer(), 'Position'), ...
                struct('Born', struct('Value', -150, 'Type', tc.G.type('Kind', "clock", 'Dimension', D.T()))), ...
                struct('Charge', struct('Value', struct('Value', -1, 'Rate', 0), 'Type', ...
                    tc.G.type('Kind', "linear", 'Dimension', D.E())))};
            wrongDim = tc.G.walkerOffer();  wrongDim.Speed.Type = tc.G.type('Dimension', D.E());
            offers{end + 1} = wrongDim;
            s1 = tc.G.storeFor(doors);
            s2 = tc.G.storeFor(renamed);
            nameOf = @(door) string(map(char(door)));
            for k = 1:numel(offers)
                [ok1, r1] = OceanAgents.World.Guard.GuardEngine.admitEntity(offers{k}, doors, s1, 0);
                [ok2, r2] = OceanAgents.World.Guard.GuardEngine.admitEntity(tc.R.renameFields(offers{k}, map), renamed, s2, 0);
                tc.verifyEqual(ok2, ok1, sprintf('offer %d', k));
                tc.verifyEqual([r2.Reason], [r1.Reason]);
                tc.verifyEqual([r2.AxiomID], [r1.AxiomID]);
                tc.verifyEqual([r2.Door], arrayfun(nameOf, [r1.Door]));
            end
            % monitors: same rows, same axiom IDs
            for s = {s1, s2}
                rows = s{1}.addRows(2);
            end
            s1.setOffer(rows, struct('Speed', struct('Value', [5; 20]), 'Mass', struct('Value', [70; 80])), 0);
            s2.setOffer(rows, tc.R.renameFields(struct('Speed', struct('Value', [5; 20]), 'Mass', struct('Value', [70; 80])), map), 0);
            v1 = OceanAgents.World.Guard.GuardEngine.checkMonitors(doors, s1, struct(), 0);
            v2 = OceanAgents.World.Guard.GuardEngine.checkMonitors(renamed, s2, struct(), 0);
            tc.verifyEqual([v2.AxiomID], [v1.AxiomID]);
            tc.verifyEqual([v1.AxiomID], "HB-1");
            tc.verifyEqual({v2.EntityRows}, {v1.EntityRows});
        end
    end
end
