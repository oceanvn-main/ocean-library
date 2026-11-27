classdef SpecRecordsTest < OceanTest.BaseTestCase
    % SPECRECORDSTEST  W1 Phase-0 records: Dimension algebra, ComponentType matching, AxiomSpec forms, Certificate
    % honesty, RuleSpec/DoorSpec declarations, Event and Command construction (W1_Kernel_Contract.md sections 3-4).

    methods (Test)
        function dimensionAlgebra(tc)
            D = OceanAgents.World.Guard.Dimension;
            v = D.L() / D.T();
            tc.verifyEqual(v.Exponents, [1 0 -1 0 0 0]);
            tc.verifyTrue(v * D.T() == D.L());                          % L T^-1 * T = L
            tc.verifyTrue((D.L() ^ 2) / D.L() == D.L());
            tc.verifyTrue(D.none().isDimensionless());
            tc.verifyFalse(D.E() == v);
            tc.verifyEqual(char(v), 'L^1 T^-1');
            tc.verifyError(@() OceanAgents.World.Guard.Dimension([1 0 0.5 0 0 0]), 'OceanAgents:Guard:BadDimension');
            tc.verifyError(@() D.L() ^ 0.5, 'OceanAgents:Guard:BadDimension');
            tc.verifyEqual(numel(D.L()), 1);                            % builtin numel/length unaffected
            tc.verifyEqual(length(D.L()), 1);
        end

        function componentTypeRules(tc)
            D = OceanAgents.World.Guard.Dimension;
            T = @(varargin) OceanAgents.World.Guard.ComponentType(varargin{:});
            speed = T('Dimension', D.L() / D.T());
            fuel = T('Dimension', D.E());
            tc.verifyTrue(speed.matches(T('Dimension', D.L() / D.T(), 'Visibility', "hidden")));   % visibility not compared
            tc.verifyFalse(speed.matches(fuel));
            tc.verifyEqual(speed.mismatch(fuel), "Dimension");
            tc.verifyEqual(speed.mismatch(T('Dimension', D.L() / D.T(), 'Width', 2)), "Width");
            charge = T('Kind', "linear", 'Dimension', D.E());
            tc.verifyTrue(charge.rateDimension() == D.E() / D.T());
            T('Kind', "clock", 'Dimension', D.T());                                            % valid clock
            id = 'OceanAgents:Guard:BadType';
            tc.verifyError(@() T('Kind', "clock", 'Dimension', D.L()), id);
            tc.verifyError(@() T('Class', "logical", 'Dimension', D.L()), id);
            tc.verifyError(@() T('Class', "string", 'Kind', "linear"), id);
            tc.verifyError(@() T('Width', 0), id);
        end

        function axiomForms(tc)
            A = @(varargin) OceanAgents.World.Guard.AxiomSpec(varargin{:});
            r = A('ID', "HB-1", 'Mode', "monitor", 'Component', "Speed", 'Range', [0 12]);
            tc.verifyTrue(r.isRange());
            tc.verifyEqual(r.readsAll(), "Speed");
            p = A('ID', "HB-2", 'Reads', "Mass", 'Predicate', @(v) v.Mass <= 300);
            tc.verifyFalse(p.isRange());
            A('ID', "M-9", 'Scope', "module", 'Mode', "assumed");
            id = 'OceanAgents:Guard:BadAxiom';
            tc.verifyError(@() A('ID', "x", 'Mode', "probe"), id);                               % entity axioms: static/monitor
            tc.verifyError(@() A('ID', "x", 'Scope', "module", 'Mode', "monitor", 'Predicate', @(m) true), id);
            tc.verifyError(@() A('ID', "x", 'Component', "S", 'Range', [0 1], 'Predicate', @(v) true), id);  % both
            tc.verifyError(@() A('ID', "x"), id);                                                % neither
            tc.verifyError(@() A('ID', "x", 'Component', "S", 'Range', [2 1]), id);              % lo > hi
            tc.verifyError(@() A('ID', "x", 'Scope', "module", 'Mode', "assumed", 'Predicate', @(m) true), id);
            tc.verifyError(@() A('ID', ""), id);
        end

        function certificateNeverOverclaims(tc)
            ax = struct('ID', {"M-1", "M-9"}, 'Mode', {"probe", "assumed"}, 'Status', {"probed", "assumed"}, ...
                'Probes', {32, 0}, 'Seeds', {0:31, []}, 'Detail', {"", ""});
            c = OceanAgents.World.Guard.Certificate('Subject', "Toy", 'ModuleVersion', "1", 'Door', "rl", ...
                'DoorVersion', "2", 'Axioms', ax);
            tc.verifyTrue(c.Admitted);
            tc.verifyTrue(c.validFor("1", "2"));
            tc.verifyFalse(c.validFor("1.1", "2"));
            tc.verifyFalse(c.validFor("1", "3"));
            bad = ax;  bad(2).Status = "probed";                                                 % assumed reported as checked
            tc.verifyError(@() OceanAgents.World.Guard.Certificate('Subject', "Toy", 'ModuleVersion', "1", 'Door', "rl", ...
                'DoorVersion', "2", 'Axioms', bad), 'OceanAgents:Guard:BadCertificate');
            f = ax;  f(1).Status = "failed";
            c2 = OceanAgents.World.Guard.Certificate('Subject', "Toy", 'ModuleVersion', "1", 'Door', "rl", ...
                'DoorVersion', "2", 'Axioms', f);
            tc.verifyFalse(c2.Admitted);
            tc.verifyFalse(c2.validFor("1", "2"));
            tc.verifyTrue(c.validFor("1", "2", 'Subject', "Toy", 'Door', "rl"));      % critic W1 F11
            tc.verifyFalse(c.validFor("1", "2", 'Subject', "OtherModule"));
            tc.verifyFalse(c.validFor("1", "2", 'Door', "otherdoor"));
            tc.verifyEqual(c2.failed(), "M-1");
        end

        function ruleAndDoorSpecs(tc)
            D = OceanAgents.World.Guard.Dimension;
            T = @(varargin) OceanAgents.World.Guard.ComponentType(varargin{:});
            R = @(varargin) OceanAgents.World.RuleSpec(varargin{:});
            mv = R('Name', "Move", 'Phase', "act", 'Requires', "Position", 'Reads', ["Position" "Velocity"], ...
                'Writes', "Position", 'Fn', @(ctx) struct(), ...
                'DimChecks', struct('Target', "Position", 'Terms', {{"Position", ["Velocity" "dt"]}}));
            tc.verifyEqual(mv.Kind, "rule");
            id = 'OceanAgents:World:BadRule';
            tc.verifyError(@() R('Name', "P", 'Phase', "project", 'Kind', "projection", 'Writes', "X", 'Fn', @(c) c), id);
            tc.verifyError(@() R('Name', "Q", 'Phase', "settle", 'Kind', "predictor", 'Fn', @(c) c), id);
            tc.verifyError(@() R('Name', "Q", 'Phase', "settle", 'DueEvent', "E", 'Fn', @(c) c), id);
            tc.verifyError(@() R('Name', "Q", 'Phase', "settle", 'Commands', "teleport", 'Fn', @(c) c), id);
            tc.verifyError(@() R('Name', "Q", 'Phase', "settle", 'Fn', "notafunction"), id);
            tc.verifyError(@() R('Name', "Q", 'Phase', "settle", 'Reads', "A", 'Reveals', "B", 'Fn', @(c) c), id);
            rv = R('Name', "Q", 'Phase', "settle", 'Reads', ["A" "B"], 'Reveals', "B", 'Fn', @(c) c);
            tc.verifyEqual(rv.Reveals, "B");
            child = OceanAgents.World.DoorSpec('Name', "collide", 'Owns', struct('Name', "Occupancy", 'Type', T('Class', "logical")));
            door = OceanAgents.World.DoorSpec('Name', "mover", 'Version', "3", ...
                'Owns', struct('Name', {"Position", "Velocity"}, 'Type', {T('Width', 2, 'Dimension', D.L()), ...
                                                                       T('Width', 2, 'Dimension', D.L() / D.T())}), ...
                'Resources', struct('Grid', [4 12]), 'Membership', ["Position" "Velocity"], 'Rules', mv, ...
                'Children', child);
            tc.verifyEqual(door.ownedNames(), ["Position" "Velocity" "Grid"]);
            [flat, depth] = door.flatten();
            tc.verifyEqual([flat.Name], ["mover" "collide"]);
            tc.verifyEqual(depth, [1 2]);
            bid = 'OceanAgents:World:BadAssembly';
            tc.verifyError(@() OceanAgents.World.DoorSpec('Name', "x", 'Owns', struct('Name', "not valid", 'Type', T())), bid);
            tc.verifyError(@() OceanAgents.World.DoorSpec('Name', "x", 'Drive', "sometimes"), bid);
            tc.verifyEqual(door.Stream, "keyed");
            lg = OceanAgents.World.DoorSpec('Name', "g", 'Stream', "legacy-mt19937ar");
            tc.verifyEqual(lg.Stream, "legacy-mt19937ar");
            tc.verifyError(@() OceanAgents.World.DoorSpec('Name', "x", 'Stream', "philox"), bid);
            tc.verifyError(@() OceanAgents.World.DoorSpec('Name', "x", 'Resources', struct('h', containers.Map())), ...
                'OceanAgents:Core:InvalidRecord');
        end

        function eventsAndCommands(tc)
            e = OceanAgents.World.Event.make("CliffFall", struct('Penalty', -100), 'SourceEntity', 12);
            tc.verifyTrue(isnan(e.Time));
            s = e.stamp(5, "act", "mover", 7);
            tc.verifyEqual([s.Time s.Seq s.SourceEntity], [5 7 12]);
            tc.verifyEqual(s.SourceDoor, "mover");
            tc.verifyError(@() OceanAgents.World.Event.make("x", struct('h', {{@sin}})), 'OceanAgents:Core:InvalidRecord');
            D = OceanAgents.World.Guard.Dimension;
            T = @(varargin) OceanAgents.World.Guard.ComponentType(varargin{:});
            offer = struct('Position', struct('Value', [2 3], 'Type', T('Width', 2, 'Dimension', D.L())), ...
                'Charge', struct('Value', struct('Value', 10, 'Rate', -0.5), 'Type', T('Kind', "linear", 'Dimension', D.E())));
            c = OceanAgents.World.Command.spawn(offer);
            tc.verifyEqual(c.Kind, "spawn");
            tc.verifyEmpty(c.Time);
            r = OceanAgents.World.Command.remove(4, 'At', 370, 'Guard', @(v) v.TargetAlive);
            tc.verifyEqual([r.Target r.Time], [4 370]);
            st = r.stamp("battery", 4, 9, 370, "settle");
            tc.verifyEqual([st.Seq st.IssuerEntity], [9 4]);
            id = 'OceanAgents:World:BadValue';
            tc.verifyError(@() OceanAgents.World.Command.remove(0), id);
            tc.verifyError(@() OceanAgents.World.Command.spawn(struct()), id);
            tc.verifyError(@() OceanAgents.World.Command.spawn(struct('P', struct('Value', 1))), id);   % no Type
            tc.verifyError(@() OceanAgents.World.Command.detach(3, strings(1, 0)), id);
            tc.verifyError(@() OceanAgents.World.Command.spawn(struct('P', struct('Value', {{containers.Map()}}, ...
                'Type', T()))), 'OceanAgents:Core:InvalidRecord');
        end
    end
end
