classdef GuardDoors
    % GUARDDOORS  Test-only doors for the guard tests (W1_Kernel_Contract.md section 10). Rule functions are inert
    % (@(ctx) struct()): the guard never runs rules, it only checks declarations, types, values and axioms.
    %
    %   G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
    %   doors = G.kernelAssembly();        % mover, battery, ager, counter: a VALID assembly (phases G.Phases,
    %                                      % injectable G.Injectable)
    %   [h, r] = deal(G.human(true), G.road());   % the car scenario of design section 9A.6
    %   offer  = G.carOffer();             % Speed 30 m/s, Mass 1200 kg, Position [5 0] m
    %
    % Axiom IDs used: HB-1 (Speed in [0, 12] m/s, range, monitor), HB-2 (Mass <= 300 kg, predicate, static),
    % RD-1 (Position in [0, 1000] m, range, monitor), AG-1 (age <= 100, predicate on a clock, monitor),
    % TM-1..TM-5 (toy module axioms, see ToyAxioms).

    properties (Constant)
        Phases = ["resolve" "act" "settle" "project"]
        Injectable = "Ping"
    end

    methods (Static)
        function t = type(varargin)
            t = OceanAgents.World.Guard.ComponentType(varargin{:});
        end

        function D = dim()
            D = OceanAgents.World.Guard.Dimension;
        end

        function d = mover(dimTerms)
            % Position (L, width 2), Velocity (L T^-1, width 2); Move: Position <- Position + Velocity*dt.
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            D = G.dim();
            if nargin < 1, dimTerms = {"Position", ["Velocity" "dt"]}; end
            move = OceanAgents.World.RuleSpec('Name', "Move", 'Phase', "act", 'Requires', ["Position" "Velocity"], ...
                'Reads', ["Position" "Velocity"], 'Writes', "Position", ...
                'DimChecks', struct('Target', "Position", 'Terms', {dimTerms}), 'Fn', @(ctx) struct());
            view = OceanAgents.World.RuleSpec('Name', "See", 'Phase', "project", 'Kind', "projection", ...
                'Requires', "Position", 'Reads', "Position", 'Fn', @(ctx) struct());
            d = OceanAgents.World.DoorSpec('Name', "mover", 'Version', "1", ...
                'Owns', struct('Name', {"Position", "Velocity"}, 'Type', {G.type('Width', 2, 'Dimension', D.L()), ...
                G.type('Width', 2, 'Dimension', D.L() / D.T())}), ...
                'Membership', ["Position" "Velocity"], 'Rules', [move view], 'Drive', "every-tick");
        end

        function d = battery()
            % Charge: linear, E; predictor for the zero crossing (DueEvent Empty); Drain removes the empty entity.
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            D = G.dim();
            predict = OceanAgents.World.RuleSpec('Name', "ZeroCrossing", 'Phase', "settle", 'Kind', "predictor", ...
                'Requires', "Charge", 'Reads', "Charge", 'Predicts', "Charge", 'DueEvent', "Empty", 'Fn', @(ctx) struct());
            drain = OceanAgents.World.RuleSpec('Name', "Drain", 'Phase', "resolve", 'Consumes', "Empty", ...
                'Commands', "remove", 'Fn', @(ctx) struct());
            d = OceanAgents.World.DoorSpec('Name', "battery", 'Version', "1", ...
                'Owns', struct('Name', "Charge", 'Type', G.type('Kind', "linear", 'Dimension', D.E())), ...
                'Membership', "Charge", 'Rules', [predict drain], 'Drive', "on-event", ...
                'Axioms', OceanAgents.World.Guard.AxiomSpec('ID', "BT-1", 'Mode', "monitor", 'Component', "Charge", ...
                'Range', [-1e-9 Inf], 'Description', "charge is never negative"));
        end

        function d = ager()
            % Born: clock. Monitor axiom AG-1: age = now - Born <= 100.
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            D = G.dim();
            ax = OceanAgents.World.Guard.AxiomSpec('ID', "AG-1", 'Mode', "monitor", 'Reads', "Born", ...
                'Predicate', @(v) v.Born <= 100, 'Description', "age <= 100");
            d = OceanAgents.World.DoorSpec('Name', "ager", 'Version', "1", ...
                'Owns', struct('Name', "Born", 'Type', G.type('Kind', "clock", 'Dimension', D.T())), ...
                'Membership', "Born", 'Axioms', ax, 'Drive', "never");
        end

        function d = counter()
            % on-event door: counts delivered Ping events in the resource Pings (Ping is injectable).
            count = OceanAgents.World.RuleSpec('Name', "Count", 'Phase', "resolve", 'Consumes', "Ping", ...
                'Reads', "Pings", 'Writes', "Pings", 'Fn', @(ctx) struct());
            d = OceanAgents.World.DoorSpec('Name', "counter", 'Version', "1", 'Resources', struct('Pings', 0), ...
                'Rules', count, 'Drive', "on-event");
        end

        function doors = kernelAssembly()
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            doors = [G.mover(), G.battery(), G.ager(), G.counter()];
        end

        function d = human(withAxioms, extraAxioms)
            % Human behaviour door: Speed (L T^-1), Mass (M). HB-1 range Speed in [0, 12]; HB-2 predicate Mass <= 300.
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            D = G.dim();
            if nargin < 1, withAxioms = true; end
            ax = OceanAgents.World.Guard.AxiomSpec.empty(1, 0);
            if withAxioms
                ax = [OceanAgents.World.Guard.AxiomSpec('ID', "HB-1", 'Mode', "monitor", 'Component', "Speed", ...
                    'Range', [0 12], 'Description', "self-propelled speed at most 12 m/s"), ...
                    OceanAgents.World.Guard.AxiomSpec('ID', "HB-2", 'Mode', "static", 'Reads', "Mass", ...
                    'Predicate', @(v) v.Mass <= 300, 'Description', "body mass at most 300 kg")];
            end
            if nargin >= 2, ax = [ax extraAxioms]; end
            d = OceanAgents.World.DoorSpec('Name', "human", 'Version', "1", ...
                'Owns', struct('Name', {"Speed", "Mass"}, 'Type', {G.type('Dimension', D.L() / D.T()), ...
                G.type('Dimension', D.M())}), 'Membership', ["Speed" "Mass"], 'Axioms', ax, 'Drive', "never");
        end

        function d = road(extraAxioms)
            % Road door: Position (L, width 2). RD-1 range Position in [0, 1000] m.
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            D = G.dim();
            ax = OceanAgents.World.Guard.AxiomSpec('ID', "RD-1", 'Mode', "monitor", 'Component', "Position", ...
                'Range', [0 1000], 'Description', "on the modelled road segment");
            if nargin >= 1, ax = [ax extraAxioms]; end
            d = OceanAgents.World.DoorSpec('Name', "road", 'Version', "1", ...
                'Owns', struct('Name', "Position", 'Type', G.type('Width', 2, 'Dimension', D.L())), ...
                'Membership', "Position", 'Axioms', ax, 'Drive', "never");
        end

        function o = offer(speed, mass, position)
            % Spawn offer in the Command.Offer format for the car scenario.
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            D = G.dim();
            o = struct('Speed', struct('Value', speed, 'Type', G.type('Dimension', D.L() / D.T())), ...
                'Mass', struct('Value', mass, 'Type', G.type('Dimension', D.M())), ...
                'Position', struct('Value', position, 'Type', G.type('Width', 2, 'Dimension', D.L())));
        end

        function o = carOffer()
            o = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors.offer(30, 1200, [5 0]);
        end

        function o = walkerOffer()
            o = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors.offer(1.4, 70, [3 0]);
        end

        function s = storeFor(doors)
            % A real OceanAgents.World.Store holding every component the doors own.
            types = struct();
            for d = doors
                for c = d.Owns
                    types.(char(c.Name)) = c.Type;
                end
            end
            s = OceanAgents.World.Store(types);
        end

        function d = modules(varargin)
            % Door certifying ToyModule implementations with axioms TM-1..TM-5 (extra axioms may be appended).
            A = @(varargin) OceanAgents.World.Guard.AxiomSpec(varargin{:});
            X = 'OceanTest.OceanAgents.Fixtures.World.Guard.ToyAxioms';
            ax = [A('ID', "TM-1", 'Scope', "module", 'Kind', "invariant", 'Mode', "static", ...
                    'Predicate', str2func([X '.capacityPositive'])), ...
                  A('ID', "TM-2", 'Scope', "module", 'Kind', "postcondition", 'Mode', "probe", 'Probes', 16, ...
                    'Predicate', str2func([X '.applyWellFormed'])), ...
                  A('ID', "TM-3", 'Scope', "module", 'Kind', "trajectory", 'Mode', "probe", 'Probes', 8, ...
                    'Predicate', str2func([X '.sameSeedSameSamples'])), ...
                  A('ID', "TM-4", 'Scope', "module", 'Kind', "invariant", 'Mode', "probe", 'Probes', 4, ...
                    'Predicate', str2func([X '.globalStreamUntouched'])), ...
                  A('ID', "TM-5", 'Scope', "module", 'Kind', "invariant", 'Mode', "assumed", ...
                    'Description', "samples are meaningful for the task (not checkable)")];
            opts = struct('Version', "1", 'Extra', OceanAgents.World.Guard.AxiomSpec.empty(1, 0));
            for k = 1:2:numel(varargin), opts.(varargin{k}) = varargin{k + 1}; end
            d = OceanAgents.World.DoorSpec('Name', "modules", 'Version', opts.Version, ...
                'Admits', "OceanTest.OceanAgents.Fixtures.World.Guard.ToyModule", 'Axioms', [ax opts.Extra], ...
                'Drive', "never");
        end
    end
end
