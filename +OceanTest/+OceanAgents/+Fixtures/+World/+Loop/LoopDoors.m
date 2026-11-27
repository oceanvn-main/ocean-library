classdef LoopDoors
    % LOOPDOORS  Test-only doors WITH working rules for the World loop tests (W1_Kernel_Contract.md section 10).
    %
    %   L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
    %   w = OceanAgents.World.World([L.mover(), L.battery(), L.ager(), L.spawner(5, ["Position" "Velocity"])], ...
    %           'Injectable', "Ping");
    %
    %   mover      Position (L, width 2), Velocity (L T^-1, width 2), every-tick. Move (act): Position += Velocity*dt
    %              (DimChecks). Optional Kick (resolve): random velocity changes from the door stream. Projection See.
    %   battery    Charge: linear, E. Predictor ZeroCrossing: t* = ceil(now + Charge/(-Rate)) for Rate < 0 (Inf
    %              otherwise). Drain (resolve) consumes Empty and removes the entity (guarded: TargetAlive).
    %              Monitor BT-1: Charge >= 0 (range; skip-safe: the kernel schedules a check at its crossing). The
    %              entity removed at its (ceil) predicted crossing is not monitored there (post-command monitors).
    %              Optional Tweak (act, every-k): random rebases of the rate from the door stream.
    %              Optional RebaseAt t: at instant t set every rate to -1 (keeping the value) - K7 hand example.
    %   ager       Born: clock (T). Monitor AG-1: age <= 100 - a RANGE axiom on the clock by default (skip-safe without
    %              a predictor), or 'Monitor', "predicate" (then a predictor is required when skipping).
    %              Optional predictor OldAge (Born + 100) with Retire.
    %   tank       Level: linear, dimensionless, NO predictor and no rules; range monitor TK-1: Level in [0, Inf]
    %              (skip-safe through the kernel's scheduled monitor checks alone).
    %   spawner    every-k: Spawn (settle) issues 0..2 spawns with random offers for the listed components.
    %   counter    on-event: Count (resolve) adds the number of delivered Ping events to resource Pings.
    %   echo/listener  K4: echo emits Tock in "act" every tick; listener logs deliveries in resolve/act/settle.
    %   swap       K4: IncX and CopyX in one phase, in either order.
    % Monitor/axiom IDs: BT-1, AG-1.

    methods (Static)
        function t = type(varargin)
            t = OceanAgents.World.Guard.ComponentType(varargin{:});
        end

        function T = types()
            % The component types the doors declare (spawn offers must use the same ones).
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            D = OceanAgents.World.Guard.Dimension;
            T = struct('Position', L.type('Width', 2, 'Dimension', D.L()), ...
                'Velocity', L.type('Width', 2, 'Dimension', D.L() / D.T()), ...
                'Charge', L.type('Kind', "linear", 'Dimension', D.E()), ...
                'Level', L.type('Kind', "linear"), ...
                'Born', L.type('Kind', "clock", 'Dimension', D.T()));
        end

        % ------------------------------------------------------------ mover
        function d = mover(options)
            arguments
                options.Kick (1,1) logical = false
                options.Init = []
                options.Visibility (1,1) string = "public"
            end
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            T = L.types();
            pos = L.type('Width', 2, 'Dimension', T.Position.Dimension, 'Visibility', options.Visibility);
            rules = OceanAgents.World.RuleSpec('Name', "Move", 'Phase', "act", 'Requires', ["Position" "Velocity"], ...
                'Reads', ["Position" "Velocity"], 'Writes', "Position", ...
                'DimChecks', struct('Target', "Position", 'Terms', {{"Position", ["Velocity" "dt"]}}), ...
                'Fn', @OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors.moveFn);
            if options.Kick
                rules = [OceanAgents.World.RuleSpec('Name', "Kick", 'Phase', "resolve", 'Requires', "Velocity", ...
                    'Reads', "Velocity", 'Writes', "Velocity", 'Fn', @OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors.kickFn), rules];
            end
            rules(end + 1) = OceanAgents.World.RuleSpec('Name', "See", 'Phase', "project", 'Kind', "projection", ...
                'Requires', "Position", 'Reads', "Position", 'Fn', @(ctx) struct('View', ...
                struct('IDs', ctx.IDs, 'Position', ctx.Read.Position)));
            d = OceanAgents.World.DoorSpec('Name', "mover", 'Owns', struct('Name', {"Position", "Velocity"}, ...
                'Type', {pos, T.Velocity}), 'Membership', ["Position" "Velocity"], 'Rules', rules, ...
                'Drive', "every-tick", 'Init', options.Init);
        end

        function out = moveFn(ctx)
            % Position <- Position + Velocity * dt  (dt = 1 tick)
            out = struct('Writes', struct('Component', "Position", 'Rows', ctx.Rows, ...
                'Values', ctx.Read.Position + ctx.Read.Velocity * 1));
        end

        function out = kickFn(ctx)
            n = numel(ctx.Rows);
            u = rand(ctx.Stream, n, 1);
            sel = u < 0.2;
            out = struct();
            if ~any(sel), return; end
            v = ctx.Read.Velocity(sel, :) + randi(ctx.Stream, [-1 1], nnz(sel), 2);
            out.Writes = struct('Component', "Velocity", 'Rows', ctx.Rows(sel), 'Values', v);
        end

        % ------------------------------------------------------------ battery
        function d = battery(options)
            arguments
                options.Tweak (1,1) logical = false
                options.DriveK (1,1) double = 1
                options.RebaseAt double = []
                options.Init = []
            end
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            T = L.types();
            rules = [OceanAgents.World.RuleSpec('Name', "ZeroCrossing", 'Phase', "settle", 'Kind', "predictor", ...
                'Requires', "Charge", 'Reads', "Charge", 'Predicts', "Charge", 'DueEvent', "Empty", 'Fn', @OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors.zeroCrossingFn), ...
                OceanAgents.World.RuleSpec('Name', "Drain", 'Phase', "resolve", 'Consumes', "Empty", ...
                'Commands', "remove", 'Fn', @OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors.drainFn)];
            drive = "on-event";
            if options.Tweak
                rules(end + 1) = OceanAgents.World.RuleSpec('Name', "Tweak", 'Phase', "act", 'Requires', "Charge", ...
                    'Reads', "Charge", 'Writes', "Charge", 'Fn', @OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors.tweakFn);
                drive = "every-k";
            end
            if ~isempty(options.RebaseAt)
                at = options.RebaseAt;
                rules(end + 1) = OceanAgents.World.RuleSpec('Name', "Rebase", 'Phase', "act", 'Requires', "Charge", ...
                    'Reads', "Charge", 'Writes', "Charge", 'Fn', @(ctx) L.rebaseFn(ctx, at));
                drive = "every-tick";
            end
            ax = OceanAgents.World.Guard.AxiomSpec('ID', "BT-1", 'Mode', "monitor", 'Component', "Charge", ...
                'Range', [0 Inf], 'Description', "charge is never negative");
            d = OceanAgents.World.DoorSpec('Name', "battery", 'Owns', struct('Name', "Charge", 'Type', T.Charge), ...
                'Membership', "Charge", 'Rules', rules, 'Axioms', ax, 'Drive', drive, 'DriveK', options.DriveK, ...
                'Init', options.Init);
        end

        function out = zeroCrossingFn(ctx)
            % Charge(t) = c + r (t - now) hits 0 at now + c/(-r) for r < 0; integer ticks: ceil (first tick at or
            % after the crossing).
            c = ctx.Read.Charge;
            r = ctx.Read.Charge_Rate;
            times = inf(numel(ctx.Rows), 1);
            falling = r < 0;
            times(falling) = ceil(ctx.Now + c(falling) ./ (-r(falling)));
            out = struct('Predictions', struct('Rows', ctx.Rows, 'Times', times));
        end

        function out = drainFn(ctx)
            ids = reshape(unique([ctx.Events.SourceEntity]), 1, []);
            cmds = OceanAgents.World.Command.empty(1, 0);
            for id = ids
                cmds(end + 1) = OceanAgents.World.Command.remove(id, 'Guard', @(v) v.TargetAlive); %#ok<AGROW>
            end
            out = struct('Commands', cmds);
        end

        function out = tweakFn(ctx)
            n = numel(ctx.Rows);
            sel = rand(ctx.Stream, n, 1) < 0.3;
            out = struct();
            if ~any(sel), return; end
            rates = -(0.1 + 0.8 * rand(ctx.Stream, nnz(sel), 1));
            out.Writes = struct('Component', "Charge", 'Rows', ctx.Rows(sel), 'Values', ctx.Read.Charge(sel), ...
                'Rates', rates);
        end

        function out = rebaseFn(ctx, at)
            out = struct();
            if ctx.Now ~= at, return; end
            out.Writes = struct('Component', "Charge", 'Rows', ctx.Rows, 'Values', ctx.Read.Charge, ...
                'Rates', -ones(numel(ctx.Rows), 1));
        end

        % ------------------------------------------------------------ ager
        function d = ager(options)
            arguments
                options.Retire (1,1) logical = true
                options.Monitor (1,1) string {mustBeMember(options.Monitor, ["range" "predicate"])} = "range"
                options.Init = []
            end
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            T = L.types();
            rules = OceanAgents.World.RuleSpec.empty(1, 0);
            if options.Retire
                rules = [OceanAgents.World.RuleSpec('Name', "OldAge", 'Phase', "settle", 'Kind', "predictor", ...
                    'Requires', "Born", 'Reads', "Born", 'Predicts', "Born", 'DueEvent', "Old", ...
                    'Fn', @(ctx) struct('Predictions', struct('Rows', ctx.Rows, 'Times', ctx.Now + (100 - ctx.Read.Born)))), ...
                    OceanAgents.World.RuleSpec('Name', "Retire", 'Phase', "resolve", 'Consumes', "Old", ...
                    'Commands', "remove", 'Fn', @OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors.drainFn)];
            end
            if options.Monitor == "range"
                ax = OceanAgents.World.Guard.AxiomSpec('ID', "AG-1", 'Mode', "monitor", 'Component', "Born", ...
                    'Range', [-Inf 100], 'Description', "age <= 100");
            else
                ax = OceanAgents.World.Guard.AxiomSpec('ID', "AG-1", 'Mode', "monitor", 'Reads', "Born", ...
                    'Predicate', @(v) v.Born <= 100, 'Description', "age <= 100");
            end
            d = OceanAgents.World.DoorSpec('Name', "ager", 'Owns', struct('Name', "Born", 'Type', T.Born), ...
                'Membership', "Born", 'Rules', rules, 'Axioms', ax, 'Drive', "never", 'Init', options.Init);
        end

        % ------------------------------------------------------------ tank
        function d = tank(options)
            arguments
                options.Init = []
            end
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            T = L.types();
            ax = OceanAgents.World.Guard.AxiomSpec('ID', "TK-1", 'Mode', "monitor", 'Component', "Level", ...
                'Range', [0 Inf], 'Description', "level never negative");
            d = OceanAgents.World.DoorSpec('Name', "tank", 'Owns', struct('Name', "Level", 'Type', T.Level), ...
                'Membership', "Level", 'Axioms', ax, 'Drive', "never", 'Init', options.Init);
        end

        % ------------------------------------------------------------ spawner
        function d = spawner(k, components, name)
            % every-k door spawning 0..2 entities per firing with random offers of the given components.
            if nargin < 3, name = "spawner"; end
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            comps = string(components);
            rule = OceanAgents.World.RuleSpec('Name', "Spawn", 'Phase', "settle", 'Commands', "spawn", ...
                'Fn', @(ctx) L.spawnFn(ctx, comps));
            d = OceanAgents.World.DoorSpec('Name', name, 'Rules', rule, 'Drive', "every-k", 'DriveK', k);
        end

        function out = spawnFn(ctx, comps)
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            n = randi(ctx.Stream, [0 2]);
            cmds = OceanAgents.World.Command.empty(1, 0);
            for k = 1:n
                cmds(end + 1) = OceanAgents.World.Command.spawn(L.randomOffer(ctx.Stream, comps, ctx.Now)); %#ok<AGROW>
            end
            out = struct('Commands', cmds);
        end

        function o = randomOffer(stream, comps, now)
            % Random, admissible values (all draws from the given stream, in a fixed order).
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            T = L.types();
            o = struct();
            for c = comps
                switch c
                    case "Position", v = randi(stream, [0 50], 1, 2);
                    case "Velocity", v = randi(stream, [-2 2], 1, 2);
                    case "Charge",   v = struct('Value', randi(stream, [1 20]), 'Rate', -(0.1 + 0.8 * rand(stream)));
                    case "Born",     v = now - randi(stream, [0 60]);
                    case "Level",    v = struct('Value', randi(stream, [5 30]), 'Rate', -(0.1 + 0.8 * rand(stream)));
                end
                o.(char(c)) = struct('Value', v, 'Type', T.(char(c)));
            end
        end

        function cmds = initSpawns(offers)
            % Init function body: one spawn per offer (cell of offers).
            cmds = OceanAgents.World.Command.empty(1, 0);
            for k = 1:numel(offers)
                cmds(end + 1) = OceanAgents.World.Command.spawn(offers{k}); %#ok<AGROW>
            end
        end

        function o = offer(varargin)
            % offer('Position', [x y], 'Charge', struct('Value', 10, 'Rate', -0.5), ...) with the declared types.
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors;
            T = L.types();
            o = struct();
            for k = 1:2:numel(varargin)
                o.(varargin{k}) = struct('Value', varargin{k + 1}, 'Type', T.(varargin{k}));
            end
        end

        % ------------------------------------------------------------ counter
        function d = counter()
            L = OceanTest.OceanAgents.Fixtures.World.Loop.LoopDoors; %#ok<NASGU>
            count = OceanAgents.World.RuleSpec('Name', "Count", 'Phase', "resolve", 'Consumes', "Ping", ...
                'Reads', "Pings", 'Writes', "Pings", ...
                'Fn', @(ctx) struct('Writes', struct('Resource', "Pings", 'Values', ctx.Read.Pings + numel(ctx.Events))));
            d = OceanAgents.World.DoorSpec('Name', "counter", 'Resources', struct('Pings', 0), 'Rules', count, ...
                'Drive', "on-event");
        end

        % ------------------------------------------------------------ K4 doors
        function d = echo()
            emit = OceanAgents.World.RuleSpec('Name', "Emit", 'Phase', "act", 'Emits', "Tock", ...
                'Fn', @(ctx) struct('Events', OceanAgents.World.Event.make("Tock", struct('at', ctx.Now))));
            d = OceanAgents.World.DoorSpec('Name', "echo", 'Rules', emit, 'Drive', "every-tick");
        end

        function d = listener()
            mk = @(name, phase, res) OceanAgents.World.RuleSpec('Name', name, 'Phase', phase, 'Consumes', "Tock", ...
                'Reads', res, 'Writes', res, 'Fn', @(ctx) struct('Writes', struct('Resource', res, 'Values', ...
                [ctx.Read.(char(res)); repmat(ctx.Now, numel(ctx.Events), 1) [ctx.Events.Time]'])));
            d = OceanAgents.World.DoorSpec('Name', "listener", 'Resources', struct('EarlierLog', zeros(0, 2), ...
                'SameLog', zeros(0, 2), 'LaterLog', zeros(0, 2)), 'Drive', "on-event", ...
                'Rules', [mk("SeeEarlier", "resolve", "EarlierLog"), mk("SeeSame", "act", "SameLog"), ...
                mk("SeeLater", "settle", "LaterLog")]);
        end

        function d = swap(order)
            % IncX: X <- X + 1 and CopyX: Y <- X in the SAME phase; order = [1 2] or [2 1].
            inc = OceanAgents.World.RuleSpec('Name', "IncX", 'Phase', "act", 'Reads', "X", 'Writes', "X", ...
                'Fn', @(ctx) struct('Writes', struct('Resource', "X", 'Values', ctx.Read.X + 1)));
            cp = OceanAgents.World.RuleSpec('Name', "CopyX", 'Phase', "act", 'Reads', "X", 'Writes', "Y", ...
                'Fn', @(ctx) struct('Writes', struct('Resource', "Y", 'Values', ctx.Read.X)));
            rules = [inc cp];
            d = OceanAgents.World.DoorSpec('Name', "swap", 'Resources', struct('X', 0, 'Y', -1), ...
                'Rules', rules(order), 'Drive', "every-tick");
        end
    end
end
