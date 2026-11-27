classdef AssemblyValidationTest < OceanTest.BaseTestCase
    % ASSEMBLYVALIDATIONTEST  GuardEngine.validateAssembly: construction checks V1..V10 with the section 11 error IDs
    % and messages naming the door / rule / axiom (W1_Kernel_Contract.md section 8; K1, K19).

    properties (Constant)
        G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors
    end

    methods (Static, Access = private)
        function validate(doors, phases, injectable)
            G = OceanTest.OceanAgents.Fixtures.World.Guard.GuardDoors;
            if nargin < 2, phases = G.Phases; end
            if nargin < 3, injectable = G.Injectable; end
            OceanAgents.World.Guard.GuardEngine.validateAssembly(doors, phases, injectable);
        end

        function err = failure(fn)
            err = [];
            try
                fn();
            catch err
            end
        end

        function r = rule(varargin)
            r = OceanAgents.World.RuleSpec('Phase', "act", 'Fn', @(ctx) struct(), varargin{:});
        end

        function d = door(name, varargin)
            d = OceanAgents.World.DoorSpec('Name', name, varargin{:});
        end
    end

    methods
        function verifyFails(tc, fn, id, varargin)
            % The error ID and every expected name in the message.
            err = tc.failure(fn);
            tc.assertNotEmpty(err, sprintf('expected %s', id));
            tc.verifyEqual(err.identifier, id);
            for k = 1:numel(varargin)
                tc.verifySubstring(err.message, char(varargin{k}));
            end
        end
    end

    methods (Test)
        function validAssemblyPasses(tc)
            tc.validate(tc.G.kernelAssembly());
            tc.validate([tc.G.human(), tc.G.road()]);
            % sub-doors: a parent with one child (depth 2) is fine
            child = tc.door("sensor", 'Owns', struct('Name', "Lux", 'Type', tc.G.type()), 'Membership', "Lux");
            parent = tc.door("room", 'Owns', struct('Name', "Temp", 'Type', tc.G.type()), 'Children', child);
            flat = OceanAgents.World.Guard.GuardEngine.flattenDoors(parent);
            tc.verifyEqual([flat.Name], ["room" "sensor"]);
            tc.validate(flat);
        end

        function v1Assembly(tc)
            id = 'OceanAgents:World:BadAssembly';
            doors = tc.G.kernelAssembly();
            tc.verifyFails(@() tc.validate([doors doors(1)]), id, "mover");
            tc.verifyFails(@() tc.validate(doors, strings(1, 0)), id, "phase");
            tc.verifyFails(@() tc.validate(doors, ["act" "act"]), id, "act");
            grand = tc.door("leaf", 'Owns', struct('Name', "Leaf", 'Type', tc.G.type()));
            mid = tc.door("mid", 'Children', grand);
            top = tc.door("top", 'Children', mid);
            tc.verifyFails(@() tc.validate(OceanAgents.World.Guard.GuardEngine.flattenDoors(top)), id, "top");
            % a Membership naming something that is not a component
            tc.verifyFails(@() tc.validate([doors tc.door("ghost", 'Membership', "Nothing")]), id, "ghost", "Nothing");
        end

        function v2Ownership(tc)
            id = 'OceanAgents:World:DuplicateOwner';
            tc.verifyFails(@() tc.validate([tc.G.mover(), tc.G.road()]), id, "Position", "mover", "road");
            clash = tc.door("clash", 'Resources', struct('Charge', 0));        % resource named like a component
            tc.verifyFails(@() tc.validate([tc.G.kernelAssembly() clash]), id, "Charge", "battery", "clash");
            dt = tc.door("timer", 'Owns', struct('Name', "dt", 'Type', tc.G.type()));
            tc.verifyFails(@() tc.validate(dt), id, "dt");
        end

        function v3Rules(tc)
            id = 'OceanAgents:World:BadRule';
            base = tc.G.kernelAssembly();
            steal = tc.door("thief", 'Rules', tc.rule('Name', "Steal", 'Writes', "Charge"));
            tc.verifyFails(@() tc.validate([base steal]), id, "thief", "Steal", "Charge");
            blind = tc.door("blind", 'Rules', tc.rule('Name', "Peek", 'Reads', "Nowhere"));
            tc.verifyFails(@() tc.validate([base blind]), id, "blind", "Peek", "Nowhere");
            needy = tc.door("needy", 'Rules', tc.rule('Name', "Need", 'Requires', "Pings"));   % a resource
            tc.verifyFails(@() tc.validate([base needy]), id, "Need", "Pings");
            lost = tc.door("lost", 'Rules', tc.rule('Name', "Wander", 'Phase', "dream"));
            tc.verifyFails(@() tc.validate([base lost]), id, "Wander", "dream");
            twice = tc.door("twice", 'Rules', [tc.rule('Name', "Same"), tc.rule('Name', "Same")]);
            tc.verifyFails(@() tc.validate([base twice]), id, "twice");
            % reads of any door's components and resources are allowed
            ok = tc.door("reader", 'Rules', tc.rule('Name', "Look", 'Reads', ["Charge" "Pings" "Born"]));
            tc.validate([base ok]);
        end

        function v4AndV5Events(tc)
            base = tc.G.kernelAssembly();
            loud = tc.door("loud", 'Rules', tc.rule('Name', "Shout", 'Emits', "Boom"));
            tc.verifyFails(@() tc.validate([base loud]), 'OceanAgents:World:UnconsumedEvent', "loud", "Shout", "Boom");
            ear = tc.door("ear", 'Rules', tc.rule('Name', "Hear", 'Consumes', "Boom"));
            tc.validate([base loud ear]);
            tc.verifyFails(@() tc.validate(base, tc.G.Phases, strings(1, 0)), 'OceanAgents:World:UnknownEvent', ...
                "counter", "Count", "Ping");
            % a predictor's DueEvent needs a consumer too
            noDrain = tc.door("battery2", 'Owns', struct('Name', "Charge2", 'Type', tc.G.type('Kind', "linear")), ...
                'Rules', OceanAgents.World.RuleSpec('Name', "Z", 'Phase', "settle", 'Kind', "predictor", ...
                'Predicts', "Charge2", 'DueEvent', "Empty2", 'Fn', @(c) struct()));
            tc.verifyFails(@() tc.validate([base noDrain]), 'OceanAgents:World:UnconsumedEvent', "Empty2");
        end

        function v6HiddenLeak(tc)
            T = @(varargin) tc.G.type(varargin{:});
            secret = tc.door("vault", 'Owns', struct('Name', {"Gold", "Tag", "Mood"}, ...
                'Type', {T('Visibility', "hidden"), T('Visibility', "public"), T('Visibility', "self")}), ...
                'Rules', OceanAgents.World.RuleSpec('Name', "OwnView", 'Phase', "project", 'Kind', "projection", ...
                'Reads', "Gold", 'Fn', @(c) struct()));
            % F7: no projection reads a hidden component, not even its own door's
            tc.verifyFails(@() tc.validate(secret), 'OceanAgents:Guard:HiddenLeak', "vault", "OwnView", "Gold");
            secret = tc.door("vault", 'Owns', struct('Name', {"Gold", "Tag", "Mood"}, ...
                'Type', {T('Visibility', "hidden"), T('Visibility', "public"), T('Visibility', "self")}), ...
                'Rules', OceanAgents.World.RuleSpec('Name', "OwnView", 'Phase', "project", 'Kind', "projection", ...
                'Reads', ["Tag" "Mood"], 'Fn', @(c) struct()));
            tc.validate(secret);
            spy = tc.door("spy", 'Rules', OceanAgents.World.RuleSpec('Name', "Snoop", 'Phase', "project", ...
                'Kind', "projection", 'Reads', ["Tag" "Gold"], 'Fn', @(c) struct()));
            tc.verifyFails(@() tc.validate([secret spy]), 'OceanAgents:Guard:HiddenLeak', "spy", "Snoop", "Gold", "vault");
            fair = tc.door("fair", 'Rules', OceanAgents.World.RuleSpec('Name', "Look", 'Phase', "project", ...
                'Kind', "projection", 'Reads', ["Tag" "Mood"], 'Fn', @(c) struct()));
            tc.validate([secret fair]);
        end

        function projectionRequiresOnlyPublicN3(tc)
            % ctx.Rows/IDs of a projection reveal which entities hold its Requires: only PUBLIC components allowed.
            T = @(varargin) tc.G.type(varargin{:});
            proj = @(name, req) OceanAgents.World.RuleSpec('Name', name, 'Phase', "project", 'Kind', "projection", ...
                'Requires', req, 'Fn', @(c) struct());
            vault = tc.door("vault", 'Owns', struct('Name', {"Secret", "Pub", "Mine"}, ...
                'Type', {T('Visibility', "hidden"), T(), T('Visibility', "self")}));
            id = 'OceanAgents:Guard:HiddenLeak';
            tc.verifyFails(@() tc.validate([vault tc.door("spy", 'Rules', proj("Look", "Secret"))]), id, ...
                "V6", "spy", "Look", "Secret", "vault", "hidden");
            tc.verifyFails(@() tc.validate([vault tc.door("spy", 'Rules', proj("LookSelf", "Mine"))]), id, ...
                "LookSelf", "Mine", "self");
            own = tc.door("vault", 'Owns', vault.Owns, 'Rules', proj("OwnLook", "Secret"));   % own door included
            tc.verifyFails(@() tc.validate(own), id, "OwnLook", "Secret");
            tc.validate([vault tc.door("spy", 'Rules', proj("LookPub", "Pub"))]);
        end

        function v12CoversRequiresAndPredictsN3N4(tc)
            T = @(varargin) tc.G.type(varargin{:});
            id = 'OceanAgents:Guard:HiddenLeak';
            vault = tc.door("vault", 'Owns', struct('Name', {"Secret", "Hid"}, ...
                'Type', {T('Visibility', "hidden"), T('Kind', "linear", 'Visibility', "hidden")}));
            % N3: a rule selecting its rows by another door's hidden component (the critic's "Mark" attach rule)
            mark = tc.door("spy", 'Owns', struct('Name', "Flag", 'Type', T()), ...
                'Rules', tc.rule('Name', "Mark", 'Requires', "Secret", 'Commands', "attach"));
            tc.verifyFails(@() tc.validate([vault mark]), id, "V12", "Mark", "spy", "requires", "Secret", "vault", "Reads");
            declared = tc.door("spy", 'Owns', struct('Name', "Flag", 'Type', T()), 'Rules', tc.rule('Name', "Mark", ...
                'Requires', "Secret", 'Reads', "Secret", 'Reveals', "Secret", 'Commands', "attach"));
            tc.validate([vault declared]);
            ownReq = tc.door("vault2", 'Owns', struct('Name', "S2", 'Type', T('Visibility', "hidden")), ...
                'Rules', tc.rule('Name', "Mine", 'Requires', "S2"));
            tc.validate(ownReq);                                               % own door: no Reveals needed
            % N4: the critic's "spy" predictor predicting another door's hidden linear with no Reads
            sniff = @(varargin) tc.door("spy", 'Resources', struct('Seen', 0), 'Rules', [ ...
                OceanAgents.World.RuleSpec('Name', "Sniff", 'Phase', "act", 'Kind', "predictor", 'Predicts', "Hid", ...
                'DueEvent', "Changed", 'Fn', @(c) struct(), varargin{:}), ...
                tc.rule('Name', "Note", 'Phase', "resolve", 'Consumes', "Changed", 'Reads', "Seen", 'Writes', "Seen")]);
            tc.verifyFails(@() tc.validate([vault sniff()]), id, "V12", "Sniff", "predicts", "Hid", "vault", ...
                "list it in Reads");
            tc.validate([vault sniff('Reads', "Hid", 'Reveals', "Hid")]);
        end

        function v13AxiomsAndMembershipSeeNoForeignHiddenR3(tc)
            % Critic probe R3: door "a" gates membership [X H] with axiom A-1 (H <= 5) over door "b"'s HIDDEN H, then
            % shows which entities got X - the admission verdicts would reveal H. Refused at construction.
            T = @(varargin) tc.G.type(varargin{:});
            A = @(varargin) OceanAgents.World.Guard.AxiomSpec(varargin{:});
            id = 'OceanAgents:Guard:HiddenLeak';
            b = tc.door("b", 'Owns', struct('Name', {"H", "P"}, 'Type', {T('Visibility', "hidden"), T()}), 'Drive', "never");
            show = OceanAgents.World.RuleSpec('Name', "Show", 'Phase', "project", 'Kind', "projection", ...
                'Requires', "X", 'Fn', @(c) struct());
            tag = tc.rule('Name', "Tag", 'Commands', "attach");
            a = tc.door("a", 'Owns', struct('Name', "X", 'Type', T()), 'Membership', ["X" "H"], ...
                'Axioms', A('ID', "A-1", 'Reads', "H", 'Predicate', @(v) v.H <= 5), 'Rules', [tag show]);
            tc.verifyFails(@() tc.validate([b a]), id, "V13", "A-1", "a", "H", "b");
            % membership alone (no axiom) is refused too
            a2 = tc.door("a", 'Owns', struct('Name', "X", 'Type', T()), 'Membership', ["X" "H"], 'Rules', [tag show]);
            tc.verifyFails(@() tc.validate([b a2]), id, "V13", "Membership", "a", "H", "b");
            % a range axiom naming the hidden component
            a3 = tc.door("a", 'Owns', struct('Name', "X", 'Type', T()), 'Membership', ["X" "H"], ...
                'Axioms', A('ID', "A-2", 'Component', "H", 'Range', [0 5]));
            tc.verifyFails(@() tc.validate([b a3]), id, "A-2", "H");
            % own-door hidden components in axioms and membership stay allowed
            own = tc.door("b", 'Owns', b.Owns, 'Membership', ["H" "P"], ...
                'Axioms', A('ID', "B-1", 'Component', "H", 'Range', [0 10]));
            tc.validate(own);
            % another door's PUBLIC component is fine
            a4 = tc.door("a", 'Owns', struct('Name', "X", 'Type', T()), 'Membership', ["X" "P"], ...
                'Axioms', A('ID', "A-3", 'Component', "P", 'Range', [0 5]));
            tc.validate([b a4]);
        end

        function v12RevealsHiddenReads(tc)
            % F8: a non-projection rule reading ANOTHER door's hidden component must declare it in Reveals.
            T = @(varargin) tc.G.type(varargin{:});
            vault = tc.door("vault", 'Owns', struct('Name', {"Gold", "Tag"}, ...
                'Type', {T('Visibility', "hidden"), T()}), ...
                'Rules', tc.rule('Name', "Count", 'Reads', "Gold", 'Writes', "Tag"));   % own hidden: no Reveals needed
            tc.validate(vault);
            engine = tc.door("engine", 'Rules', tc.rule('Name', "Use", 'Reads', ["Tag" "Gold"]));
            tc.verifyFails(@() tc.validate([vault engine]), 'OceanAgents:Guard:HiddenLeak', "V12", "engine", "Use", ...
                "Gold", "vault");
            declared = tc.door("engine", 'Rules', tc.rule('Name', "Use", 'Reads', ["Tag" "Gold"], 'Reveals', "Gold"));
            tc.validate([vault declared]);
            % predictors are rules too (only projections are exempt, and they may not read hidden data at all)
            clockDoor = tc.door("secretClock", 'Owns', struct('Name', "Hid", 'Type', T('Kind', "linear", ...
                'Visibility', "hidden")));
            peek = tc.door("peeker", 'Rules', [OceanAgents.World.RuleSpec('Name', "When", 'Phase', "settle", ...
                'Kind', "predictor", 'Reads', "Hid", 'Predicts', "Hid", 'DueEvent', "Due", 'Fn', @(c) struct()), ...
                tc.rule('Name', "OnDue", 'Consumes', "Due")]);
            tc.verifyFails(@() tc.validate([clockDoor peek]), 'OceanAgents:Guard:HiddenLeak', "V12", "When", "Hid");
            % Reveals must be a subset of Reads (RuleSpec)
            tc.verifyError(@() tc.rule('Name', "X", 'Reads', "Tag", 'Reveals', "Gold"), 'OceanAgents:World:BadRule');
        end

        function v7Predictors(tc)
            bad = tc.door("oracle", 'Owns', struct('Name', "Level", 'Type', tc.G.type()), ...
                'Rules', [OceanAgents.World.RuleSpec('Name', "Guess", 'Phase', "settle", 'Kind', "predictor", ...
                'Predicts', "Level", 'DueEvent', "Due", 'Fn', @(c) struct()), ...
                tc.rule('Name', "OnDue", 'Consumes', "Due")]);
            tc.verifyFails(@() tc.validate(bad), 'OceanAgents:World:BadRule', "oracle", "Guess", "Level");
        end

        function v8DimensionsAtConstruction(tc)
            tc.verifyFails(@() tc.validate(tc.G.mover({"Position", "Velocity"})), 'OceanAgents:Guard:DimensionMismatch', ...
                "mover", "Move", "Velocity");
        end

        function v9Axioms(tc)
            id = 'OceanAgents:Guard:BadAxiom';
            A = @(varargin) OceanAgents.World.Guard.AxiomSpec(varargin{:});
            dupe = tc.G.road(A('ID', "HB-1", 'Component', "Position", 'Range', [0 5000]));
            tc.verifyFails(@() tc.validate([tc.G.human(), dupe]), id, "HB-1", "road");
            unknown = tc.G.road(A('ID', "RD-9", 'Reads', "Altitude", 'Predicate', @(v) v.Altitude > 0));
            tc.verifyFails(@() tc.validate(unknown), id, "RD-9", "road", "Altitude");
            unknownRange = tc.G.road(A('ID', "RD-8", 'Component', "Altitude", 'Range', [0 1]));
            tc.verifyFails(@() tc.validate(unknownRange), id, "RD-8", "Altitude");
        end

        function v10ContradictoryRangesK19(tc)
            id = 'OceanAgents:Guard:ContradictoryAxioms';
            A = @(varargin) OceanAgents.World.Guard.AxiomSpec(varargin{:});
            % a race door demanding Speed >= 20 m/s contradicts HB-1 (Speed <= 12 m/s)
            race = tc.door("race", 'Membership', "Speed", 'Axioms', A('ID', "RC-1", 'Component', "Speed", 'Range', [20 Inf]));
            tc.verifyFails(@() tc.validate([tc.G.human(), race]), id, "HB-1", "RC-1", "human", "race", "Speed");
            % overlapping and touching (closed) intervals can both hold
            tc.validate([tc.G.human(), tc.door("race", 'Membership', "Speed", ...
                'Axioms', A('ID', "RC-1", 'Component', "Speed", 'Range', [10 Inf]))]);
            tc.validate([tc.G.human(), tc.door("race", 'Membership', "Speed", ...
                'Axioms', A('ID', "RC-1", 'Component', "Speed", 'Range', [12 30]))]);
            % inside one door as well
            tc.verifyFails(@() tc.validate(tc.G.human(true, A('ID', "HB-9", 'Component', "Speed", 'Range', [-5 -1]))), ...
                id, "HB-1", "HB-9");
            % different components never contradict
            tc.validate(tc.G.human(true, A('ID', "HB-9", 'Component', "Mass", 'Range', [40 60])));
        end

        function v11AxiomReadsWithinMembership(tc)
            id = 'OceanAgents:Guard:BadAxiom';
            A = @(varargin) OceanAgents.World.Guard.AxiomSpec(varargin{:});
            doors = [tc.G.human(), tc.G.road()];
            reachOut = tc.G.human(true, A('ID', "HB-5", 'Mode', "monitor", 'Reads', ["Speed" "Position"], ...
                'Predicate', @(v) v.Speed < 10 | v.Position(:, 1) > 0));
            tc.verifyFails(@() tc.validate([reachOut, doors(2)]), id, "V11", "HB-5", "human", "Position");
            rangeOut = tc.G.road(A('ID', "RD-2", 'Component', "Speed", 'Range', [0 50]));
            tc.verifyFails(@() tc.validate([doors(1), rangeOut]), id, "RD-2", "road", "Speed");
            memberless = tc.door("rules", 'Axioms', A('ID', "RL-1", 'Component', "Mass", 'Range', [0 500]));
            tc.verifyFails(@() tc.validate([doors memberless]), id, "RL-1", "rules", "Mass");
            % module axioms are not entity axioms: no membership needed
            tc.validate([doors, tc.G.modules()]);
        end

        function firstFailureInCheckOrder(tc)
            % V3 (bad write) and V10 (contradiction) both present: the earlier check is reported.
            A = @(varargin) OceanAgents.World.Guard.AxiomSpec(varargin{:});
            steal = tc.door("thief", 'Rules', tc.rule('Name', "Steal", 'Writes', "Mass"), ...
                'Axioms', A('ID', "TH-1", 'Component', "Speed", 'Range', [50 60]));
            tc.verifyFails(@() tc.validate([tc.G.human(), steal]), 'OceanAgents:World:BadRule', "Steal");
        end

        function pillars(tc)
            g = OceanAgents.World.Guard.GuardEngine();
            meta = g.getMetadata();
            tc.verifyFalse(meta.IsSymbolic);
            ref = OceanStats.RL.Env.Classic.RandomWalkEnv();
            tc.verifyEqual(sort(fieldnames(meta)), sort(fieldnames(ref.getMetadata())));
            tc.verifyEqual(sort(fieldnames(g.getPipelineSpec())), sort(fieldnames(ref.getPipelineSpec())));
        end
    end
end
