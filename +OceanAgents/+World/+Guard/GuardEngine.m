classdef GuardEngine < handle
    % GUARDENGINE  The world's guard (design U8 / section 9A): it checks a whole assembly at construction, admits or
    % refuses each newcomer entity, monitors invariants at synchronisation points, and certifies modules by probing.
    % Every verdict is computed from TYPES (class, width, dimension, kind), VALUES and AXIOMS - never from the text of a
    % component, resource or door name (names are addresses, not evidence; tested by consistent renaming, K14).
    %
    %   G = OceanAgents.World.Guard.GuardEngine;
    %   G.validateAssembly(flatDoors, phases, injectable)                  % throws the first V1..V13 failure (section 8)
    %   [ok, reasons] = G.admitEntity(offer, flatDoors, store, now)        % never throws for a refusal (section 9.2)
    %   [ok, reasons] = G.admitEntity(offer, flatDoors, store, now, 'Row', r, 'Proposed', overlay)   % attach to row r
    %   viol = G.checkMonitors(flatDoors, store, proposed, now)            % monitor violations on post-write values
    %   idx  = G.buildIndex(flatDoors)             % precomputed owners, types and door gates (a plain value struct);
    %   G.admitEntity(..., 'Index', idx), G.checkMonitors(..., 'Index', idx)   skip the rebuild (identical verdicts)
    %   cert = G.admitModule(module, door, 'ModuleVersion', "1", 'Seed', 0)        % Guard.Certificate (section 9.3)
    %
    % Layer 1 (types with dimensions): an offered component must be owned by some door, its offered ComponentType must
    %   MATCH the owner's declared type (Class, Width, Dimension, Kind; Kennedy, ESOP 1994), and its value must be
    %   well formed (class, shape, finiteness). For an attach ('Row'), a component the entity already has - in the
    %   store or through the overlay - is refused with Reason "already-has": commands never change values.
    % Refusal reasons: struct array Reason ("no-owner" | "type-mismatch" | "bad-value" | "already-has" | "axiom"),
    %   Component (the offered component for layer-1 reasons, "" for axioms), AxiomID, Door, Detail. They are returned
    %   sorted by (Reason, Component, AxiomID, Door), so the order never depends on the offer's field order.
    %
    % OVERLAY ('Proposed' of admitEntity; 'proposed' of checkMonitors): the state an admission must see instead of
    %   the store, for rows changed earlier in the same synchronisation point. One field per component name:
    %     overlay.(name) = struct('Rows', n x 1 store rows, 'Values', n x W [, 'Has', n x 1 logical])
    %   Values: value kind -> the new data; linear -> the value NOW (a write rebases at now; alternatively give
    %   'Base', 'Rate', 'T0' instead of 'Values'); clock -> T0 (clock writes set T0 = Values; or give 'T0').
    %   Has (optional, admitEntity only): per listed row, whether the entity has the component after the earlier
    %   commands of this synchronisation point (true = attached earlier - its Values are the attached values;
    %   false = detached earlier). Rows without a Has entry use the store's Has. checkMonitors ignores Has (the World
    %   calls it on the post-command state).
    % Layer 2 (axioms): every door whose non-empty Membership is a subset of the entity's components (after the
    %   command) applies all its entity-scope axioms (static and monitor) to the entity's values (Meyer 1992, design by
    %   contract; Liskov & Wing 1994, behavioural subtyping). V11 guarantees every component an entity axiom reads is
    %   in its door's Membership, so every member has it; meeting a member without it is an internal fault
    %   (OceanAgents:Guard:BadAxiom), never a silent pass.
    % Layer 3 (module probing): property-based probes (Claessen & Hughes, ICFP 2000) on per-probe streams
    %   RandStream.create('mrg32k3a', 'Seed', s, 'NumStreams', Probes, 'StreamIndices', k). Passing probes is evidence,
    %   not proof; assumed axioms are never run and never reported as verified.
    % Hidden components (V6, V12): no projection reads any hidden component (not even its own door's, F7), and a
    %   projection's Requires names only PUBLIC components (its Rows/IDs would reveal who has a hidden or self one,
    %   N3). A non-projection rule that reads, requires or (predictor) predicts ANOTHER door's hidden component must
    %   declare it in RuleSpec.Reveals (N3, N4); Reveals must be a subset of Reads. V13: an entity axiom and a door's
    %   Membership may not name ANOTHER door's hidden component - admission verdicts (which entities hold a component)
    %   would encode the hidden value; own-door hidden components stay allowed.
    % Dimension checks (V8): each DimChecks term is a product of factors "Name" or "Name^p" (integer p), Name a
    %   component or the reserved factor "dt" (dimension T); every term's dimension must equal the Target's.
    % The guard never touches the global random stream. Contract: W1_Kernel_Contract.md sections 8-9.

    methods
        function meta = getMetadata(~)
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {'admission by types-with-dimensions and axioms, never by names (U8)', ...
                'probing gives evidence, not proof; invariants stay monitored after admission'};
            meta.IsSymbolic = false;
            meta.Reference = "world guard (W1 contract sections 8-9; design section 9A)";
        end

        function spec = getPipelineSpec(~)
            spec = struct();
            spec.ComponentName = "GuardEngine";
            spec.Inputs = struct('Name', {"doors", "offer", "module"}, 'Domain', {"World-DoorSpec", "World-Offer", ...
                "World-Module"}, 'Dtype', {"OceanAgents.World.DoorSpec", "struct", "object"}, 'Shape', {"1xD", "1x1", "1x1"});
            spec.Outputs = struct('Name', {"verdict", "certificate"}, 'Domain', {"World-Admission", "World-Certificate"}, ...
                'Dtype', {"struct", "OceanAgents.World.Guard.Certificate"}, 'Shape', {"1xK", "1x1"});
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "guardengine");
            spec.ComposesFrom = {};
        end
    end

    methods (Static)
        function validateAssembly(flatDoors, phases, injectable)
            % VALIDATEASSEMBLY  Checks V1..V13 in order; throws the first failure with the section 11 error ID.
            arguments
                flatDoors
                phases
                injectable = strings(1, 0)
            end
            doors = checkDoors_(flatDoors);
            phases = textRow_(phases);
            injectable = textRow_(injectable);
            v1_(doors, phases);
            W = OceanAgents.World.Guard.GuardEngine.buildIndex(doors);   % raises V2 on duplicate ownership
            v2_(doors, W);
            v3_(doors, phases, W);
            [emitted, consumed] = eventSets_(doors);
            v4_(doors, emitted, consumed);
            v5_(doors, emitted, injectable);
            v6_(doors, W);
            v7_(doors, W);
            v8_(doors, W);
            v9_(doors, W);
            v10_(doors);
            v11_(doors);
            v12_(doors, W);
            v13_(doors, W);
        end

        function [ok, reasons] = admitEntity(offer, flatDoors, store, now, options)
            % ADMITENTITY  Layer 1 and layer 2 admission of one spawn/attach offer. Never throws for a refusal.
            %   'Row'      store row of the target entity for an attach (its current components join the check)
            %   'Proposed' overlay of post-write values (same form as checkMonitors) used for that row
            arguments
                offer (1,1) struct
                flatDoors
                store
                now (1,1) double
                options.Row double = []
                options.Proposed struct = struct()
                options.Index = []
            end
            W = indexFor_(flatDoors, options.Index);
            reasons = emptyReasons_();
            values = struct();
            names = string(fieldnames(offer))';
            for nm = names
                o = offer.(char(nm));
                if ~isComponent_(W, nm)
                    reasons(end + 1) = reason_("no-owner", nm, "", "", sprintf('no door owns a component "%s"', nm)); %#ok<AGROW>
                    continue
                end
                owner = W.Owner.(char(nm));
                declared = W.Type.(char(nm));
                if ~(isstruct(o) && isfield(o, 'Type') && isa(o.Type, 'OceanAgents.World.Guard.ComponentType'))
                    reasons(end + 1) = reason_("bad-value", nm, "", owner, sprintf('offer for "%s" has no ComponentType', nm)); %#ok<AGROW>
                    continue
                end
                if ~o.Type.matches(declared)
                    what = o.Type.mismatch(declared);
                    reasons(end + 1) = reason_("type-mismatch", nm, "", owner, sprintf('"%s": %s differs: offered %s, declared %s', ...
                        nm, what, attr_(o.Type, what), attr_(declared, what))); %#ok<AGROW>
                    continue
                end
                [v, why] = offeredValue_(o.Value, declared, now);
                if strlength(why) > 0
                    reasons(end + 1) = reason_("bad-value", nm, "", owner, sprintf('"%s": %s', nm, why)); %#ok<AGROW>
                    continue
                end
                if ~isempty(options.Row) && overlayHas_(store, options.Proposed, nm, options.Row)
                    reasons(end + 1) = reason_("already-has", nm, "", owner, sprintf( ...
                        'the entity already has "%s"; commands never change values (write it through its owner)', nm)); %#ok<AGROW>
                    continue
                end
                values.(char(nm)) = v;
            end
            if ~isempty(reasons)
                ok = false;
                reasons = sortReasons_(reasons);
                return
            end
            % the entity's components after the command (attach: current components of its row, post-write values)
            if ~isempty(options.Row)
                row = options.Row;
                for nm = W.Components
                    if ~isfield(values, char(nm)) && overlayHas_(store, options.Proposed, nm, row)
                        values.(char(nm)) = overlayValues_(store, options.Proposed, nm, W.Type.(char(nm)), row, now);
                    end
                end
            end
            for g = W.Gates
                if ~all(isfield(values, g.Membership)), continue; end
                for a = g.Entity
                    [pass, detail] = evalEntityAxiom_(a, values, 1, g.Door);
                    if ~pass
                        reasons(end + 1) = reason_("axiom", "", a.ID, g.Door, detail); %#ok<AGROW>
                    end
                end
            end
            reasons = sortReasons_(reasons);
            ok = isempty(reasons);
        end

        function viol = checkMonitors(flatDoors, store, proposed, now, options)
            % CHECKMONITORS  Entity monitor axioms of every door on its member rows, seeing post-write values.
            %   proposed: struct name -> struct(Rows, Values) or, for linear, (Rows, Base, Rate, T0); clock (Rows, T0)
            %   viol: struct array AxiomID, Door, EntityRows (store rows), Detail. The World raises the error.
            arguments
                flatDoors
                store
                proposed struct = struct()
                now (1,1) double = 0
                options.Index = []
            end
            W = indexFor_(flatDoors, options.Index);
            viol = struct('AxiomID', {}, 'Door', {}, 'EntityRows', {}, 'Detail', {});
            for g = W.Gates
                if isempty(g.Monitors), continue; end
                members = reshape(store.rowsWith(string(g.Membership)), [], 1);
                if isempty(members), continue; end
                d = struct('Name', g.Door);
                for a = g.Monitors
                    reads = a.readsAll();
                    rows = members;
                    for nm = reads                                    % V11: members always have the reads
                        if ~all(store.has(nm, rows))
                            missingRead_(a, d.Name, nm);
                        end
                    end
                    values = struct();
                    for nm = reads
                        values.(char(nm)) = overlayValues_(store, proposed, nm, W.Type.(char(nm)), rows, now);
                    end
                    [pass, detail] = evalEntityAxiom_(a, values, numel(rows), d.Name);
                    if ~all(pass)
                        viol(end + 1) = struct('AxiomID', a.ID, 'Door', d.Name, 'EntityRows', rows(~pass)', ...
                            'Detail', detail); %#ok<AGROW>
                    end
                end
            end
        end

        function cert = admitModule(module, door, options)
            % ADMITMODULE  Runs every module-scope axiom of door on module and returns a Guard.Certificate.
            % Coverage: a probe predicate may return a third output, a struct of named conditions -> logical (did THIS
            % probe exercise that condition). A probed axiom one of whose conditions no probe reached is reported as
            % "partial", never "probed" (a check that never ran is not a pass). 'RequireCoverage', true turns a
            % partial axiom into a failed one.
            arguments
                module
                door (1,1) OceanAgents.World.DoorSpec
                options.ModuleVersion = "1"
                options.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
                options.RequireCoverage (1,1) logical = false
            end
            if isempty(door.Admits) || ~any(arrayfun(@(c) isa(module, char(c)), door.Admits))
                error('OceanAgents:Guard:NotAdmissible', ...
                    'Door "%s" admits instances of [%s]; the offered module (%s) is none of them.', ...
                    door.Name, strjoin(door.Admits, ', '), class(module));
            end
            ax = struct('ID', {}, 'Mode', {}, 'Status', {}, 'Probes', {}, 'Seeds', {}, 'Detail', {}, 'Coverage', {});
            for a = door.Axioms
                if a.Scope ~= "module", continue; end
                entry = struct('ID', a.ID, 'Mode', a.Mode, 'Status', "failed", 'Probes', 0, 'Seeds', zeros(1, 0), ...
                    'Detail', "", 'Coverage', struct());
                switch a.Mode
                    case "assumed"
                        entry.Status = "assumed";
                        entry.Detail = "assumed: never checked";
                    case "static"
                        [pass, detail] = callModulePredicate_(a.Predicate, module, []);
                        entry.Probes = 1;
                        if pass, entry.Status = "verified-static"; end
                        entry.Detail = detail;
                    case "probe"
                        entry.Status = "probed";
                        for k = 1:a.Probes
                            makeStream = @() RandStream.create('mrg32k3a', 'Seed', options.Seed, 'NumStreams', a.Probes, ...
                                'StreamIndices', k);
                            [pass, detail, reached] = callModulePredicate_(a.Predicate, module, makeStream);
                            entry.Probes = k;
                            entry.Seeds(end + 1) = k;
                            for f = reshape(string(fieldnames(reached)), 1, [])
                                if ~isfield(entry.Coverage, f), entry.Coverage.(f) = 0; end
                                entry.Coverage.(f) = entry.Coverage.(f) + double(any(logical(reached.(f))));
                            end
                            if ~pass
                                entry.Status = "failed";
                                entry.Detail = sprintf("probe %d (seed %d, stream %d): %s", k, options.Seed, k, detail);
                                break
                            end
                        end
                        if entry.Status == "probed"
                            entry.Detail = sprintf("%d probes passed (mrg32k3a seed %d, StreamIndices 1..%d)", ...
                                a.Probes, options.Seed, a.Probes);
                            names = reshape(string(fieldnames(entry.Coverage)), 1, []);
                            missed = names(arrayfun(@(f) entry.Coverage.(f) == 0, names));
                            if ~isempty(missed)
                                entry.Status = "partial";
                                if options.RequireCoverage, entry.Status = "failed"; end
                                entry.Detail = sprintf("%s; never reached in any probe: %s", entry.Detail, ...
                                    strjoin(missed, ", "));
                            end
                        end
                end
                ax(end + 1) = entry; %#ok<AGROW>
            end
            cert = OceanAgents.World.Guard.Certificate('Subject', string(class(module)), ...
                'ModuleVersion', string(options.ModuleVersion), 'Door', door.Name, 'DoorVersion', door.Version, ...
                'Axioms', ax);
        end

        function idx = buildIndex(flatDoors)
            % BUILDINDEX  Everything admitEntity/checkMonitors precompute from the doors, as a plain value struct:
            %   Doors (DoorSpec array), Owner / Type / IsResource (structs keyed by component or resource name),
            %   Components (component names in rank order), Gates (struct array, one per door with a non-empty
            %   Membership: Door, Membership (cellstr), Entity = its entity-scope axioms, Monitors = the monitor ones).
            % Raises OceanAgents:World:DuplicateOwner when a name is owned twice. Build once per assembly (the World
            % does this at construction) and pass it with 'Index' to skip the rebuild on every call.
            doors = checkDoors_(flatDoors);
            idx = struct('Doors', doors, 'Owner', struct(), 'Type', struct(), 'IsResource', struct(), ...
                'Components', strings(1, 0), 'Gates', struct('Door', {}, 'Membership', {}, 'Entity', {}, 'Monitors', {}));
            for d = doors
                for nm = d.ownedNames()
                    key = char(nm);
                    if isfield(idx.Owner, key)
                        error('OceanAgents:World:DuplicateOwner', 'V2: "%s" is owned by door "%s" and by door "%s".', ...
                            nm, idx.Owner.(key), d.Name);
                    end
                    idx.Owner.(key) = d.Name;
                    idx.IsResource.(key) = true;
                end
                for c = d.Owns
                    idx.Type.(char(c.Name)) = c.Type;
                    idx.IsResource.(char(c.Name)) = false;
                    idx.Components(end + 1) = c.Name;
                end
            end
            for d = doors
                if isempty(d.Membership), continue; end
                ent = OceanAgents.World.Guard.AxiomSpec.empty(1, 0);
                for a = d.Axioms
                    if a.Scope == "entity", ent(end + 1) = a; end %#ok<AGROW>
                end
                mon = OceanAgents.World.Guard.AxiomSpec.empty(1, 0);
                for a = ent
                    if a.Mode == "monitor", mon(end + 1) = a; end %#ok<AGROW>
                end
                idx.Gates(end + 1) = struct('Door', d.Name, 'Membership', {cellstr(d.Membership)}, ...
                    'Entity', ent, 'Monitors', mon);
            end
        end

        function [flat, depth] = flattenDoors(doors)
            % FLATTENDOORS  Depth-first flattening of top-level doors (parent before children); position = door rank.
            flat = OceanAgents.World.DoorSpec.empty(1, 0);
            depth = zeros(1, 0);
            for d = reshape(doors, 1, [])
                [f, k] = d.flatten();
                flat = [flat, f]; %#ok<AGROW>
                depth = [depth, k]; %#ok<AGROW>
            end
        end
    end
end

% =====================================================================================================================
% construction validation (section 8)

function doors = checkDoors_(flatDoors)
if isempty(flatDoors), doors = OceanAgents.World.DoorSpec.empty(1, 0); return; end
if ~isa(flatDoors, 'OceanAgents.World.DoorSpec')
    error('OceanAgents:World:BadAssembly', 'The assembly must be a DoorSpec array.');
end
doors = reshape(flatDoors, 1, []);
end

function r = textRow_(x)
if isempty(x), r = strings(1, 0); return; end
r = reshape(string(x), 1, []);
end

function W = indexFor_(flatDoors, idx)
% The caller's prebuilt index, or a fresh one.
if isempty(idx)
    W = OceanAgents.World.Guard.GuardEngine.buildIndex(flatDoors);
elseif isstruct(idx) && isscalar(idx) && all(isfield(idx, {'Doors', 'Owner', 'Type', 'IsResource', 'Components', 'Gates'}))
    W = idx;
else
    error('OceanAgents:World:BadAssembly', 'Index must come from GuardEngine.buildIndex.');
end
end

function tf = isComponent_(W, nm)
key = char(nm);
tf = isfield(W.IsResource, key) && ~W.IsResource.(key);
end

function tf = exists_(W, nm)
tf = isfield(W.Owner, char(nm));
end

function v1_(doors, phases)
names = [doors.Name];
if isempty(phases) || numel(unique(phases)) ~= numel(phases) || any(strlength(phases) == 0)
    error('OceanAgents:World:BadAssembly', 'V1: the phase list must be non-empty and unique (got [%s]).', strjoin(phases, ', '));
end
[u, ~, j] = unique(names);
if numel(u) ~= numel(names)
    dup = u(accumarray(j(:), 1) > 1);
    error('OceanAgents:World:BadAssembly', 'V1: door name(s) used more than once: %s.', strjoin(dup, ', '));
end
for d = doors
    if subtreeDepth_(d) > 2
        error('OceanAgents:World:BadAssembly', 'V1: door "%s" nests sub-doors deeper than 2 levels.', d.Name);
    end
end
end

function k = subtreeDepth_(d)
k = 1;
for c = d.Children
    k = max(k, 1 + subtreeDepth_(c));
end
end

function v2_(doors, W)
% Duplicate ownership across doors is caught while indexing; here: duplicates inside one door, reserved names,
% and component/resource name collisions inside one door.
for d = doors
    names = d.ownedNames();
    if numel(unique(names)) ~= numel(names)
        error('OceanAgents:World:DuplicateOwner', 'V2: door "%s" declares a component or resource name twice.', d.Name);
    end
    if any(names == "dt")
        error('OceanAgents:World:DuplicateOwner', ...
            'V2: door "%s" declares "dt", which is reserved as the time factor of dimension checks.', d.Name);
    end
end
for d = doors                                                      % membership addresses must be components
    for nm = d.Membership
        if ~isComponent_(W, nm)
            error('OceanAgents:World:BadAssembly', 'V2: door "%s" Membership names "%s", which is no component.', ...
                d.Name, nm);
        end
    end
end
end

function v3_(doors, phases, W)
for d = doors
    rnames = [d.Rules.Name];
    if numel(unique(rnames)) ~= numel(rnames)
        error('OceanAgents:World:BadRule', 'V3: door "%s" has two rules with the same name.', d.Name);
    end
    owned = d.ownedNames();
    for r = d.Rules
        where = sprintf('door "%s" rule "%s"', d.Name, r.Name);
        if ~any(r.Phase == phases)
            error('OceanAgents:World:BadRule', 'V3: %s runs in unknown phase "%s".', where, r.Phase);
        end
        bad = r.Writes(~ismember(r.Writes, owned));
        if ~isempty(bad)
            error('OceanAgents:World:BadRule', 'V3: %s writes [%s], not owned by its door.', where, strjoin(bad, ', '));
        end
        missing = r.Reads(~arrayfun(@(n) exists_(W, n), r.Reads));
        if ~isempty(missing)
            error('OceanAgents:World:BadRule', 'V3: %s reads unknown [%s].', where, strjoin(missing, ', '));
        end
        missing = r.Requires(~arrayfun(@(n) isComponent_(W, n), r.Requires));
        if ~isempty(missing)
            error('OceanAgents:World:BadRule', 'V3: %s requires unknown component(s) [%s].', where, strjoin(missing, ', '));
        end
        if r.Kind ~= "rule" && (~isempty(r.Writes) || ~isempty(r.Emits) || ~isempty(r.Commands))
            error('OceanAgents:World:BadRule', 'V3: %s is a %s and may not write, emit or command.', where, r.Kind);
        end
        if (r.Kind == "predictor") ~= (strlength(r.Predicts) > 0 && strlength(r.DueEvent) > 0)
            error('OceanAgents:World:BadRule', 'V3: %s: only predictors set Predicts and DueEvent (both).', where);
        end
    end
end
end

function [emitted, consumed] = eventSets_(doors)
emitted = strings(1, 0);
consumed = strings(1, 0);
for d = doors
    for r = d.Rules
        emitted = [emitted, r.Emits]; %#ok<AGROW>
        if r.Kind == "predictor", emitted(end + 1) = r.DueEvent; end %#ok<AGROW>
        consumed = [consumed, r.Consumes]; %#ok<AGROW>
    end
end
emitted = unique(emitted);
consumed = unique(consumed);
end

function v4_(doors, ~, consumed)
for d = doors
    for r = d.Rules
        out = r.Emits;
        if r.Kind == "predictor", out(end + 1) = r.DueEvent; end %#ok<AGROW>
        orphan = out(~ismember(out, consumed));
        if ~isempty(orphan)
            error('OceanAgents:World:UnconsumedEvent', 'V4: door "%s" rule "%s" emits [%s], which no rule consumes.', ...
                d.Name, r.Name, strjoin(orphan, ', '));
        end
    end
end
end

function v5_(doors, emitted, injectable)
for d = doors
    for r = d.Rules
        unknown = r.Consumes(~ismember(r.Consumes, [emitted injectable]));
        if ~isempty(unknown)
            error('OceanAgents:World:UnknownEvent', ...
                'V5: door "%s" rule "%s" consumes [%s], which no rule emits and which is not injectable.', ...
                d.Name, r.Name, strjoin(unknown, ', '));
        end
    end
end
end

function v6_(doors, W)
% Projections face agents: they read no hidden component (own door included, F7), and they may SELECT rows only by
% public components - Requires on a hidden or self component would reveal, through ctx.Rows/IDs, which entities
% hold it (N3).
for d = doors
    for r = d.Rules
        if r.Kind ~= "projection", continue; end
        for nm = r.Reads
            if isComponent_(W, nm) && W.Type.(char(nm)).Visibility == "hidden"
                error('OceanAgents:Guard:HiddenLeak', ...
                    'V6: projection "%s" of door "%s" reads hidden component "%s" of door "%s".', ...
                    r.Name, d.Name, nm, W.Owner.(char(nm)));
            end
        end
        for nm = r.Requires
            if isComponent_(W, nm) && W.Type.(char(nm)).Visibility ~= "public"
                error('OceanAgents:Guard:HiddenLeak', ...
                    ['V6: projection "%s" of door "%s" requires %s component "%s" of door "%s"; a projection may ' ...
                    'require only public components (its rows would reveal which entities have it).'], ...
                    r.Name, d.Name, W.Type.(char(nm)).Visibility, nm, W.Owner.(char(nm)));
            end
        end
    end
end
end

function v7_(doors, W)
for d = doors
    for r = d.Rules
        if r.Kind ~= "predictor", continue; end
        p = r.Predicts;
        if ~isComponent_(W, p) || ~any(W.Type.(char(p)).Kind == ["linear" "clock"])
            error('OceanAgents:World:BadRule', ...
                'V7: predictor "%s" of door "%s" predicts "%s", which is not a linear or clock component.', ...
                r.Name, d.Name, p);
        end
    end
end
end

function v8_(doors, W)
for d = doors
    for r = d.Rules
        for chk = r.DimChecks
            where = sprintf('door "%s" rule "%s"', d.Name, r.Name);
            target = string(chk.Target);
            tdim = factorDimension_(target, 1, W, where);
            for j = 1:numel(chk.Terms)
                term = reshape(string(chk.Terms{j}), 1, []);
                dim = OceanAgents.World.Guard.Dimension();
                for tok = term
                    [nm, p] = parseFactor_(tok, where);
                    dim = dim .* factorDimension_(nm, p, W, where);
                end
                if dim ~= tdim
                    error('OceanAgents:Guard:DimensionMismatch', ...
                        'V8: %s: term %d [%s] has dimension %s but target "%s" has dimension %s.', ...
                        where, j, strjoin(term, ' * '), char(dim), target, char(tdim));
                end
            end
        end
    end
end
end

function [nm, p] = parseFactor_(tok, where)
% "Name" or "Name^p" with an integer p (possibly negative).
parts = split(string(tok), "^");
ok = numel(parts) <= 2 && ~isempty(regexp(parts(1), '^[A-Za-z]\w*$', 'once'));
p = 1;
if ok && numel(parts) == 2
    ok = ~isempty(regexp(parts(2), '^-?\d+$', 'once'));
    if ok, p = str2double(parts(2)); end
end
if ~ok
    error('OceanAgents:World:BadRule', 'V8: %s: dimension factor "%s" is not "Name" or "Name^p".', where, tok);
end
nm = parts(1);
end

function dim = factorDimension_(nm, p, W, where)
if nm == "dt"
    dim = OceanAgents.World.Guard.Dimension.T() .^ p;
elseif isComponent_(W, nm)
    dim = W.Type.(char(nm)).Dimension .^ p;
else
    error('OceanAgents:World:BadRule', 'V8: %s: dimension factor "%s" is neither a component nor dt.', where, nm);
end
end

function v9_(doors, W)
ids = strings(1, 0);
for d = doors
    for a = d.Axioms
        where = sprintf('axiom "%s" of door "%s"', a.ID, d.Name);
        if any(ids == a.ID)
            error('OceanAgents:Guard:BadAxiom', 'V9: axiom ID "%s" is used more than once (again in door "%s").', a.ID, d.Name);
        end
        ids(end + 1) = a.ID; %#ok<AGROW>
        if a.Scope == "entity"
            if ~any(a.Mode == ["static" "monitor"])
                error('OceanAgents:Guard:BadAxiom', 'V9: %s: entity axioms are static or monitor.', where);
            end
            missing = a.readsAll();
            missing = missing(~arrayfun(@(n) isComponent_(W, n), missing));
            if ~isempty(missing)
                error('OceanAgents:Guard:BadAxiom', 'V9: %s reads unknown component(s) [%s].', where, strjoin(missing, ', '));
            end
        elseif ~any(a.Mode == ["static" "probe" "assumed"])
            error('OceanAgents:Guard:BadAxiom', 'V9: %s: module axioms are static, probe or assumed.', where);
        end
    end
end
end

function v10_(doors)
% Two entity-scope range axioms on the same component with disjoint closed intervals cannot both hold.
list = struct('ID', {}, 'Door', {}, 'Component', {}, 'Range', {});
for d = doors
    for a = d.Axioms
        if a.Scope == "entity" && a.isRange()
            list(end + 1) = struct('ID', a.ID, 'Door', d.Name, 'Component', a.Component, 'Range', a.Range); %#ok<AGROW>
        end
    end
end
for i = 1:numel(list)
    for j = i + 1:numel(list)
        a = list(i);  b = list(j);
        if a.Component == b.Component && (a.Range(2) < b.Range(1) || b.Range(2) < a.Range(1))
            error('OceanAgents:Guard:ContradictoryAxioms', ...
                'V10: axiom "%s" (door "%s", %s in %s) and axiom "%s" (door "%s", %s in %s) cannot both hold.', ...
                a.ID, a.Door, a.Component, mat2str(a.Range), b.ID, b.Door, b.Component, mat2str(b.Range));
        end
    end
end
end

function v11_(doors)
% Every component an entity-scope axiom reads must be in its door's Membership, so the axiom is checked on every
% member and can never be silently skipped.
for d = doors
    for a = d.Axioms
        if a.Scope ~= "entity", continue; end
        outside = a.readsAll();
        outside = outside(~ismember(outside, d.Membership));
        if ~isempty(outside)
            error('OceanAgents:Guard:BadAxiom', ...
                'V11: axiom "%s" of door "%s" reads [%s], which is not in the door''s Membership [%s].', ...
                a.ID, d.Name, strjoin(outside, ', '), strjoin(d.Membership, ', '));
        end
    end
end
end

function v12_(doors, W)
% A non-projection rule that reads, requires or predicts ANOTHER door's hidden component must declare it in Reveals:
% information derived from hidden data reaches a view only through rules that explicitly declare the revelation
% (F8). Requires selects ctx.Rows/IDs (presence is information, N3); a predictor's Predicts gives it the component's
% dirty rows and crossing times (N4).
for d = doors
    for r = d.Rules
        if r.Kind == "projection", continue; end
        uses = [r.Reads; repmat("reads", 1, numel(r.Reads))];
        uses = [uses, [r.Requires; repmat("requires", 1, numel(r.Requires))]]; %#ok<AGROW>
        if r.Kind == "predictor"
            uses = [uses, [r.Predicts; "predicts"]]; %#ok<AGROW>
        end
        for k = 1:size(uses, 2)
            nm = uses(1, k);
            how = uses(2, k);
            if isComponent_(W, nm) && W.Owner.(char(nm)) ~= d.Name && W.Type.(char(nm)).Visibility == "hidden" ...
                    && ~any(r.Reveals == nm)
                hint = '';
                if how ~= "reads"
                    hint = ' (Reveals must be a subset of Reads, so list it in Reads as well)';
                end
                error('OceanAgents:Guard:HiddenLeak', ...
                    'V12: rule "%s" of door "%s" %s hidden component "%s" of door "%s" without declaring it in Reveals%s.', ...
                    r.Name, d.Name, how, nm, W.Owner.(char(nm)), hint);
            end
        end
    end
end
end

function v13_(doors, W)
% Entity axioms and memberships gate admission; if they could name another door's hidden component, WHICH entities
% get admitted (and hold the door's components) would reveal the hidden value (critic probe R3).
for d = doors
    for a = d.Axioms
        if a.Scope ~= "entity", continue; end
        for nm = a.readsAll()
            if hiddenOfOther_(W, nm, d.Name)
                error('OceanAgents:Guard:HiddenLeak', ...
                    'V13: axiom "%s" of door "%s" names hidden component "%s" of door "%s".', ...
                    a.ID, d.Name, nm, W.Owner.(char(nm)));
            end
        end
    end
    for nm = d.Membership
        if hiddenOfOther_(W, nm, d.Name)
            error('OceanAgents:Guard:HiddenLeak', ...
                'V13: the Membership of door "%s" names hidden component "%s" of door "%s".', ...
                d.Name, nm, W.Owner.(char(nm)));
        end
    end
end
end

function tf = hiddenOfOther_(W, nm, door)
tf = isComponent_(W, nm) && W.Owner.(char(nm)) ~= door && W.Type.(char(nm)).Visibility == "hidden";
end

function missingRead_(a, door, nm)
% Defensive: unreachable for an assembly that passed validateAssembly (V11).
error('OceanAgents:Guard:BadAxiom', ['Internal guard fault: a member of door "%s" lacks component "%s" read by ' ...
    'axiom "%s" (the assembly was not validated, or V11 was bypassed).'], door, nm, a.ID);
end

% =====================================================================================================================
% entity admission and monitoring (section 9.2, 7.1 step 5.4)

function r = emptyReasons_()
r = struct('Reason', {}, 'Component', {}, 'AxiomID', {}, 'Door', {}, 'Detail', {});
end

function r = reason_(kind, component, axiomID, door, detail)
r = struct('Reason', string(kind), 'Component', string(component), 'AxiomID', string(axiomID), ...
    'Door', string(door), 'Detail', string(detail));
end

function r = sortReasons_(r)
% Deterministic order, independent of the offer's field order (F15).
if numel(r) < 2, return; end
key = [[r.Reason]' [r.Component]' [r.AxiomID]' [r.Door]'];
[~, order] = sortrows(key);
r = r(order);
end

function h = overlayHas_(store, proposed, nm, row)
% Whether the entity in row has component nm after the earlier commands of this synchronisation point.
h = store.has(nm, row);
if isfield(proposed, char(nm)) && isfield(proposed.(char(nm)), 'Has')
    p = proposed.(char(nm));
    at = find(p.Rows(:) == row, 1);
    if ~isempty(at), h = logical(p.Has(at)); end
end
end

function s = attr_(t, what)
switch what
    case "Class", s = char(t.Class);
    case "Width", s = num2str(t.Width);
    case "Dimension", s = char(t.Dimension);
    case "Kind", s = char(t.Kind);
    otherwise, s = '';
end
end

function [v, why] = offeredValue_(value, type, now)
% Value of an offered component at now, validated against the declared type ("" why = well formed).
why = "";
v = [];
w = type.Width;
switch type.Kind
    case "value"
        if ~classOk_(value, type.Class) || ~isequal(size(value), [1 w])
            why = sprintf('expected a 1x%d %s row', w, type.Class);
        elseif type.Class == "double" && ~all(isfinite(value))
            why = "non-finite value";
        elseif type.Class == "string" && any(ismissing(value))
            why = "missing string value";
        else
            v = value;
        end
    case "linear"
        if ~(isstruct(value) && isscalar(value) && all(isfield(value, {'Value', 'Rate'})))
            why = "a linear component is offered as struct(Value, Rate)";
        elseif ~(isa(value.Value, 'double') && isequal(size(value.Value), [1 w]) && all(isfinite(value.Value)) ...
                && isa(value.Rate, 'double') && isequal(size(value.Rate), [1 w]) && all(isfinite(value.Rate)))
            why = sprintf('linear Value and Rate must be finite 1x%d doubles', w);
        else
            v = value.Value;                                          % base at now: the value now
        end
    case "clock"
        if ~(isa(value, 'double') && isscalar(value) && isfinite(value))
            why = "a clock component is offered as a finite scalar T0";
        else
            v = now - value;
        end
end
end

function tf = classOk_(value, cls)
switch cls
    case "double", tf = isa(value, 'double') && isreal(value);
    case "logical", tf = islogical(value);
    case "string", tf = isstring(value);
    otherwise, tf = false;
end
end

function v = overlayValues_(store, proposed, nm, type, rows, now)
% Current values of component nm on rows, with any proposed (post-write) values laid over them.
v = store.valueAt(nm, rows, now);
if ~isfield(proposed, char(nm)), return; end
p = proposed.(char(nm));
[hit, at] = ismember(rows(:), p.Rows(:));
if ~any(hit), return; end
switch type.Kind
    case "value"
        new = p.Values;
    case "linear"
        if isfield(p, 'Base')
            new = p.Base + p.Rate .* (now - p.T0);
        else
            new = p.Values;                                           % a write rebases: Values is the value now
        end
    case "clock"
        if isfield(p, 'T0'), t0 = p.T0; else, t0 = p.Values; end      % clock writes set T0 = Values
        new = now - t0;
end
v(hit, :) = new(at(hit), :);
end

function [pass, detail] = evalEntityAxiom_(a, values, n, door)
% pass: n x 1 logical. Every read component must be present (V11); a missing one is an internal fault.
detail = "";
reads = a.readsAll();
for nm = reads
    if ~isfield(values, char(nm)), missingRead_(a, door, nm); end
end
if a.isRange()
    x = values.(char(a.Component));
    pass = all(x >= a.Range(1) & x <= a.Range(2), 2);
    if ~all(pass)
        detail = sprintf('%s must lie in %s', a.Component, mat2str(a.Range));
    end
    pass = reshape(pass, [], 1);
    return
end
sub = struct();
for nm = reads
    sub.(char(nm)) = values.(char(nm));
end
try
    ok = a.Predicate(sub);
catch err
    pass = false(n, 1);
    detail = sprintf('predicate error: %s', err.message);
    return
end
if ~((islogical(ok) || isnumeric(ok)) && numel(ok) == n) || any(isnan(double(ok(:))))
    pass = false(n, 1);
    detail = sprintf('predicate returned a %s %s, expected a %dx1 logical', strjoin(string(size(ok)), 'x'), class(ok), n);
    return
end
pass = reshape(logical(ok), [], 1);
if ~all(pass)
    detail = string(a.Description);
    if strlength(detail) == 0, detail = "predicate is false"; end
end
end

% =====================================================================================================================
% module admission (section 9.3)

function [pass, detail, reached] = callModulePredicate_(fn, module, makeStream)
% Runs a module predicate; any error is a failed axiom with the error message (it never escapes).
% makeStream: [] for a static axiom, else a function returning the probe's fresh RandStream.
% reached: the predicate's optional third output (struct of condition -> logical), else an empty struct.
pass = false;
reached = struct();
try
    if nargout(fn) >= 3
        a = argList_(module, makeStream);
        [ok, detail, reached] = fn(a{:});
        if ~(isstruct(reached) && isscalar(reached)), reached = struct(); end
    else
        [ok, detail] = twoOutputs_(fn, module, makeStream);
    end
catch err
    detail = sprintf('predicate error: %s', err.message);
    return
end
detail = string(detail);
if isempty(detail), detail = ""; end
detail = strjoin(detail(:)', "; ");
pass = (islogical(ok) || isnumeric(ok)) && isscalar(ok) && ~isnan(double(ok)) && logical(ok);
if ~pass && strlength(detail) == 0, detail = "predicate is false"; end
end

function [ok, detail] = twoOutputs_(fn, module, makeStream)
% A predicate returns [ok, detail]; a one-output predicate (e.g. a plain anonymous expression) is accepted with an
% empty detail. The retry uses a FRESH stream, so a probe's draws are the same whichever form is used.
args = @() argList_(module, makeStream);
if nargout(fn) == 1
    a = args();
    ok = fn(a{:});
    detail = "";
    return
end
try
    a = args();
    [ok, detail] = fn(a{:});
catch err
    if ~any(strcmp(err.identifier, {'MATLAB:TooManyOutputs', 'MATLAB:maxlhs'}))
        rethrow(err);
    end
    a = args();
    ok = fn(a{:});
    detail = "";
end
end

function a = argList_(module, makeStream)
if isempty(makeStream), a = {module}; else, a = {module, makeStream()}; end
end
