classdef ActiveSet < handle
    % ACTIVESET  Registration of which kernel participants are active (builder plan, "registration, not local clocks").
    % The world clock and calendar stay the single authority: this class only DECLARES participants (owner, dependencies,
    % shared resources, binding, trigger), validates/expands a requested active set and reports why, and applies the
    % change through World.deactivate / World.activate at a commit boundary.
    %
    %   reg = OceanAgents.Time.ActiveSet();
    %   reg.declare(3, 'Owner', "mind1", 'Binding', "formula", 'Trigger', "on-read", 'DependsOn', 5, 'Resources', "charger");
    %   p = reg.plan(world, [3 4]);                       % p.Active, p.ExpandedBy(k).ID/.Reason, p.Equivalence
    %   rep = reg.apply(world, [3 4], 'Expand', true, 'Deadlines', "shift");
    %
    % world: a World, an AgentWorld or a KernelDomain (anything with liveEntityIDs/inactiveEntityIDs/deactivate/
    % activate). Undeclared participants have no dependencies. Roster agents of an AgentWorld are pinned active
    % (action-driven). Unregistered (inactive) participants receive no rule runs, predictions, RNG draws or interactions;
    % their formula values are held (see World.deactivate). A selective set is a CHANGED experiment whenever a frozen
    % participant would have influenced the outcome - the report says so; never claim equivalence to the full world.
    properties (SetAccess = private)
        Declarations struct = struct('EntityID', {}, 'Owner', {}, 'DependsOn', {}, 'Resources', {}, 'Binding', {}, 'Trigger', {})
        Revision double = 0              % revision of the world's active set after the last apply
    end
    properties (Constant)
        Bindings = ["formula" "calendar" "array" "clock"]
    end
    methods
        function declare(obj, entityID, opts)
            arguments
                obj
                entityID (1,1) double
                opts.Owner (1,1) string
                opts.DependsOn (1,:) double = zeros(1, 0)
                opts.Resources string = strings(1, 0)
                opts.Binding (1,1) string = "calendar"
                opts.Trigger (1,1) string = "every-tick"
            end
            if ~any(opts.Binding == obj.Bindings)
                error('OceanAgents:Time:BadValue', 'Binding must be one of %s.', strjoin(obj.Bindings, ', '));
            end
            if any([obj.Declarations.EntityID] == entityID)
                error('OceanAgents:Time:DuplicateOwnership', ...
                    'Entity %g is already declared (owner %s); one owner per participant.', entityID, ...
                    obj.Declarations([obj.Declarations.EntityID] == entityID).Owner);
            end
            obj.Declarations(end+1) = struct('EntityID', entityID, 'Owner', opts.Owner, 'DependsOn', opts.DependsOn, ...
                'Resources', reshape(opts.Resources, 1, []), 'Binding', opts.Binding, 'Trigger', opts.Trigger);
        end

        function p = plan(obj, world, requested)
            % PLAN  Validate and expand the requested active set; says why each participant was added. No mutation.
            arguments
                obj
                world
                requested (1,:) double
            end
            live = reshape(world.liveEntityIDs(), 1, []);
            bad = setdiff(requested, live);
            if ~isempty(bad)
                error('OceanAgents:Time:UnknownEntity', 'Requested entit%s %s not alive.', ...
                    plural_(numel(bad)), mat2str(bad));
            end
            active = unique(requested);
            reasons = struct('ID', {}, 'Reason', {});
            pinned = reshape(setdiff(world.pinnedEntityIDs(), active), 1, []);
            for id = pinned
                active(end+1) = id; %#ok<AGROW>
                reasons(end+1) = struct('ID', id, 'Reason', "action-driven roster agent (always active)"); %#ok<AGROW>
            end
            queue = active;
            while ~isempty(queue)
                cur = queue(1); queue(1) = [];
                d = obj.Declarations([obj.Declarations.EntityID] == cur);
                if isempty(d), continue, end
                for dep = d.DependsOn
                    if ~any(active == dep)
                        if ~any(live == dep)
                            error('OceanAgents:Time:UnknownEntity', 'Entity %g depends on %g, which is not alive.', cur, dep);
                        end
                        active(end+1) = dep; queue(end+1) = dep; %#ok<AGROW>
                        reasons(end+1) = struct('ID', dep, 'Reason', sprintf('dependency of entity %g', cur)); %#ok<AGROW>
                    end
                end
                for res = d.Resources
                    for o = obj.Declarations
                        if any(o.Resources == res) && ~any(active == o.EntityID) && any(live == o.EntityID)
                            active(end+1) = o.EntityID; queue(end+1) = o.EntityID; %#ok<AGROW>
                            reasons(end+1) = struct('ID', o.EntityID, ...
                                'Reason', sprintf('shares resource "%s" with entity %g', res, cur)); %#ok<AGROW>
                        end
                    end
                end
            end
            p = struct('Active', sort(active), 'ExpandedBy', reasons, 'Requested', requested, ...
                'Equivalence', "", 'Frozen', setdiff(live, active));
            if isempty(p.Frozen)
                p.Equivalence = "full";
            else
                p.Equivalence = "selective-registration: frozen participants act as fixed boundary conditions; " + ...
                    "not equivalent to the full world wherever they would have influenced the outcome";
            end
        end

        function rep = apply(obj, world, requested, opts)
            % APPLY  Plan, then change the world's active set at its current commit boundary. With Expand = false a
            % set that needs expansion is REFUSED before anything changes (DependencyNotActive).
            arguments
                obj
                world
                requested (1,:) double
                opts.Expand (1,1) logical = true
                opts.Deadlines (1,1) string = ""
            end
            p = obj.plan(world, requested);
            added = p.ExpandedBy;
            if ~isempty(added), added = added(~startsWith([added.Reason], "action-driven")); end
            if ~isempty(added) && ~opts.Expand
                lines = arrayfun(@(e) sprintf('%g (%s)', e.ID, e.Reason), added, 'UniformOutput', false);
                error('OceanAgents:Time:DependencyNotActive', ...
                    'The requested set is missing required participants: %s. Request them or allow expansion.', ...
                    strjoin(string(lines), '; '));
            end
            inactive = reshape(world.inactiveEntityIDs(), 1, []);
            live = reshape(world.liveEntityIDs(), 1, []);
            toActivate = intersect(p.Active, inactive);
            toDeactivate = setdiff(live, [p.Active inactive]);
            if ~isempty(toDeactivate) && ~all(ismember(toDeactivate, live))
                error('OceanAgents:Time:UnknownEntity', 'Cannot deactivate an entity that is not alive.');
            end
            if ~isempty(toActivate)
                if opts.Deadlines == ""
                    world.activate(toActivate');         % the world refuses an undeclared policy before it mutates
                else
                    world.activate(toActivate', 'Deadlines', opts.Deadlines);
                end
            end
            if ~isempty(toDeactivate), world.deactivate(toDeactivate'); end
            if isa(world, 'OceanAgents.Time.KernelDomain'), obj.Revision = world.activeRevision();
            elseif isa(world, 'OceanAgents.World.AgentWorld'), obj.Revision = world.activeRevision();
            else, obj.Revision = world.ActiveRevision;
            end
            rep = struct('Active', p.Active, 'Activated', toActivate, 'Deactivated', toDeactivate, ...
                'ExpandedBy', p.ExpandedBy, 'Revision', obj.Revision, 'Equivalence', p.Equivalence);
        end
    end
end

function s = plural_(n)
if n == 1, s = 'y is'; else, s = 'ies are'; end
end
