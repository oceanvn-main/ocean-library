classdef EnvSpec
    % ENVSPEC  Versioned description of an environment: who takes part, how decisions are timed, what each role sees
    % and may do, and which optional capabilities exist. Describing never mutates an episode.
    %
    %   spec = OceanAgents.Core.EnvSpec('Name', "CliffWalking", 'Timing', "single", 'Roster', "agent1", ...
    %            'ObservationSpaces', {obsSpace}, 'ActionSpaces', {actSpace});
    %
    % Spaces are OceanStats.RL.Env.Spaces.ISpace descriptions; they are value objects, so sharing them aliases nothing.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 2.7.

    properties (SetAccess = immutable)
        Name string = ""
        SchemaVersion string = "1"
        Timing string = "single"
        Roster string = strings(1, 0)
        Roles string = strings(1, 0)
        ObservationSpaces cell = {}
        ActionSpaces cell = {}
        Capabilities = OceanAgents.Core.CapabilitySet()
        Horizon double = Inf
    end

    properties (Constant)
        Timings = ["single" "simultaneous" "turn"]
    end

    methods
        function obj = EnvSpec(options)
            arguments
                options.Name
                options.SchemaVersion = "1"
                options.Timing
                options.Roster
                options.Roles = []
                options.ObservationSpaces
                options.ActionSpaces
                options.Capabilities = OceanAgents.Core.CapabilitySet()
                options.Horizon = Inf
            end
            requireFields_(options, {'Name', 'Timing', 'Roster', 'ObservationSpaces', 'ActionSpaces'}, 'EnvSpec');
            obj.Name = asString_(options.Name, 'Name');
            obj.SchemaVersion = asString_(options.SchemaVersion, 'SchemaVersion');
            obj.Timing = asString_(options.Timing, 'Timing');
            if ~any(obj.Timing == obj.Timings)
                fail_('InvalidRecord', 'Timing must be one of: %s.', strjoin(obj.Timings, ', '));
            end
            roster = string(options.Roster);
            if ~isrow(roster) || any(strlength(roster) == 0) || numel(unique(roster)) ~= numel(roster)
                fail_('InvalidRecord', 'Roster must be a row of unique non-empty strings.');
            end
            n = numel(roster);
            if obj.Timing == "single" && n ~= 1
                fail_('InvalidRecord', 'A "single" timing environment has exactly one roster entry.');
            end
            obj.Roster = roster;
            roles = options.Roles;
            if isempty(roles), roles = repmat("default", 1, n); end
            roles = string(roles);
            if numel(roles) ~= n, fail_('InvalidRecord', 'Roles must have one entry per roster slot.'); end
            obj.Roles = reshape(roles, 1, []);
            obj.ObservationSpaces = spaces_(options.ObservationSpaces, n, 'ObservationSpaces');
            obj.ActionSpaces = spaces_(options.ActionSpaces, n, 'ActionSpaces');
            if ~(isa(options.Capabilities, 'OceanAgents.Core.CapabilitySet') && isscalar(options.Capabilities))
                fail_('InvalidRecord', 'Capabilities must be an OceanAgents.Core.CapabilitySet.');
            end
            obj.Capabilities = options.Capabilities;
            h = options.Horizon;
            if ~(isnumeric(h) && isscalar(h) && ~isnan(h) && h > 0 && (isinf(h) || h == fix(h)))
                fail_('InvalidRecord', 'Horizon must be a positive integer or Inf.');
            end
            obj.Horizon = double(h);
        end

        function n = numSlots(obj)
            n = numel(obj.Roster);
        end

        function k = slotOf(obj, agentID)
            k = find(obj.Roster == string(agentID), 1);
            if isempty(k), fail_('InvalidRecord', 'AgentID %s is not on the roster.', string(agentID)); end
        end
    end
end

function c = spaces_(c, n, name)
if ~iscell(c), c = {c}; end
if numel(c) ~= n, fail_('InvalidRecord', '%s must have one space per roster slot.', name); end
c = reshape(c, 1, []);
for k = 1:n
    if ~(isa(c{k}, 'OceanStats.RL.Env.Spaces.ISpace') && isscalar(c{k}))
        fail_('InvalidRecord', '%s{%d} must be an OceanStats.RL.Env.Spaces.ISpace.', name, k);
    end
end
end
