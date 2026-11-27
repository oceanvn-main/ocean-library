classdef AxiomSpec
    % AXIOMSPEC  One axiom a door guarantees about its members or admitted modules (design by contract: precondition,
    % postcondition, invariant, or trajectory property). The guard admits newcomers that satisfy the axioms - never by
    % name. A world admits everything its axioms do not exclude.
    %
    %   Range axiom (entity scope; statically analysable, used to detect contradictions):
    %     a = AxiomSpec('ID', "HB-1", 'Scope', "entity", 'Kind', "invariant", 'Mode', "monitor", ...
    %                   'Component', "Speed", 'Range', [0 12]);
    %   Predicate axiom (entity scope): ok = Predicate(values) with values.<Reads{k}> (n x W) -> n x 1 logical
    %     a = AxiomSpec('ID', "HB-2", 'Scope', "entity", 'Kind', "invariant", 'Mode', "static", ...
    %                   'Reads', "Mass", 'Predicate', @(v) v.Mass <= 300);
    %   Module axiom: [ok, detail] = Predicate(module) (static) or Predicate(module, stream) (probe); "assumed" has none.
    % Contract: W1_Kernel_Contract.md section 3.3.

    properties (SetAccess = immutable)
        ID string = ""
        Scope string = "entity"
        Kind string = "invariant"
        Mode string = "static"
        Component string = ""
        Range double = zeros(1, 0)
        Reads string = strings(1, 0)
        Predicate = []
        Probes double = 32
        Description string = ""
    end

    properties (Constant)
        Scopes = ["entity" "module"]
        Kinds = ["precondition" "postcondition" "invariant" "trajectory"]
        Modes = ["static" "probe" "monitor" "assumed"]
    end

    methods
        function obj = AxiomSpec(options)
            arguments
                options.ID
                options.Scope = "entity"
                options.Kind = "invariant"
                options.Mode = "static"
                options.Component = ""
                options.Range = []
                options.Reads = strings(1, 0)
                options.Predicate = []
                options.Probes = 32
                options.Description = ""
            end
            bad = @(msg) error('OceanAgents:Guard:BadAxiom', 'AxiomSpec: %s', msg);
            if ~isfield(options, 'ID'), bad('ID is required.'); end
            obj.ID = string(options.ID);
            if ~isscalar(obj.ID) || strlength(obj.ID) == 0, bad('ID must be a non-empty string.'); end
            obj.Scope = string(options.Scope);
            obj.Kind = string(options.Kind);
            obj.Mode = string(options.Mode);
            if ~any(obj.Scope == obj.Scopes), bad(sprintf('%s: unknown Scope.', obj.ID)); end
            if ~any(obj.Kind == obj.Kinds), bad(sprintf('%s: unknown Kind.', obj.ID)); end
            if ~any(obj.Mode == obj.Modes), bad(sprintf('%s: unknown Mode.', obj.ID)); end
            if obj.Scope == "entity" && ~any(obj.Mode == ["static" "monitor"])
                bad(sprintf('%s: entity axioms are "static" or "monitor".', obj.ID));
            end
            if obj.Scope == "module" && obj.Mode == "monitor"
                bad(sprintf('%s: module axioms are "static", "probe" or "assumed".', obj.ID));
            end
            obj.Component = string(options.Component);
            obj.Reads = reshape(string(options.Reads), 1, []);
            r = double(options.Range);
            isRange = ~isempty(r);
            hasPred = ~isempty(options.Predicate);
            if obj.Mode == "assumed"
                if isRange || hasPred, bad(sprintf('%s: an assumed axiom has no range or predicate.', obj.ID)); end
            elseif isRange == hasPred
                bad(sprintf('%s: set exactly one of Range and Predicate.', obj.ID));
            end
            if isRange
                if obj.Scope ~= "entity", bad(sprintf('%s: range axioms are entity-scope.', obj.ID)); end
                if ~(isequal(size(r), [1 2]) && ~any(isnan(r)) && r(1) <= r(2)) || strlength(obj.Component) == 0
                    bad(sprintf('%s: a range axiom needs a Component and Range [lo hi] with lo <= hi.', obj.ID));
                end
            end
            if hasPred && ~isa(options.Predicate, 'function_handle')
                bad(sprintf('%s: Predicate must be a function handle.', obj.ID));
            end
            obj.Range = r;
            obj.Predicate = options.Predicate;
            p = options.Probes;
            if ~(isnumeric(p) && isscalar(p) && p >= 1 && p == fix(p)), bad('Probes must be a positive integer.'); end
            obj.Probes = double(p);
            obj.Description = string(options.Description);
        end

        function tf = isRange(obj), tf = ~isempty(obj.Range); end

        function names = readsAll(obj)
            % READSALL  Components this axiom reads (the range component or the predicate's Reads).
            if obj.isRange(), names = obj.Component; else, names = obj.Reads; end
        end
    end
end
