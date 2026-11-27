classdef RuleSpec
    % RULESPEC  Declaration of one rule of a door: when it runs, which entities it processes, what it may read, write,
    % emit and command, and the pure function that computes it. The rule only ever receives what it declares; the
    % World checks its output against the declaration before anything is applied.
    %
    %   r = OceanAgents.World.RuleSpec('Name', "Move", 'Phase', "act", 'Requires', ["Position" "Velocity"], ...
    %         'Reads', ["Position" "Velocity"], 'Writes', "Position", ...
    %         'DimChecks', struct('Target', "Position", 'Terms', {{"Position", ["Velocity" "dt"]}}), ...
    %         'Fn', @(ctx) struct('Writes', struct('Component', "Position", 'Rows', ctx.Rows, ...
    %                                               'Values', ctx.Read.Position + ctx.Read.Velocity, 'Rates', [])));
    %
    %   Kind "rule"       : may write/emit/command
    %        "projection" : read-only; returns out.View (struct of named arrays) for agents' views
    %        "predictor"  : read-only; returns out.Predictions (Rows, Times) = when Predicts next needs attention
    %   Reveals : another door's HIDDEN components this rule reads and may reveal through its writes/events. A rule that
    %             reads another door's hidden component without declaring it here is refused at construction (V12), so
    %             every route by which hidden information can reach a view is explicit and reviewable.
    % Contract: W1_Kernel_Contract.md section 4.1.

    properties (SetAccess = immutable)
        Name string = ""
        Phase string = ""
        Kind string = "rule"
        Requires string = strings(1, 0)
        Reads string = strings(1, 0)
        Writes string = strings(1, 0)
        Consumes string = strings(1, 0)
        Emits string = strings(1, 0)
        Commands string = strings(1, 0)
        Predicts string = ""
        DueEvent string = ""
        DimChecks struct = struct('Target', {}, 'Terms', {})
        Reveals string = strings(1, 0)
        Fn = []
    end

    properties (Constant)
        Kinds = ["rule" "projection" "predictor"]
        CommandKinds = ["spawn" "remove" "attach" "detach"]
    end

    methods
        function obj = RuleSpec(options)
            arguments
                options.Name
                options.Phase
                options.Kind = "rule"
                options.Requires = strings(1, 0)
                options.Reads = strings(1, 0)
                options.Writes = strings(1, 0)
                options.Consumes = strings(1, 0)
                options.Emits = strings(1, 0)
                options.Commands = strings(1, 0)
                options.Predicts = ""
                options.DueEvent = ""
                options.DimChecks = struct('Target', {}, 'Terms', {})
                options.Reveals = strings(1, 0)
                options.Fn
            end
            bad = @(msg) error('OceanAgents:World:BadRule', 'RuleSpec %s: %s', string(getfield_(options, 'Name')), msg);
            if ~all(isfield(options, {'Name', 'Phase', 'Fn'})), bad('Name, Phase and Fn are required.'); end
            obj.Name = scalarText_(options.Name, 'Name', bad);
            obj.Phase = scalarText_(options.Phase, 'Phase', bad);
            obj.Kind = string(options.Kind);
            if ~any(obj.Kind == obj.Kinds), bad('Kind must be "rule", "projection" or "predictor".'); end
            obj.Requires = row_(options.Requires);
            obj.Reads = row_(options.Reads);
            obj.Writes = row_(options.Writes);
            obj.Consumes = row_(options.Consumes);
            obj.Emits = row_(options.Emits);
            obj.Commands = row_(options.Commands);
            if ~all(ismember(obj.Commands, obj.CommandKinds)), bad('Commands must be spawn/remove/attach/detach.'); end
            obj.Predicts = string(options.Predicts);
            obj.DueEvent = string(options.DueEvent);
            if ~isa(options.Fn, 'function_handle'), bad('Fn must be a function handle.'); end
            obj.Fn = options.Fn;
            if obj.Kind ~= "rule" && (~isempty(obj.Writes) || ~isempty(obj.Emits) || ~isempty(obj.Commands))
                bad('projections and predictors may not write, emit or command.');
            end
            if obj.Kind == "predictor"
                if strlength(obj.Predicts) == 0 || strlength(obj.DueEvent) == 0
                    bad('a predictor needs Predicts and DueEvent.');
                end
            elseif strlength(obj.Predicts) > 0 || strlength(obj.DueEvent) > 0
                bad('only predictors set Predicts/DueEvent.');
            end
            dc = options.DimChecks;
            if ~isstruct(dc) || (~isempty(dc) && ~all(isfield(dc, {'Target', 'Terms'})))
                bad('DimChecks must be a struct array with fields Target and Terms.');
            end
            for k = 1:numel(dc)
                if ~iscell(dc(k).Terms), bad('DimChecks(k).Terms must be a cell of string arrays.'); end
            end
            obj.DimChecks = reshape(dc, 1, []);
            obj.Reveals = row_(options.Reveals);
            if ~all(ismember(obj.Reveals, obj.Reads)), bad('Reveals must be a subset of Reads.'); end
        end
    end
end

function v = getfield_(s, f)
if isfield(s, f), v = s.(f); else, v = "<unnamed>"; end
end

function s = scalarText_(x, name, bad)
s = string(x);
if ~isscalar(s) || strlength(s) == 0, bad(sprintf('%s must be a non-empty string.', name)); end
end

function r = row_(x)
r = reshape(string(x), 1, []);
if isempty(r), r = strings(1, 0); end
end
