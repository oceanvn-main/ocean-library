classdef PolicyArtifact
    % POLICYARTIFACT  A trained (or hand-written) tabular policy as portable value data, together with everything
    % needed to interpret it: which state each row is, which external action each column (or entry) means, and for
    % which role it was made. Loading a table without this metadata is how label/column mix-ups happen (RI04).
    %
    %   a = OceanAgents.Artifacts.PolicyArtifact('Kind', "ActionTable", 'Representation', "Label", ...
    %           'Table', piTD, 'ActionLabels', [0 1 2], 'NumStates', 16, 'StateStart', 1, 'Role', "default");
    %   a.validateAgainst(obsSpace, actSpace, "default")    % OceanAgents:Artifacts:Incompatible on any mismatch
    %   cols = a.columnTable()                               % ActionTable entries as columns 1..A (validated)
    %
    % Kind            "ActionTable" (S x 1) | "ProbabilityTable" (S x A) | "QTable" (S x A)
    % Representation  ActionTable only: "Label" (entries are external labels, e.g. the TD learner's pi) or
    %                 "ColumnIndex" (entries are columns, e.g. the Monte Carlo PolicyTable). Other kinds: "ColumnIndex".
    %                 Required: there is deliberately no default, because guessing it is exactly the bug it prevents.
    % ActionLabels    1 x A, the column order of the table (external labels).
    % NumStates, StateStart   DiscreteStateCodec parameters: observation StateStart is row 1.
    %
    % Construction checks types and shapes (OceanAgents:Core:InvalidRecord / NonFinite); whether the entries make sense
    % for the declared representation is checked by validateEntries / validateAgainst (OceanAgents:Artifacts:Incompatible),
    % so an artifact with a bad entry can still be loaded and inspected, but never acted on.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md sections 6.3 and 9.6.

    properties (SetAccess = immutable)
        Kind string = ""
        Representation string = ""
        Table double = []
        ActionLabels double = zeros(1, 0)
        NumStates double = 0
        StateStart double = 1
        Role string = "default"
        Version string = "1"
        Provenance struct = struct()
    end

    properties (Constant, Hidden)
        ProbTol = 1e-12     % row-sum tolerance, same as OceanAgents.Core.Decision
    end

    methods
        function obj = PolicyArtifact(options)
            arguments
                options.Kind (1,1) string {mustBeMember(options.Kind, ["ActionTable", "ProbabilityTable", "QTable"])}
                options.Representation (1,1) string {mustBeMember(options.Representation, ["Label", "ColumnIndex"])}
                options.Table {mustBeNumeric, mustBeReal}
                options.ActionLabels {mustBeNumeric, mustBeReal}
                options.NumStates double = []
                options.StateStart (1,1) double {mustBeInteger} = 1
                options.Role (1,1) string = "default"
                options.Version (1,1) string = "1"
                options.Provenance = struct()      % checked below (no conversion of objects to struct)
            end
            required = {'Kind', 'Representation', 'Table', 'ActionLabels'};
            missing = required(~isfield(options, required));
            if ~isempty(missing)
                error('OceanAgents:Core:InvalidRecord', 'PolicyArtifact requires %s.', strjoin(missing, ', '));
            end
            if options.Kind ~= "ActionTable" && options.Representation ~= "ColumnIndex"
                error('OceanAgents:Core:InvalidRecord', 'A %s must use Representation "ColumnIndex".', options.Kind);
            end
            codec = OceanAgents.Artifacts.ActionCodec(options.ActionLabels);   % unique, finite, non-empty
            T = double(options.Table);
            if ~all(isfinite(T(:)))
                error('OceanAgents:Core:NonFinite', 'PolicyArtifact.Table contains NaN or Inf.');
            end
            if isempty(options.NumStates), options.NumStates = size(T, 1); end
            S = options.NumStates;
            if ~(isscalar(S) && S >= 1 && S == fix(S))
                error('OceanAgents:Core:InvalidRecord', 'NumStates must be a positive integer.');
            end
            wantCols = codec.numActions();
            if options.Kind == "ActionTable", wantCols = 1; end
            if ~ismatrix(T) || ~isequal(size(T), [S wantCols])
                error('OceanAgents:Core:InvalidRecord', 'A %s over %d states and %d actions must be %d x %d (got %s).', ...
                    options.Kind, S, codec.numActions(), S, wantCols, mat2str(size(T)));
            end
            obj.Kind = options.Kind;
            obj.Representation = options.Representation;
            obj.Table = T;
            obj.ActionLabels = codec.Labels;
            obj.NumStates = S;
            obj.StateStart = options.StateStart;
            obj.Role = options.Role;
            obj.Version = options.Version;
            if ~(isstruct(options.Provenance) && isscalar(options.Provenance))
                error('OceanAgents:Core:InvalidRecord', 'Provenance must be a scalar struct, got %s.', class(options.Provenance));
            end
            assertValueData_(options.Provenance, 'Provenance');   % no handles anywhere inside (value data)
            obj.Provenance = options.Provenance;
        end

        function c = actionCodec(obj)
            % ACTIONCODEC  Label <-> column codec of this artifact's columns.
            c = OceanAgents.Artifacts.ActionCodec(obj.ActionLabels);
        end

        function c = stateCodec(obj)
            % STATECODEC  Observation -> row codec of this artifact's rows.
            c = OceanAgents.Artifacts.DiscreteStateCodec(obj.NumStates, 'Start', obj.StateStart);
        end

        function validateEntries(obj)
            % VALIDATEENTRIES  Throw OceanAgents:Artifacts:Incompatible if a table entry is out of range for its
            % declared representation: a label not in ActionLabels, a column not in 1..A, or a probability row that is
            % negative or (when it carries any mass) does not sum to 1. All-zero probability rows are "no policy here"
            % (e.g. terminal states) and are allowed; acting on one is refused by the policy.
            A = numel(obj.ActionLabels);
            switch obj.Kind
                case "ActionTable"
                    if obj.Representation == "Label"
                        bad = ~ismember(obj.Table, obj.ActionLabels);
                        what = sprintf('not an action label %s', mat2str(obj.ActionLabels));
                    else
                        bad = obj.Table ~= fix(obj.Table) | obj.Table < 1 | obj.Table > A;
                        what = sprintf('not a column in 1..%d', A);
                    end
                    if any(bad)
                        rows = find(bad);
                        error('OceanAgents:Artifacts:Incompatible', ...
                            'ActionTable (%s) entry %s at row %d is %s.', obj.Representation, ...
                            mat2str(obj.Table(rows(1))), rows(1), what);
                    end
                case "ProbabilityTable"
                    if any(obj.Table(:) < 0)
                        error('OceanAgents:Artifacts:Incompatible', 'ProbabilityTable has negative entries.');
                    end
                    s = sum(obj.Table, 2);
                    bad = s > 0 & abs(s - 1) > obj.ProbTol;
                    if any(bad)
                        r = find(bad, 1);
                        error('OceanAgents:Artifacts:Incompatible', ...
                            'ProbabilityTable row %d sums to %.17g, not 1.', r, s(r));
                    end
                case "QTable"
                    % any finite value is a valid action value (finiteness checked at construction)
            end
        end

        function cols = columnTable(obj)
            % COLUMNTABLE  ActionTable entries converted to columns 1..A through the codec and Representation
            % (validated first). Label representation: column = codec.column(entry); ColumnIndex: entry itself.
            if obj.Kind ~= "ActionTable"
                error('OceanAgents:Artifacts:Incompatible', 'columnTable is defined for an ActionTable, not a %s.', obj.Kind);
            end
            obj.validateEntries();
            if obj.Representation == "Label"
                cols = reshape(obj.actionCodec().column(obj.Table), [], 1);
            else
                cols = obj.Table;
            end
        end

        function validateAgainst(obj, obsSpace, actSpace, role)
            % VALIDATEAGAINST  Throw OceanAgents:Artifacts:Incompatible unless this artifact fits a role whose
            % observation space is obsSpace and action space is actSpace (both OceanStats.RL.Env.Spaces.Discrete):
            % same role, same NumStates/StateStart, same action-label SET, and every entry in range.
            arguments
                obj
                obsSpace
                actSpace
                role (1,1) string
            end
            if role ~= obj.Role
                error('OceanAgents:Artifacts:Incompatible', 'Artifact is for role "%s", not "%s".', obj.Role, role);
            end
            if ~isa(obsSpace, 'OceanStats.RL.Env.Spaces.Discrete')
                error('OceanAgents:Artifacts:Incompatible', 'A tabular artifact needs a Discrete observation space, got %s.', ...
                    class(obsSpace));
            end
            if obsSpace.N ~= obj.NumStates || obsSpace.Start ~= obj.StateStart
                error('OceanAgents:Artifacts:Incompatible', ...
                    'Artifact states are %d starting at %d; the observation space is %s.', ...
                    obj.NumStates, obj.StateStart, obsSpace.describe());
            end
            if ~isa(actSpace, 'OceanStats.RL.Env.Spaces.Discrete')
                error('OceanAgents:Artifacts:Incompatible', 'A tabular artifact needs a Discrete action space, got %s.', ...
                    class(actSpace));
            end
            envLabels = actSpace.Start:(actSpace.Start + actSpace.N - 1);
            if ~obj.actionCodec().sameLabelSet(envLabels)
                error('OceanAgents:Artifacts:Incompatible', 'Artifact action labels %s differ from the role''s labels %s.', ...
                    mat2str(obj.ActionLabels), mat2str(envLabels));
            end
            obj.validateEntries();
        end
    end
end
