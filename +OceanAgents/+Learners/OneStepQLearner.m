classdef OneStepQLearner < OceanAgents.Core.ILearner
    % ONESTEPQLEARNER  Tabular one-step Q-learning (Watkins 1989; Sutton & Barto 2018 eq. 6.8) on the agent's own
    % Transition records:
    %   Q(s,a) <- Q(s,a) + alpha [ r + gamma^k (1 - d) max_{a' in L(s')} Q(s',a') - Q(s,a) ],
    % with k = Duration (gamma^k = Transition.Discount), d = Terminated, and L(s') the successor legality from the
    % transition (NextMask; for a truncation, the declared successor mask). Truncation is NOT termination: it
    % bootstraps, so it needs L(s'); a truncated transition without one is refused (Learners:MissingSuccessorMask).
    %
    %   q   = OceanStats.RL.TD.Values.TabularQ(true(48, 4));
    %   lrn = OceanAgents.Learners.OneStepQLearner(q, 1:4, 'Alpha', 0.5, 'Gamma', 1, ...
    %             'StateCodec', OceanAgents.Artifacts.DiscreteStateCodec(48));
    %   lrn.consume(t);            % validates and stores; changes nothing yet
    %   n = lrn.update("step");    % applies every stored transition in order; n = #Q updates
    %   n = lrn.update("episode"); % 0: this learner's boundary is "step"
    %
    % The update itself is OceanStats.RL.TD.Values.TabularQ.oneStep with QLearningTarget and ConstantStepSize(alpha);
    % this class only translates the agent-level record (external labels, observations) into table indices through
    % the ActionCodec / DiscreteStateCodec. q is a HANDLE shared with the same agent's policy (intended, intra-agent).
    % See Planning/System/Game Theory/A0_Contract_Freeze.md sections 6.5 and 9.2; metadata in meta/OneStepQLearner.meta.json.

    properties (SetAccess = private)
        Q                           % OceanStats.RL.TD.Values.TabularQ (shared with this agent's policy)
        Alpha (1,1) double          % constant step size in (0, 1]
        Gamma (1,1) double          % discount per unit duration in [0, 1]; must equal the owning host's Gamma
        ActionCodec                 % OceanAgents.Artifacts.ActionCodec: Q column <-> external label
        StateCodec                  % OceanAgents.Artifacts.DiscreteStateCodec: observation -> Q row
        NumUpdates (1,1) double = 0 % Q updates applied so far
    end

    properties (Access = private)
        Pending_ = struct('si', {}, 'ai', {}, 'r', {}, 's2i', {}, 'term', {}, 'discount', {}, 'valid2', {})
        Target_                     % OceanStats.RL.TD.Targets.QLearningTarget
        Step_                       % OceanStats.RL.TD.StepSize.ConstantStepSize(Alpha)
    end

    properties (Constant, Hidden)
        DiscountTol = 1e-12
    end

    methods
        function obj = OneStepQLearner(q, labels, options)
            arguments
                q (1,1) OceanStats.RL.TD.Values.TabularQ
                labels {mustBeNumeric, mustBeReal}
                options.Alpha (1,1) double {mustBePositive, mustBeLessThanOrEqual(options.Alpha, 1)} = 0.1
                options.Gamma (1,1) double {mustBeBetween(options.Gamma, 0, 1)} = 1
                options.StateCodec = []
            end
            codec = OceanAgents.Artifacts.ActionCodec(labels);
            [S, A] = size(q.Q);
            if A ~= codec.numActions()
                error('OceanAgents:Learners:Incompatible', 'Q has %d columns but %d action labels were given.', ...
                    A, codec.numActions());
            end
            states = options.StateCodec;
            if isempty(states), states = OceanAgents.Artifacts.DiscreteStateCodec(S); end
            if ~isa(states, 'OceanAgents.Artifacts.DiscreteStateCodec') || states.N ~= S
                error('OceanAgents:Learners:Incompatible', 'StateCodec must be a DiscreteStateCodec over the %d rows of Q.', S);
            end
            obj.Q = q;
            obj.Alpha = options.Alpha;
            obj.Gamma = options.Gamma;
            obj.ActionCodec = codec;
            obj.StateCodec = states;
            obj.Target_ = OceanStats.RL.TD.Targets.QLearningTarget();
            obj.Step_ = OceanStats.RL.TD.StepSize.ConstantStepSize(options.Alpha);
        end

        function d = describe(obj)
            d = struct('Name', "OneStepQLearner", 'Version', "1", 'Gamma', obj.Gamma, 'Boundary', "step", ...
                'Alpha', obj.Alpha, 'ActionLabels', obj.ActionCodec.Labels);
        end

        function checkCompatible(obj, policyDescription)
            % CHECKCOMPATIBLE  A policy that declares action labels must declare this learner's labels (same set).
            % A label-agnostic policy (empty ActionLabels, e.g. RandomLegalPolicy) is accepted: Q-learning is
            % off-policy, so any behaviour policy over the same environment actions is valid.
            arguments
                obj
                policyDescription (1,1) struct
            end
            name = "policy";
            if isfield(policyDescription, 'Name'), name = string(policyDescription.Name); end
            if isfield(policyDescription, 'ActionLabels') && ~isempty(policyDescription.ActionLabels)
                if ~obj.ActionCodec.sameLabelSet(policyDescription.ActionLabels)
                    error('OceanAgents:Learners:Incompatible', '%s acts on labels %s; the learner''s Q columns are %s.', ...
                        name, mat2str(policyDescription.ActionLabels), mat2str(obj.ActionCodec.Labels));
                end
            elseif isfield(policyDescription, 'NumActions') && ~isempty(policyDescription.NumActions) && ...
                    policyDescription.NumActions ~= obj.ActionCodec.numActions()
                error('OceanAgents:Learners:Incompatible', '%s acts over %d actions; the learner has %d.', ...
                    name, policyDescription.NumActions, obj.ActionCodec.numActions());
            end
        end

        function consume(obj, t)
            % CONSUME  Validate one Transition and store it (table indices only); Q is not touched until update.
            arguments
                obj
                t (1,1) OceanAgents.Core.Transition
            end
            codec = obj.ActionCodec;
            if ~codec.sameLabelSet(t.ActionLabels)
                error('OceanAgents:Learners:Incompatible', 'Transition action labels %s differ from the Q column labels %s.', ...
                    mat2str(t.ActionLabels), mat2str(codec.Labels));
            end
            expected = obj.Gamma ^ t.Duration;
            if abs(t.Discount - expected) > obj.DiscountTol
                error('OceanAgents:Learners:Incompatible', ...
                    'Transition discount %.17g is not Gamma^Duration = %.17g (Gamma %.17g, Duration %d).', ...
                    t.Discount, expected, obj.Gamma, t.Duration);
            end
            si = obj.StateCodec.encode(t.Observation);
            ai = codec.column(t.Action);
            valid2 = false(1, codec.numActions());
            if t.Terminated
                s2i = si;       % placeholder: a terminal target never reads Q(s', .)
            else
                if ~any(t.NextMask)
                    error('OceanAgents:Learners:MissingSuccessorMask', ...
                        ['%s transition (episode %d) has no legal successor action, so max_a'' Q(s'',a'') is undefined. ' ...
                        'A truncation must carry NextView.Info.successor_action_mask.'], ...
                        ternary_(t.Truncated, "Truncated", "Non-terminal"), t.EpisodeID);
                end
                if ~codec.sameLabelSet(t.NextActionLabels)
                    error('OceanAgents:Learners:Incompatible', 'Successor labels %s differ from the Q column labels %s.', ...
                        mat2str(t.NextActionLabels), mat2str(codec.Labels));
                end
                s2i = obj.StateCodec.encode(t.NextObservation);
                valid2(codec.column(t.NextActionLabels(t.NextMask))) = true;   % successor legality in Q columns
            end
            obj.Pending_(end + 1) = struct('si', si, 'ai', ai, 'r', t.Reward, 's2i', s2i, ...
                'term', t.Terminated, 'discount', t.Discount, 'valid2', valid2);
        end

        function n = update(obj, boundary)
            % UPDATE  "step": apply every stored transition in arrival order; "episode": nothing is due (returns 0).
            arguments
                obj
                boundary (1,1) string {mustBeMember(boundary, ["step", "episode"])}
            end
            n = 0;
            if boundary ~= "step", return; end
            for k = 1:numel(obj.Pending_)
                p = obj.Pending_(k);
                obj.Q.oneStep(obj.Target_, obj.Step_, p.si, p.ai, p.r, p.s2i, [], p.term, p.discount, [], p.valid2);
                n = n + 1;
            end
            obj.Pending_ = obj.Pending_([]);
            obj.NumUpdates = obj.NumUpdates + n;
        end

        function n = numPending(obj)
            % NUMPENDING  Transitions consumed but not yet applied.
            n = numel(obj.Pending_);
        end

        function s = exportState(obj)
            % EXPORTSTATE  Numeric copies of Q and its visit counts, plus settings and any pending transitions.
            s = struct('Name', "OneStepQLearner", 'Q', obj.Q.Q, 'Counts', obj.Q.Counts, 'Alpha', obj.Alpha, ...
                'Gamma', obj.Gamma, 'ActionLabels', obj.ActionCodec.Labels, 'NumUpdates', obj.NumUpdates, ...
                'Pending', obj.Pending_);
        end

        function importState(obj, s)
            % IMPORTSTATE  Load exported values INTO this learner's own TabularQ (the handle shared with its policy is
            % kept, so the pair stays consistent and nothing aliases the exporting agent).
            % Values are copied in with TabularActionValues.setTables (numeric copies: nothing aliases the exporter).
            arguments
                obj
                s (1,1) struct
            end
            need = {'Name', 'Q', 'Counts', 'Alpha', 'Gamma', 'ActionLabels', 'NumUpdates', 'Pending'};
            if ~all(isfield(s, need)) || s.Name ~= "OneStepQLearner"
                error('OceanAgents:Learners:Incompatible', 'State is not a OneStepQLearner state.');
            end
            if ~isequal(size(s.Q), size(obj.Q.Q)) || ~isequal(size(s.Counts), size(obj.Q.Counts)) || ...
                    ~obj.ActionCodec.matches(s.ActionLabels) || s.Alpha ~= obj.Alpha || s.Gamma ~= obj.Gamma
                error('OceanAgents:Learners:Incompatible', 'State does not fit this learner (table size, labels, Alpha or Gamma).');
            end
            if ~all(isfinite(s.Q(:))) || any(s.Counts(:) < 0 | s.Counts(:) ~= fix(s.Counts(:)))
                error('OceanAgents:Learners:Incompatible', 'State Q must be finite and Counts non-negative integers.');
            end
            if ~isstruct(s.Pending) || ~isempty(setxor(fieldnames(s.Pending), fieldnames(obj.Pending_)))
                error('OceanAgents:Learners:Incompatible', 'State Pending is not a list of stored transitions.');
            end
            obj.validatePending_(s.Pending);
            if ~(isnumeric(s.NumUpdates) && isscalar(s.NumUpdates) && s.NumUpdates >= 0 && s.NumUpdates == fix(s.NumUpdates))
                error('OceanAgents:Learners:Incompatible', 'State NumUpdates must be a non-negative integer.');
            end
            obj.Q.setTables(double(s.Q), double(s.Counts));
            obj.NumUpdates = s.NumUpdates;
            obj.Pending_ = reshape(s.Pending, 1, []);
        end

        function spec = getPipelineSpec(~)
            % GETPIPELINESPEC  Pillar 3: Transition in, updated action-value table out.
            spec = struct();
            spec.ComponentName = "OneStepQLearner";
            spec.Inputs = struct('Name', "transition", 'Domain', "Agent-Transition", 'Dtype', "OceanAgents.Core.Transition", 'Shape', "1 x 1");
            spec.Outputs = struct('Name', "Q", 'Domain', "RL-ActionValues", 'Dtype', "double", 'Shape', "S x A");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "qlearning");
            spec.ComposesFrom = {"OceanStats.RL.TD.Values.TabularQ", "OceanStats.RL.TD.Targets.QLearningTarget", ...
                "OceanStats.RL.TD.StepSize.ConstantStepSize"};
        end
    end

    methods (Access = private)
        function validatePending_(obj, pending)
            % VALIDATEPENDING_  Each imported pending entry must be exactly what consume() would have stored, so a
            % corrupt state fails here (Learners:Incompatible) rather than later inside update().
            [S, A] = size(obj.Q.Q);
            isIndex = @(x, n) isnumeric(x) && isreal(x) && isscalar(x) && x == fix(x) && x >= 1 && x <= n;
            isFlag = @(x) islogical(x) && isscalar(x);
            for k = 1:numel(pending)
                p = pending(k);
                ok = isIndex(p.si, S) && isIndex(p.ai, A) && isIndex(p.s2i, S) ...
                    && isnumeric(p.r) && isreal(p.r) && isscalar(p.r) && isfinite(p.r) ...
                    && isFlag(p.term) ...
                    && isnumeric(p.discount) && isscalar(p.discount) && p.discount >= 0 && p.discount <= 1 ...
                    && islogical(p.valid2) && isequal(size(p.valid2), [1 A]) ...
                    && (p.term || any(p.valid2));
                if ~ok
                    error('OceanAgents:Learners:Incompatible', ...
                        'State Pending entry %d is not a valid stored transition for a %d x %d table.', k, S, A);
                end
            end
        end
    end

    methods (Static)
        function meta = getMetadata()
            % GETMETADATA  Pillar 1: pedagogical metadata from meta/OneStepQLearner.meta.json plus the OceanAgents
            % pillar fields (SymbolicLatex, Symbols, IsSymbolic, Reference; A0 section 8).
            here = fileparts(mfilename('fullpath'));
            meta = jsondecode(fileread(fullfile(here, 'meta', 'OneStepQLearner.meta.json')));
            meta.SymbolicLatex = string(meta.equations.update.latex);
            meta.Assumptions = meta.pedagogy.assumptions;
            meta.IsSymbolic = true;
            meta.Reference = string(meta.textbook.reference);
        end
    end
end

function out = ternary_(cond, a, b)
if cond, out = a; else, out = b; end
end
