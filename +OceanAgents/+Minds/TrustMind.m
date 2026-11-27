classdef TrustMind < OceanAgents.Core.IDecisionProcess
    % TRUSTMIND  A mind that decides whom to trust with a task. It keeps beliefs about providers in a belief store,
    % feeds the store the evidence its views carry, and turns the beliefs into one choice with a declared rule:
    % delegate to provider j, or take the outside option.
    %
    %   store = OceanAgents.Beliefs.BetaTrustStore('Prior', [1 1]);
    %   rule  = OceanAgents.Minds.TrustRules.ThompsonRule();
    %   pay   = struct('cheap', [10 -20 0], 'expensive', [30 -40 0]);   % [Success Failure Outside] per context
    %   mind  = OceanAgents.Minds.TrustMind(store, rule, pay);
    %   mind.describe().Profile            % "evidence-trust-adaptive" (store UpdateRule "beta-bernoulli")
    %
    % What each view must carry (A2 world contract): Info.context (string), Info.action_subjects (1 x A string
    % aligned with ActionLabels; "outside" marks the outside option) and Evidence (events the agent may observe).
    %
    % Order of use (A2 contract section 4):
    %   beginEpisode(view)  store.beginEpisode(), then store.observe(view.Evidence)
    %   observe(feedback)   store.observe(feedback.NextView.Evidence)  - evidence is used only after the transition
    %   decide(view,stream) reads belief(subject, context) for every LEGAL delegate action (never consumes evidence),
    %                       builds the candidates in increasing label order and calls rule.choose(candidates, stream)
    % The Decision carries Probabilities/Likelihood only when the rule gives them, and a Provenance trace of the
    % actual inputs with exactly the fields Rule, Context, BeliefVersion, Labels, Subjects, Estimates, Variances,
    % EvidenceMass, SourceKinds, Scores, Chosen (entries over the legal actions in increasing label order, given in
    % Labels; the
    % outside option has Estimate/Variance NaN, EvidenceMass 0, SourceKind "outside"; a fixed score has Variance NaN).
    %
    % Profile: "fixed-belief-reactive" for a store with UpdateRule "fixed", "evidence-trust-adaptive" for
    % "beta-bernoulli" (an explicit 'Profile' may only restate that; it is required for any other UpdateRule).
    % Incompatibilities -> OceanAgents:Minds:Incompatible: a rule that needs variances/Beta parameters on a store that
    % cannot give them, a missing Info.context/action_subjects, a context absent from the payoffs, an
    % action_subjects count that differs from the action count, a non-Discrete action space.
    % See Planning/System/Game Theory/A2_Beliefs_Contract.md section 4.

    properties (SetAccess = immutable)
        Store               % OceanAgents.Beliefs.IBeliefStore (this mind's private beliefs)
        Rule                % OceanAgents.Minds.TrustRules.ITrustRule
        Payoffs struct      % context -> [Success Failure Outside]
        Profile string
    end

    properties (SetAccess = private)
        LastDispositions string = strings(1, 0)   % the store's dispositions for the last evidence consumed
        EvidenceConsumed double = 0               % evidence records handed to the store so far
    end

    properties (Constant)
        OutsideSubject = "outside"
    end

    methods
        function obj = TrustMind(store, rule, payoffs, options)
            arguments
                store (1,1) OceanAgents.Beliefs.IBeliefStore
                rule (1,1) OceanAgents.Minds.TrustRules.ITrustRule
                payoffs (1,1) struct
                options.Profile string = strings(0, 0)
            end
            contexts = fieldnames(payoffs);
            if isempty(contexts)
                incompatible_('payoffs must name at least one context.');
            end
            for k = 1:numel(contexts)
                v = payoffs.(contexts{k});
                if ~(isnumeric(v) && isreal(v) && numel(v) == 3 && all(isfinite(v)))
                    incompatible_('payoffs.%s must be a finite [Success Failure Outside] triple.', contexts{k});
                end
                payoffs.(contexts{k}) = double(reshape(v, 1, 3));
            end
            sd = store.describe();
            updateRule = string(sd.UpdateRule);
            % What the store's records provide, and what the rule needs (A2 section 4.1).
            switch updateRule
                case "fixed",          provides = strings(1, 0);          derived = "fixed-belief-reactive";
                case "beta-bernoulli", provides = ["variance" "beta-parameters"]; derived = "evidence-trust-adaptive";
                otherwise,             provides = strings(1, 0);          derived = "";
            end
            rd = rule.describe();
            if isfield(rd, 'Requires')
                missing = setdiff(string(rd.Requires), provides);
                if ~isempty(missing) && derived ~= ""
                    incompatible_('%s needs %s, which a "%s" store does not provide.', string(rd.Name), ...
                        strjoin(missing, ' and '), updateRule);
                end
            end
            if isempty(options.Profile)
                if derived == ""
                    incompatible_('Store UpdateRule "%s" has no standard profile; pass ''Profile''.', updateRule);
                end
                profile = derived;
            else
                profile = options.Profile(1);
                if derived ~= "" && profile ~= derived
                    incompatible_('A "%s" store makes a "%s" mind, not "%s".', updateRule, derived, profile);
                end
            end
            obj.Store = store;
            obj.Rule = rule;
            obj.Payoffs = payoffs;
            obj.Profile = profile;
        end

        function d = describe(obj)
            d = struct('Name', "TrustMind", 'Profile', obj.Profile, ...
                'Components', struct('Store', obj.Store.describe(), 'Rule', obj.Rule.describe()));
        end

        function checkCompatible(~, ~, actSpace, ~)
            % CHECKCOMPATIBLE  Needs a Discrete action space; the action count is checked per view.
            if ~isa(actSpace, 'OceanStats.RL.Env.Spaces.Discrete')
                incompatible_('TrustMind needs a Discrete action space, got %s.', class(actSpace));
            end
        end

        function beginEpisode(obj, view)
            % BEGINEPISODE  Store persistence first, then the evidence carried by the episode's first view.
            obj.Store.beginEpisode();
            obj.consume_(view.Evidence);
        end

        function observe(obj, feedback)
            % OBSERVE  Evidence revealed by the committed transition (it arrives in the next view).
            obj.consume_(feedback.NextView.Evidence);
        end

        function endEpisode(obj)
            obj.Store.endEpisode();
        end

        function dec = decide(obj, view, stream)
            % DECIDE  Beliefs of the legal delegate actions -> rule.choose -> Decision (+ provenance trace).
            [context, subjects] = obj.worldInfo_(view);
            legal = find(view.ActionMask);
            if isempty(legal)
                error('OceanAgents:Core:IllegalAction', 'No legal action in the view of %s.', view.AgentID);
            end
            [labels, order] = sort(view.ActionLabels(legal));   % candidates in increasing label order
            cols = legal(order);
            n = numel(cols);
            c = struct('Labels', labels, 'Subjects', subjects(cols), ...
                'IsOutside', subjects(cols) == obj.OutsideSubject, ...
                'Estimate', nan(1, n), 'Variance', nan(1, n), 'Alpha', nan(1, n), 'Beta', nan(1, n), ...
                'Payoff', obj.Payoffs.(char(context)));
            mass = zeros(1, n);
            kinds = repmat(obj.OutsideSubject, 1, n);
            for j = find(~c.IsOutside)
                rec = obj.Store.belief(c.Subjects(j), context);   % beliefs are keyed by SubjectID, never by slot
                c.Estimate(j) = rec.Estimate;
                if ~isempty(rec.Variance), c.Variance(j) = rec.Variance; end
                if ~isempty(rec.Alpha), c.Alpha(j) = rec.Alpha; c.Beta(j) = rec.Beta; end
                mass(j) = rec.EvidenceMass;
                kinds(j) = rec.SourceKind;
            end
            [label, info] = obj.Rule.choose(c, stream);
            provenance = struct('Rule', string(obj.Rule.describe().Name), 'Context', context, ...
                'BeliefVersion', obj.Store.version(), 'Labels', c.Labels, 'Subjects', c.Subjects, 'Estimates', c.Estimate, ...
                'Variances', c.Variance, 'EvidenceMass', mass, 'SourceKinds', kinds, ...
                'Scores', info.Scores, 'Chosen', label);
            args = {'AgentID', view.AgentID, 'EpisodeID', view.EpisodeID, 'FrameID', view.FrameID, ...
                'Action', label, 'Provenance', provenance};
            if ~isempty(info.Probabilities)
                % the rule's probabilities are over the candidates; the Decision's are over the view's columns
                p = zeros(1, numel(view.ActionLabels));
                p(cols) = info.Probabilities;
                args = [args, {'Probabilities', p, 'Likelihood', p(view.columnOf(label))}];
            end
            dec = OceanAgents.Core.Decision(args{:});
        end

        function s = exportState(obj)
            s = struct('Name', "TrustMind", 'Store', obj.Store.exportState());
        end

        function importState(obj, s)
            arguments
                obj
                s (1,1) struct
            end
            if ~isfield(s, 'Name') || s.Name ~= "TrustMind" || ~isfield(s, 'Store')
                incompatible_('State is not a TrustMind state.');
            end
            obj.Store.importState(s.Store);
        end

        function meta = getMetadata(obj)
            % GETMETADATA  Pillar 1: a container (IsSymbolic = false); the formulas live in the rule and the store.
            meta = struct();
            meta.SymbolicLatex = strings(1, 0);
            meta.Symbols = struct('Symbol', {}, 'Description', {});
            meta.Assumptions = {sprintf('profile %s: beliefs from %s, choice by %s', obj.Profile, ...
                class(obj.Store), class(obj.Rule))};
            meta.IsSymbolic = false;
            meta.Reference = "runtime/container component (A2 contract section 7)";
        end

        function spec = getPipelineSpec(obj)
            % GETPIPELINESPEC  Pillar 3: ActorView (+ evidence) and Feedback in, Decision out.
            spec = struct();
            spec.ComponentName = "TrustMind";
            spec.Inputs = [ ...
                struct('Name', "view", 'Domain', "Agent-ActorView", 'Dtype', "OceanAgents.Core.ActorView", 'Shape', "1 x 1"), ...
                struct('Name', "feedback", 'Domain', "Agent-Feedback", 'Dtype', "OceanAgents.Core.Feedback", 'Shape', "1 x 1")];
            spec.Outputs = struct('Name', "decision", 'Domain', "Agent-Decision", 'Dtype', "OceanAgents.Core.Decision", 'Shape', "1 x 1");
            spec.Kind = "NONVIRTUAL";
            spec.DiagramSymbol = struct('Type', "default", 'Label', "trustmind");
            spec.ComposesFrom = {string(class(obj.Store)), string(class(obj.Rule))};
        end
    end

    methods (Access = private)
        function consume_(obj, evidence)
            % CONSUME_  Hand evidence to the store exactly once (the store also de-duplicates by EvidenceID).
            if isempty(evidence), return; end
            obj.LastDispositions = obj.Store.observe(evidence);
            obj.EvidenceConsumed = obj.EvidenceConsumed + numel(evidence);
        end

        function [context, subjects] = worldInfo_(obj, view)
            % WORLDINFO_  Info.context and Info.action_subjects, validated against the view and the payoffs.
            info = view.Info;
            if ~isfield(info, 'context') || ~(isstring(info.context) || ischar(info.context)) || ...
                    ~isscalar(string(info.context))
                incompatible_('View of %s has no Info.context string.', view.AgentID);
            end
            if ~isfield(info, 'action_subjects') || ~(isstring(info.action_subjects) || iscellstr(info.action_subjects))
                incompatible_('View of %s has no Info.action_subjects string array.', view.AgentID);
            end
            context = string(info.context);
            subjects = reshape(string(info.action_subjects), 1, []);
            if numel(subjects) ~= numel(view.ActionLabels)
                incompatible_('Info.action_subjects has %d entries; the view has %d actions.', ...
                    numel(subjects), numel(view.ActionLabels));
            end
            if ~isvarname(char(context)) || ~isfield(obj.Payoffs, char(context))
                incompatible_('No declared payoffs for context "%s".', context);
            end
        end
    end
end

function incompatible_(fmt, varargin)
error('OceanAgents:Minds:Incompatible', fmt, varargin{:});
end
