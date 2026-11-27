classdef Decision
    % DECISION  One proposed primitive action from one agent for one frame. Proposing is not executing:
    % the runner validates it against the view and only the environment decides what actually happens.
    %
    %   d = OceanAgents.Core.Decision('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 3, 'Action', 2, ...
    %           'Probabilities', [0.1 0.9], 'Likelihood', 0.9, 'PolicyVersion', 4);
    %   d.validateAgainst(view)     throws unless d is a legal, consistent answer to view
    %
    % Action is an external label. Probabilities (optional) are over the view's columns; Likelihood (optional) is the
    % behaviour probability of Action and must equal Probabilities at its column when both are given.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 2.2.

    properties (SetAccess = immutable)
        AgentID string = ""
        EpisodeID double = 0
        FrameID double = 0
        Action double = 0
        Probabilities double = []
        Likelihood double = []
        PolicyVersion double = 0
        Provenance struct = struct()
    end

    properties (Constant, Hidden)
        ProbTol = 1e-12
    end

    methods
        function obj = Decision(options)
            arguments
                options.AgentID
                options.EpisodeID
                options.FrameID
                options.Action
                options.Probabilities = []
                options.Likelihood = []
                options.PolicyVersion = 0
                options.Provenance = struct()
            end
            requireFields_(options, {'AgentID', 'EpisodeID', 'FrameID', 'Action'}, 'Decision');
            obj.AgentID = asString_(options.AgentID, 'AgentID');
            obj.EpisodeID = asId_(options.EpisodeID, 'EpisodeID');
            obj.FrameID = asId_(options.FrameID, 'FrameID');
            a = asFinite_(options.Action, 'Action');
            if ~isscalar(a), fail_('InvalidRecord', 'Action must be a scalar label.'); end
            obj.Action = a;
            p = asFinite_(options.Probabilities, 'Probabilities');
            if ~isempty(p)
                if ~isrow(p), fail_('InvalidRecord', 'Probabilities must be a row vector.'); end
                if any(p < 0) || abs(sum(p) - 1) > obj.ProbTol
                    fail_('InvalidRecord', 'Probabilities must be non-negative and sum to 1 (sum = %.17g).', sum(p));
                end
            end
            obj.Probabilities = p;
            lik = asFinite_(options.Likelihood, 'Likelihood');
            if ~isempty(lik) && (~isscalar(lik) || lik <= 0 || lik > 1)
                fail_('InvalidRecord', 'Likelihood must be a scalar in (0, 1].');
            end
            obj.Likelihood = lik;
            obj.PolicyVersion = asId_(options.PolicyVersion, 'PolicyVersion');
            obj.Provenance = asStruct_(options.Provenance, 'Provenance');
        end

        function tf = hasLikelihood(obj)
            tf = ~isempty(obj.Likelihood);
        end

        function validateAgainst(obj, view)
            % VALIDATEAGAINST  Throw unless this decision is a legal, self-consistent answer to view.
            arguments
                obj
                view (1,1) OceanAgents.Core.ActorView
            end
            if obj.AgentID ~= view.AgentID
                fail_('InvalidRecord', 'Decision is from %s but the view belongs to %s.', obj.AgentID, view.AgentID);
            end
            if obj.EpisodeID ~= view.EpisodeID || obj.FrameID ~= view.FrameID
                fail_('StaleFrame', 'Decision for episode %d frame %d answers view episode %d frame %d.', ...
                    obj.EpisodeID, obj.FrameID, view.EpisodeID, view.FrameID);
            end
            if ~view.isLegal(obj.Action)
                fail_('IllegalAction', 'Action %s is not legal for %s (legal: %s).', ...
                    mat2str(obj.Action), obj.AgentID, mat2str(view.legalLabels()));
            end
            if ~isempty(obj.Probabilities)
                if numel(obj.Probabilities) ~= numel(view.ActionLabels)
                    fail_('InvalidRecord', 'Probabilities has %d entries; the view has %d actions.', ...
                        numel(obj.Probabilities), numel(view.ActionLabels));
                end
                if any(obj.Probabilities(~view.ActionMask) > 0)
                    fail_('IllegalAction', 'Probabilities put mass on illegal actions.');
                end
                if obj.hasLikelihood()
                    pAct = obj.Probabilities(view.columnOf(obj.Action));
                    if abs(pAct - obj.Likelihood) > obj.ProbTol
                        fail_('InvalidRecord', 'Likelihood %.17g differs from Probabilities at the action (%.17g).', ...
                            obj.Likelihood, pAct);
                    end
                end
            end
        end
    end
end
