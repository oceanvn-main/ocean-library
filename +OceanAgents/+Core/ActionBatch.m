classdef ActionBatch
    % ACTIONBATCH  All decisions submitted for one frame. validate(frame) checks the whole batch before the
    % environment mutates anything: every requested slot exactly once, nothing extra, nothing stale, all legal.
    %
    %   b = OceanAgents.Core.ActionBatch('EpisodeID', 1, 'FrameID', 5, 'Decisions', [d1 d2]);
    %   b.validate(frame);            throws OceanAgents:Core:* on the first problem found
    %   d = b.decisionFor("alice");
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 2.6.

    properties (SetAccess = immutable)
        EpisodeID double = 0
        FrameID double = 0
        Phase string = "act"
        Decisions = OceanAgents.Core.Decision.empty(1, 0)
    end

    methods
        function obj = ActionBatch(options)
            arguments
                options.EpisodeID
                options.FrameID
                options.Phase = "act"
                options.Decisions = OceanAgents.Core.Decision.empty(1, 0)
            end
            requireFields_(options, {'EpisodeID', 'FrameID'}, 'ActionBatch');
            obj.EpisodeID = asId_(options.EpisodeID, 'EpisodeID');
            obj.FrameID = asId_(options.FrameID, 'FrameID');
            obj.Phase = asString_(options.Phase, 'Phase');
            d = options.Decisions;
            if iscell(d)
                if isempty(d), d = OceanAgents.Core.Decision.empty(1, 0); else, d = [d{:}]; end
            end
            if ~isa(d, 'OceanAgents.Core.Decision')
                fail_('InvalidRecord', 'Decisions must be OceanAgents.Core.Decision objects.');
            end
            obj.Decisions = reshape(d, 1, []);
        end

        function ids = agentIDs(obj)
            ids = [obj.Decisions.AgentID];
            if isempty(ids), ids = strings(1, 0); end
        end

        function d = decisionFor(obj, agentID)
            % DECISIONFOR  The decision submitted by agentID (error if absent).
            k = find(obj.agentIDs() == string(agentID), 1);
            if isempty(k), fail_('MissingDecision', 'No decision from %s.', string(agentID)); end
            d = obj.Decisions(k);
        end

        function validate(obj, frame)
            % VALIDATE  Throw before any world mutation unless this batch answers frame exactly.
            arguments
                obj
                frame (1,1) OceanAgents.Core.DecisionFrame
            end
            if obj.EpisodeID ~= frame.EpisodeID || obj.FrameID ~= frame.FrameID || obj.Phase ~= frame.Phase
                fail_('StaleFrame', 'Batch for episode %d frame %d (%s) answers frame %d/%d (%s).', ...
                    obj.EpisodeID, obj.FrameID, obj.Phase, frame.EpisodeID, frame.FrameID, frame.Phase);
            end
            ids = obj.agentIDs();
            if numel(unique(ids)) ~= numel(ids)
                [~, first] = unique(ids, 'stable');
                dup = ids(setdiff(1:numel(ids), first));
                fail_('DuplicateDecision', 'More than one decision from %s.', strjoin(unique(dup), ', '));
            end
            requested = frame.Roster(frame.Requested);
            extra = setdiff(ids, requested);
            if ~isempty(extra)
                fail_('UnexpectedDecision', 'Decision(s) from slot(s) not requested: %s.', strjoin(extra, ', '));
            end
            missing = setdiff(requested, ids);
            if ~isempty(missing)
                fail_('MissingDecision', 'No decision from requested slot(s): %s.', strjoin(missing, ', '));
            end
            for d = obj.Decisions
                if d.EpisodeID ~= frame.EpisodeID || d.FrameID ~= frame.FrameID
                    fail_('StaleFrame', 'Decision from %s is for episode %d frame %d.', d.AgentID, d.EpisodeID, d.FrameID);
                end
                d.validateAgainst(frame.Views{frame.slotOf(d.AgentID)});
            end
        end
    end
end
