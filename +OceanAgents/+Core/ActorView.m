classdef ActorView
    % ACTORVIEW  Everything one agent is allowed to see for one decision (an allowlisted value snapshot).
    % Holds value data only, so handing it to a policy cannot open a route back into the world or another agent.
    %
    %   v = OceanAgents.Core.ActorView('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 1, ...
    %           'Observation', 37, 'ActionLabels', 1:4, 'ActionMask', true(1, 4));
    %   v.legalLabels()    external labels the agent may choose now
    %
    % ActionMask(k) refers to ActionLabels(k); labels are the environment's external action values, not table columns.
    % A view whose mask is all false is terminal (no decision may be requested on it).
    % Evidence (1 x K OceanAgents.Core.EvidenceRecord, default none) carries the events this agent may observe since its
    % previous view (A2_Beliefs_Contract.md section 2).
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 2.1.

    properties (SetAccess = immutable)
        AgentID string = ""
        EpisodeID double = 0
        FrameID double = 0
        Tick double = 0
        Phase string = "act"
        Observation double = []
        ActionLabels double = zeros(1, 0)
        ActionMask logical = false(1, 0)
        Info struct = struct()
        Evidence = OceanAgents.Core.EvidenceRecord.empty(1, 0)
    end

    methods
        function obj = ActorView(options)
            arguments
                options.AgentID
                options.EpisodeID
                options.FrameID
                options.Tick = 0
                options.Phase = "act"
                options.Observation
                options.ActionLabels
                options.ActionMask
                options.Info = struct()
                options.Evidence = OceanAgents.Core.EvidenceRecord.empty(1, 0)
            end
            requireFields_(options, {'AgentID', 'EpisodeID', 'FrameID', 'Observation', 'ActionLabels', 'ActionMask'}, 'ActorView');
            obj.AgentID = asString_(options.AgentID, 'AgentID');
            obj.EpisodeID = asId_(options.EpisodeID, 'EpisodeID');
            obj.FrameID = asId_(options.FrameID, 'FrameID');
            obj.Tick = asId_(options.Tick, 'Tick');
            obj.Phase = asString_(options.Phase, 'Phase');
            obj.Observation = asFinite_(options.Observation, 'Observation');
            labels = asFinite_(options.ActionLabels, 'ActionLabels');
            if ~(isrow(labels) || isempty(labels)) || numel(unique(labels)) ~= numel(labels)
                fail_('InvalidRecord', 'ActionLabels must be a row of unique values.');
            end
            obj.ActionLabels = reshape(labels, 1, []);
            obj.ActionMask = asRowFlags_(options.ActionMask, numel(labels), 'ActionMask');
            obj.Info = asStruct_(options.Info, 'Info');
            ev = options.Evidence;
            if isempty(ev), ev = OceanAgents.Core.EvidenceRecord.empty(1, 0); end
            if ~isa(ev, 'OceanAgents.Core.EvidenceRecord')
                fail_('InvalidRecord', 'Evidence must be OceanAgents.Core.EvidenceRecord objects.');
            end
            if ~isrow(ev) && ~isempty(ev)
                fail_('InvalidRecord', 'Evidence must be a 1 x K row (delivery order must be explicit).');
            end
            ev = reshape(ev, 1, []);
            ids = [ev.EvidenceID];
            if numel(unique(ids)) ~= numel(ids)
                fail_('InvalidRecord', 'Evidence in one view must have distinct EvidenceIDs.');
            end
            obj.Evidence = ev;
        end

        function ids = evidenceIDs(obj)
            % EVIDENCEIDS  IDs of the evidence carried by this view (1 x K string).
            ids = [obj.Evidence.EvidenceID];
            if isempty(ids), ids = strings(1, 0); end
        end

        function labels = legalLabels(obj)
            % LEGALLABELS  External labels of the actions that are legal in this view.
            labels = obj.ActionLabels(obj.ActionMask);
        end

        function tf = isLegal(obj, label)
            % ISLEGAL  True when label is one of the legal external labels.
            tf = isnumeric(label) && isscalar(label) && any(obj.ActionLabels(obj.ActionMask) == label);
        end

        function tf = isTerminalView(obj)
            % ISTERMINALVIEW  No legal action: the agent's episode has ended at this view.
            tf = ~any(obj.ActionMask);
        end

        function k = columnOf(obj, label)
            % COLUMNOF  Column of an external label in this view's ActionLabels (error if unknown).
            k = find(obj.ActionLabels == label, 1);
            if isempty(k)
                fail_('IllegalAction', 'Label %s is not an action label of this view.', mat2str(label));
            end
        end
    end
end
