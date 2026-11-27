classdef DecisionFrame
    % DECISIONFRAME  Output of a joint environment: one committed world state seen through N per-slot views, plus the
    % reward/ending events of the transition that produced it. Slot k belongs to Roster(k).
    %
    %   Requested(k)  slot k must submit exactly one decision for this frame (Requested is a subset of Live)
    %   Views{k}      ActorView for slot k (present when requested or when slot k has final feedback), else []
    %   Rewards(k)    reward increment of slot k from the last transition (never a cumulative return)
    %   ResearchInfo  diagnostics for researchers; never handed to an agent
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 2.5.

    properties (SetAccess = immutable)
        EpisodeID double = 0
        FrameID double = 0
        Tick double = 0
        Phase string = "act"
        Roster string = strings(1, 0)
        Live logical = false(1, 0)
        Requested logical = false(1, 0)
        Views cell = {}
        Rewards double = zeros(1, 0)
        RewardValid logical = false(1, 0)
        Terminated logical = false(1, 0)
        Truncated logical = false(1, 0)
        EpisodeEnded logical = false
        EndReason string = ""
        ResearchInfo struct = struct()
    end

    methods
        function obj = DecisionFrame(options)
            arguments
                options.EpisodeID
                options.FrameID
                options.Tick = 0
                options.Phase = "act"
                options.Roster
                options.Live
                options.Requested
                options.Views
                options.Rewards = []
                options.RewardValid = []
                options.Terminated = []
                options.Truncated = []
                options.EpisodeEnded = false
                options.EndReason = ""
                options.ResearchInfo = struct()
            end
            requireFields_(options, {'EpisodeID', 'FrameID', 'Roster', 'Live', 'Requested', 'Views'}, 'DecisionFrame');
            obj.EpisodeID = asId_(options.EpisodeID, 'EpisodeID');
            obj.FrameID = asId_(options.FrameID, 'FrameID');
            obj.Tick = asId_(options.Tick, 'Tick');
            obj.Phase = asString_(options.Phase, 'Phase');
            roster = options.Roster;
            if iscellstr(roster) || ischar(roster) || isstring(roster), roster = string(roster); end
            if ~(isstring(roster) && isrow(roster)) || any(ismissing(roster)) || any(strlength(roster) == 0) ...
                    || numel(unique(roster)) ~= numel(roster)
                fail_('InvalidRecord', 'Roster must be a row of unique non-empty strings.');
            end
            n = numel(roster);
            obj.Roster = roster;
            obj.Live = asRowFlags_(options.Live, n, 'Live');
            obj.Requested = asRowFlags_(options.Requested, n, 'Requested');
            if any(obj.Requested & ~obj.Live)
                fail_('InvalidRecord', 'Requested slots must be live.');
            end
            obj.Rewards = defaultRow_(options.Rewards, zeros(1, n), 'Rewards', n, true);
            obj.RewardValid = asRowFlags_(defaultRow_(options.RewardValid, false(1, n), 'RewardValid', n, false), n, 'RewardValid');
            obj.Terminated = asRowFlags_(defaultRow_(options.Terminated, false(1, n), 'Terminated', n, false), n, 'Terminated');
            obj.Truncated = asRowFlags_(defaultRow_(options.Truncated, false(1, n), 'Truncated', n, false), n, 'Truncated');
            if any(obj.Terminated & obj.Truncated)
                fail_('InvalidRecord', 'A slot cannot be both terminated and truncated.');
            end
            if any(~obj.RewardValid & obj.Rewards ~= 0)
                fail_('InvalidRecord', 'Absent rewards (RewardValid = false) must be 0.');
            end
            views = options.Views;
            if ~(iscell(views) && numel(views) == n)
                fail_('InvalidRecord', 'Views must be a 1 x %d cell.', n);
            end
            views = reshape(views, 1, []);
            for k = 1:n
                v = views{k};
                if isempty(v)
                    if obj.Requested(k), fail_('InvalidRecord', 'Requested slot %d (%s) has no view.', k, roster(k)); end
                    continue
                end
                if ~(isa(v, 'OceanAgents.Core.ActorView') && isscalar(v))
                    fail_('InvalidRecord', 'Views{%d} must be an OceanAgents.Core.ActorView or [].', k);
                end
                if v.AgentID ~= roster(k) || v.EpisodeID ~= obj.EpisodeID || v.FrameID ~= obj.FrameID
                    fail_('InvalidRecord', 'Views{%d} does not match slot %s / episode %d / frame %d.', ...
                        k, roster(k), obj.EpisodeID, obj.FrameID);
                end
                if obj.Requested(k) && v.isTerminalView()
                    fail_('InvalidRecord', 'Requested slot %s has no legal action.', roster(k));
                end
            end
            obj.Views = views;
            obj.EpisodeEnded = asFlag_(options.EpisodeEnded, 'EpisodeEnded');
            obj.EndReason = asString_(options.EndReason, 'EndReason', true);
            obj.ResearchInfo = asStruct_(options.ResearchInfo, 'ResearchInfo');
        end

        function n = numSlots(obj)
            n = numel(obj.Roster);
        end

        function k = slotOf(obj, agentID)
            % SLOTOF  Slot index of an AgentID (error if not on the roster).
            k = find(obj.Roster == string(agentID), 1);
            if isempty(k), fail_('InvalidRecord', 'AgentID %s is not on the roster.', string(agentID)); end
        end

        function fb = feedbackFor(obj, k, decisionFrameID, duration)
            % FEEDBACKFOR  Feedback for slot k about the transition that followed its decision on decisionFrameID.
            if nargin < 4, duration = 1; end
            v = obj.Views{k};
            if isempty(v), fail_('InvalidRecord', 'Slot %d has no view to deliver as feedback.', k); end
            reason = "";
            if obj.Terminated(k) || obj.Truncated(k), reason = obj.EndReason; end
            fb = OceanAgents.Core.Feedback('AgentID', obj.Roster(k), 'EpisodeID', obj.EpisodeID, ...
                'FrameID', decisionFrameID, 'Reward', obj.Rewards(k), 'RewardValid', obj.RewardValid(k), ...
                'Duration', duration, 'Terminated', obj.Terminated(k), 'Truncated', obj.Truncated(k), ...
                'NextView', v, 'Reason', reason);
        end
    end
end

function x = defaultRow_(x, default, name, n, numeric)
if isempty(x), x = default; return; end
if numeric
    x = asFinite_(x, name);
    if ~(isrow(x) && numel(x) == n), fail_('InvalidRecord', '%s must be 1 x %d.', name, n); end
end
end
