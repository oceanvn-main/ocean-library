classdef Feedback
    % FEEDBACK  What one agent is told after a committed world transition: its own reward increment, how long the
    % transition took, whether its episode ended, and the view for its next decision.
    %
    %   fb = OceanAgents.Core.Feedback('AgentID', "alice", 'EpisodeID', 1, 'FrameID', 3, 'Reward', -1, ...
    %            'NextView', v2);
    %
    % Terminated = true MDP terminal (no bootstrapping); Truncated = cut off from outside (bootstrapping allowed).
    % When the episode ended, NextView carries the final pre-reset observation and an all-false mask; for a truncation
    % the successor legality may be given in NextView.Info.successor_action_mask.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 2.3.

    properties (SetAccess = immutable)
        AgentID string = ""
        EpisodeID double = 0
        FrameID double = 0
        Reward double = 0
        RewardValid logical = true
        Duration double = 1
        Terminated logical = false
        Truncated logical = false
        NextView
        Reason string = ""
    end

    methods
        function obj = Feedback(options)
            arguments
                options.AgentID
                options.EpisodeID
                options.FrameID
                options.Reward = 0
                options.RewardValid = true
                options.Duration = 1
                options.Terminated = false
                options.Truncated = false
                options.NextView
                options.Reason = ""
            end
            requireFields_(options, {'AgentID', 'EpisodeID', 'FrameID', 'NextView'}, 'Feedback');
            obj.AgentID = asString_(options.AgentID, 'AgentID');
            obj.EpisodeID = asId_(options.EpisodeID, 'EpisodeID');
            obj.FrameID = asId_(options.FrameID, 'FrameID');
            r = asFinite_(options.Reward, 'Reward');
            if ~isscalar(r), fail_('InvalidRecord', 'Reward must be a scalar.'); end
            obj.RewardValid = asFlag_(options.RewardValid, 'RewardValid');
            if ~obj.RewardValid && r ~= 0
                fail_('InvalidRecord', 'An absent reward (RewardValid = false) must be 0.');
            end
            obj.Reward = r;
            d = asId_(options.Duration, 'Duration');
            if d < 1, fail_('InvalidRecord', 'Duration must be a positive integer.'); end
            obj.Duration = d;
            obj.Terminated = asFlag_(options.Terminated, 'Terminated');
            obj.Truncated = asFlag_(options.Truncated, 'Truncated');
            if obj.Terminated && obj.Truncated
                fail_('InvalidRecord', 'Terminated and Truncated cannot both be true (termination takes precedence).');
            end
            nv = options.NextView;
            if ~(isa(nv, 'OceanAgents.Core.ActorView') && isscalar(nv))
                fail_('InvalidRecord', 'NextView must be a scalar OceanAgents.Core.ActorView.');
            end
            if nv.AgentID ~= obj.AgentID
                fail_('InvalidRecord', 'NextView belongs to %s, not %s.', nv.AgentID, obj.AgentID);
            end
            % the successor must come from the same episode and a later frame than the decision it answers
            if nv.EpisodeID ~= obj.EpisodeID || nv.FrameID <= obj.FrameID
                fail_('StaleFrame', 'NextView (episode %d, frame %d) cannot follow a decision on episode %d, frame %d.', ...
                    nv.EpisodeID, nv.FrameID, obj.EpisodeID, obj.FrameID);
            end
            if (obj.Terminated || obj.Truncated) && any(nv.ActionMask)
                fail_('InvalidRecord', 'An ended episode''s NextView must have an all-false ActionMask.');
            end
            if isfield(nv.Info, 'successor_action_mask')
                m = nv.Info.successor_action_mask;
                if ~obj.Truncated
                    fail_('InvalidRecord', 'successor_action_mask is only meaningful for a truncated transition.');
                end
                if ~(islogical(m) && isequal(size(m), size(nv.ActionMask)))
                    fail_('InvalidRecord', 'successor_action_mask must be a logical row matching ActionLabels.');
                end
            end
            obj.NextView = nv;
            obj.Reason = asString_(options.Reason, 'Reason', true);
        end

        function tf = ended(obj)
            % ENDED  This agent's episode is over (terminated or truncated).
            tf = obj.Terminated || obj.Truncated;
        end

        function m = successorMask(obj)
            % SUCCESSORMASK  Legality at the successor observation: the next view's mask, or for a truncation the
            % declared successor_action_mask (all false when none was supplied).
            if obj.Truncated
                if isfield(obj.NextView.Info, 'successor_action_mask')
                    m = obj.NextView.Info.successor_action_mask;
                else
                    m = false(size(obj.NextView.ActionMask));
                end
            else
                m = obj.NextView.ActionMask;
            end
        end
    end
end
