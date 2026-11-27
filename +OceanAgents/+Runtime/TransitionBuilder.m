classdef TransitionBuilder < handle
    % TRANSITIONBUILDER  Pairs one agent's decision with the feedback that followed it into a private Transition.
    % One per AgentHost. The pairing is strict: open exactly once per decision, close exactly once per feedback.
    %
    %   tb = OceanAgents.Runtime.TransitionBuilder();
    %   tb.open(view, decision);                 % at decision time
    %   t  = tb.close(feedback, gamma);          % after the world transition committed
    %
    % A1 is E0 (every physical transition has Duration = 1), so
    %   Reward   = feedback.Reward                        (0 when the reward is absent)
    %   Discount = gamma ^ Duration                        (A0 section 1, "Time")
    %   NextMask = feedback.successorMask()               (for a truncation: the declared successor legality,
    %                                                      all false when none was supplied)
    % Opening twice, or closing with nothing open -> OceanAgents:Runtime:TransitionOrder.
    % See Planning/System/Game Theory/A0_Contract_Freeze.md section 4.2.

    properties (SetAccess = private)
        IsOpen logical = false
        View = []            % ActorView the pending decision was made on
        Decision = []        % the pending Decision
    end

    methods
        function open(obj, view, decision)
            % OPEN  Record the decision (and the view it answered) that the next close() will complete.
            arguments
                obj
                view (1,1) OceanAgents.Core.ActorView
                decision (1,1) OceanAgents.Core.Decision
            end
            if obj.IsOpen
                fail_('TransitionOrder', 'A decision for frame %d is already open; close it before opening another.', ...
                    obj.Decision.FrameID);
            end
            obj.View = view;
            obj.Decision = decision;
            obj.IsOpen = true;
        end

        function t = close(obj, feedback, gamma)
            % CLOSE  Complete the open decision with its feedback; returns the Transition and clears the builder.
            arguments
                obj
                feedback (1,1) OceanAgents.Core.Feedback
                gamma (1,1) double {mustBeBetween(gamma, 0, 1)}
            end
            if ~obj.IsOpen
                fail_('TransitionOrder', 'Feedback for frame %d arrived with no open decision.', feedback.FrameID);
            end
            dec = obj.Decision;
            if feedback.AgentID ~= dec.AgentID
                error('OceanAgents:Core:InvalidRecord', 'Feedback for %s cannot close a decision by %s.', ...
                    feedback.AgentID, dec.AgentID);
            end
            if feedback.EpisodeID ~= dec.EpisodeID || feedback.FrameID ~= dec.FrameID
                error('OceanAgents:Core:StaleFrame', ...
                    'Feedback for episode %d frame %d cannot close the decision on episode %d frame %d.', ...
                    feedback.EpisodeID, feedback.FrameID, dec.EpisodeID, dec.FrameID);
            end
            next = feedback.NextView;
            t = OceanAgents.Core.Transition( ...
                'AgentID', dec.AgentID, ...
                'EpisodeID', dec.EpisodeID, ...
                'Observation', obj.View.Observation, ...
                'Action', dec.Action, ...
                'ActionLabels', obj.View.ActionLabels, ...
                'Reward', feedback.Reward, ...                 % Duration 1: the single reward (0 if absent)
                'NextObservation', next.Observation, ...
                'NextActionLabels', next.ActionLabels, ...
                'NextMask', feedback.successorMask(), ...
                'Terminated', feedback.Terminated, ...
                'Truncated', feedback.Truncated, ...
                'Duration', feedback.Duration, ...
                'Discount', gamma ^ feedback.Duration, ...     % Discount = gamma^Duration
                'PolicyVersion', dec.PolicyVersion, ...
                'Likelihood', dec.Likelihood);
            obj.reset();
        end

        function reset(obj)
            % RESET  Drop any open decision (used when a host is invalidated or state is imported).
            obj.View = [];
            obj.Decision = [];
            obj.IsOpen = false;
        end

        function meta = getMetadata(obj)
            meta = pillarMeta_(obj, {'Duration = 1 in A1 (E0); Discount = gamma^Duration; Reward = feedback reward'}, ...
                "A0 contract section 4.2 (decision-interval transition record)");
        end

        function spec = getPipelineSpec(obj)
            spec = pillarSpec_(obj, ["Agent-ActorView" "Agent-Decision" "Agent-Feedback"], "Agent-Transition");
        end
    end
end
