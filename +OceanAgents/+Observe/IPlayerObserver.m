classdef (Abstract) IPlayerObserver < handle
    % IPLAYEROBSERVER  Something that follows a Player without taking part: a display, a recorder, a live plot
    % (PG1 contract section 4). It receives value records only (the view, the decision, the feedback), so it cannot
    % change anything. If it throws, the player detaches it and records the reason in Failure; the run goes on.

    properties (SetAccess = private)
        Failure string = ""
    end

    methods (Abstract)
        onDecision(obj, playerID, view, decision)
        onFeedback(obj, playerID, feedback)
    end

    methods (Access = {?OceanAgents.Runtime.AgentHost})
        function failed_(obj, message)
            obj.Failure = string(message);
        end
    end
end
