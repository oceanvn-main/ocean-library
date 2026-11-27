classdef SpyObserver < OceanAgents.Observe.IPlayerObserver
    % SPYOBSERVER  Test fixture: an observer that holds something (e.g. another player) - Player.attach must refuse
    % it when that something is a player or one of its parts.
    properties
        Held = []
    end
    methods
        function obj = SpyObserver(held)
            obj.Held = {held};
        end
        function onDecision(~, ~, ~, ~)
        end
        function onFeedback(~, ~, ~)
        end
    end
end
