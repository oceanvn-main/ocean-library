classdef PlayerTrace < OceanAgents.Observe.IPlayerObserver
    % PLAYERTRACE  Records what a player decided and was paid, step by step: the simplest observer, and the data a
    % display or a replay recorder would draw from.
    %
    %   t = OceanAgents.Observe.PlayerTrace();  player.attach(t);  ... run ...
    %   t.Actions, t.Rewards, t.Episodes

    properties (SetAccess = private)
        Actions = []
        Episodes double = zeros(1, 0)
        Rewards double = zeros(1, 0)
        Ended logical = false(1, 0)
    end

    methods
        function onDecision(obj, ~, ~, decision)
            obj.Actions(end + 1) = decision.Action;
            obj.Episodes(end + 1) = decision.EpisodeID;
        end

        function onFeedback(obj, ~, feedback)
            obj.Rewards(end + 1) = feedback.Reward;
            obj.Ended(end + 1) = feedback.ended();
        end
    end
end
