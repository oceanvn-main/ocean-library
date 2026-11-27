classdef FailingObserver < OceanAgents.Observe.IPlayerObserver
    % FAILINGOBSERVER  Test fixture: an observer that throws on the first decision it sees (a broken display).
    methods
        function onDecision(~, ~, ~, ~)
            error('OceanTest:FailingObserver:Broken', 'display broke');
        end
        function onFeedback(~, ~, ~)
        end
    end
end
