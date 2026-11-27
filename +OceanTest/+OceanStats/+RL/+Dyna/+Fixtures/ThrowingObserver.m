classdef ThrowingObserver < OceanStats.RL.Dyna.Observers.IObserver
    %THROWINGOBSERVER Fixture proving ordered fail-stop observer delivery.
    methods
        function valueUpdated(~, ~)
            error('OceanTest:RL:Dyna:ObserverFailure', 'Deliberate observer failure.');
        end
    end
end
