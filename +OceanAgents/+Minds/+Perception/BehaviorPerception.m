classdef BehaviorPerception < OceanAgents.Minds.Perception.IPerception
    % BEHAVIORPERCEPTION  Takes the behavior observation: is a body behavior running (design D23). Optional: a body
    % without behaviors simply has no such sensor.

    methods
        function obj = BehaviorPerception()
            obj.ID = "behavior";
            obj.Use = "behavior";
            obj.Accepts = OceanAgents.Body.Sensors.BehaviorObservation.type();
            obj.Required = false;
        end

        function x = perceive(~, signal)
            x = struct('Running', logical(signal.Value(1)));
        end
    end
end
