classdef BehaviorObservation < OceanAgents.Body.Sensors.IObservation
    % BEHAVIOROBSERVATION  Reports the body's own running behavior: Running (true/false), ID and Remaining steps. The
    % mind learns that a behavior is running only through this observation; a behavior never talks to the mind
    % (design D23; AB2 section 4).

    methods
        function obj = BehaviorObservation(options)
            arguments
                options.ID (1,1) string = "behavior"
            end
            obj.init_(options.ID, "behavior", OceanAgents.Body.Sensors.BehaviorObservation.type());
        end
    end

    methods (Static)
        function t = type()
            % TYPE  One flag named "running": distinguishable from every other observation by its type alone.
            t = OceanAgents.Minds.Signals.SignalType('ValueKind', "flags", 'Width', 1, 'Labels', "running", 'Role', "sense");
        end
    end

    methods (Access = protected)
        function r = read_(~, ~, body)
            r = body.behaviorStatus();
        end

        function out = act(obj, signal)
            out = obj.emit_(logical(signal.Value.Running));
        end
    end
end
