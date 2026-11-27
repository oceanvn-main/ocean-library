classdef FixedDigitSensor < OceanAgents.Body.Sensors.IObservation
    % FIXEDDIGITSENSOR  Test fixture: a digit sensor that always reports the same number, whatever the world shows, so
    % a test can check the learner learns from what the mind perceived (AB2 I10; critic AB2 finding 1).
    properties (SetAccess = private)
        Value double
    end
    methods
        function obj = FixedDigitSensor(value, options)
            arguments
                value (1,1) double
                options.ID (1,1) string = "digits"
                options.Dimension = OceanAgents.World.Guard.Dimension()
            end
            obj.init_(options.ID, "digit", OceanAgents.Minds.Signals.SignalType('ValueKind', "number", 'Role', "sense", ...
                'Dimension', options.Dimension));
            obj.Value = value;
        end
    end
    methods (Access = protected)
        function r = read_(obj, ~, ~)
            r = struct('Value', obj.Value);
        end
        function out = act(obj, signal)
            out = obj.emit_(signal.Value.Value);
        end
    end
end
