classdef DigitObservation < OceanAgents.Body.Sensors.IObservation
    % DIGITOBSERVATION  Reports the world's numeric observation: a number (Width 1) or a vector of Width numbers
    % (a cell index, CartPole's four numbers, a battery level). AB2 section 4.
    %
    %   s = OceanAgents.Body.Sensors.DigitObservation();                 % one number
    %   s = OceanAgents.Body.Sensors.DigitObservation('Width', 4);       % four numbers

    properties (SetAccess = private)
        Width double = 1
    end

    methods
        function obj = DigitObservation(options)
            arguments
                options.Width (1,1) double {mustBeInteger, mustBePositive} = 1
                options.Dimension = OceanAgents.World.Guard.Dimension()
                options.ID (1,1) string = "digits"
            end
            if options.Width == 1
                t = OceanAgents.Minds.Signals.SignalType('ValueKind', "number", 'Role', "sense", 'Dimension', options.Dimension);
            else
                t = OceanAgents.Minds.Signals.SignalType('ValueKind', "vector", 'Width', options.Width, 'Role', "sense", ...
                    'Dimension', options.Dimension);
            end
            obj.init_(options.ID, "digit", t);
            obj.Width = options.Width;
        end
    end

    methods (Access = protected)
        function r = read_(~, view, ~)
            r = struct('Value', double(view.Observation));
        end

        function out = act(obj, signal)
            out = obj.emit_(reshape(signal.Value.Value, 1, []));
        end
    end
end
