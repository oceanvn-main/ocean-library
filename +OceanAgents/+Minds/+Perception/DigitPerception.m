classdef DigitPerception < OceanAgents.Minds.Perception.IPerception
    % DIGITPERCEPTION  Takes a digit observation (a number, or a vector of Width numbers) as the mind's state.

    methods
        function obj = DigitPerception(options)
            arguments
                options.Width (1,1) double {mustBeInteger, mustBePositive} = 1
                options.Required (1,1) logical = true
            end
            obj.ID = "digits";
            obj.Use = "state";
            if options.Width == 1
                obj.Accepts = OceanAgents.Minds.Signals.SignalType('ValueKind', "number", 'Role', "sense");
            else
                obj.Accepts = OceanAgents.Minds.Signals.SignalType('ValueKind', "vector", 'Width', options.Width, 'Role', "sense");
            end
            obj.Required = options.Required;
        end

        function x = perceive(~, signal)
            x = signal.Value;
        end
    end
end
