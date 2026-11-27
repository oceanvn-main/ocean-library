classdef EmptyBody < OceanAgents.Body.IBody
    % EMPTYBODY  A body with no components: it forwards the mind's label command unchanged as the action (chess: the
    % behavior is the move; design D5). Every agent has a body; this one only forwards.
    %
    %   body = OceanAgents.Body.EmptyBody(["1" "2" "3" "4"]);                       % digit + moves sensors
    %   body = OceanAgents.Body.EmptyBody(string(1:9), 'Sensors', {OceanAgents.Body.Sensors.MovesObservation(string(1:9))});

    properties (Access = private)
        Pending_ string = strings(0, 1)
    end

    methods
        function obj = EmptyBody(labels, options)
            arguments
                labels string
                options.ID (1,1) string = "body"
                options.Sensors = []
            end
            obj.ID = options.ID;
            labels = reshape(labels, 1, []);
            sensors = options.Sensors;
            if isempty(sensors)                       % default: the digit observation and what the Laws permit
                sensors = {OceanAgents.Body.Sensors.DigitObservation(), OceanAgents.Body.Sensors.MovesObservation(labels)};
            end
            obj.Sensors = obj.checkSensors_(sensors);
            [senses, parts] = obj.sensorDeclarations_(obj.Sensors);
            obj.Signature = OceanAgents.Body.BodySignature(OceanAgents.Body.IBody.moveType(labels), senses, parts);
        end

        function replies = command(obj, signal)
            if signal.Type.ValueKind ~= "label" || ~any(obj.Signature.Commands(1).Labels == signal.Value)
                replies = OceanAgents.Minds.Signals.Signal.refusal(sprintf('%s does not accept %s', obj.ID, ...
                    signal.Type.describe()));
                return
            end
            obj.Pending_ = string(signal.Value);
            replies = OceanAgents.Minds.Signals.Signal.ok();
        end

        function label = toAction(obj)
            if isempty(obj.Pending_), obj.noAction_(); end
            label = obj.Pending_;
            obj.Pending_ = strings(0, 1);
        end

        function beginEpisode(obj)
            obj.Pending_ = strings(0, 1);
        end
    end
end
