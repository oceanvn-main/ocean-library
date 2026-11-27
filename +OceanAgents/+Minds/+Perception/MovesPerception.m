classdef MovesPerception < OceanAgents.Minds.Perception.IPerception
    % MOVESPERCEPTION  Takes the moves observation (flags over the move labels) as the moves the Laws permit now.
    % It can read the observation of a body with MORE moves than the mind uses (a body may offer more, AB1 Dualism
    % rule): every label of the mind must be among the observation's labels. perceive returns Labels and Permitted
    % (logical) in the observation's order, so the mind lines them up with any labelling.

    properties (SetAccess = private)
        Labels string
    end

    methods
        function obj = MovesPerception(labels)
            arguments
                labels string
            end
            labels = reshape(labels, 1, []);
            obj.ID = "moves";
            obj.Use = "mask";
            obj.Labels = labels;
            obj.Accepts = OceanAgents.Minds.Signals.SignalType('ValueKind', "flags", 'Width', numel(labels), ...
                'Labels', labels, 'Role', "sense");
        end

        function tf = canRead(obj, offered)
            tf = offered.ValueKind == "flags" && offered.Role == "sense" && offered.Shape == "event" && ...
                ~isempty(offered.Labels) && all(ismember(obj.Labels, offered.Labels));
        end

        function x = perceive(~, signal)
            x = struct('Labels', signal.Type.Labels, 'Permitted', logical(signal.Value));
        end
    end
end
