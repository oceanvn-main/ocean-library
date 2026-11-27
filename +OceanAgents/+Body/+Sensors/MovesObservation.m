classdef MovesObservation < OceanAgents.Body.Sensors.IObservation
    % MOVESOBSERVATION  Reports which of the body's moves the Laws permit now (design D26): "flags", one per move
    % label, in label order (the labels are part of the type, so it can never be mistaken for numbers). AB2 section 4.
    %
    %   s = OceanAgents.Body.Sensors.MovesObservation(string(1:9));      % the nine cells of a tic-tac-toe board

    properties (SetAccess = private)
        Labels string
    end

    methods
        function obj = MovesObservation(labels, options)
            arguments
                labels string
                options.ID (1,1) string = "moves"
            end
            labels = reshape(labels, 1, []);
            t = OceanAgents.Minds.Signals.SignalType('ValueKind', "flags", 'Width', numel(labels), 'Labels', labels, ...
                'Role', "sense");
            obj.init_(options.ID, "moves", t);
            obj.Labels = labels;
        end
    end

    methods (Access = protected)
        function r = read_(~, view, ~)
            r = struct('Mask', logical(view.ActionMask));
        end

        function out = act(obj, signal)
            out = obj.emit_(reshape(signal.Value.Mask, 1, []));
        end
    end
end
