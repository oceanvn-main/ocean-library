classdef FixedMovesSensor < OceanAgents.Body.Sensors.IObservation
    % FIXEDMOVESSENSOR  Test fixture: a moves sensor that ignores the view and always reports the same permitted
    % moves, so a test can show the mind decides from its observations only (AB2 invariant I10).
    properties (SetAccess = private)
        Mask logical
    end
    methods
        function obj = FixedMovesSensor(labels, mask)
            labels = reshape(string(labels), 1, []);
            obj.init_("moves", "moves", OceanAgents.Minds.Signals.SignalType('ValueKind', "flags", ...
                'Width', numel(labels), 'Labels', labels, 'Role', "sense"));
            obj.Mask = logical(mask);
        end
    end
    methods (Access = protected)
        function r = read_(obj, ~, ~)
            r = struct('Mask', obj.Mask);
        end
        function out = act(obj, signal)
            out = obj.emit_(signal.Value.Mask);
        end
    end
end
