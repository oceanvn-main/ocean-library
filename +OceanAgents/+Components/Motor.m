classdef Motor < OceanAgents.Components.IComponent
    % MOTOR  A body component that turns a move command into the action the world receives. It accepts only the
    % moves it was built with (type check), so a body without diagonals refuses a diagonal (AB1 section 6).
    %
    %   m = OceanAgents.Components.Motor(["up" "right" "down" "left"]);
    %   out = m.receive(moveSignal);     % emits an "action" label event, or a refusal

    properties (SetAccess = private)
        Moves string
        Count double = 0
    end

    methods
        function obj = Motor(moves, options)
            arguments
                moves string
                options.ID (1,1) string = "motor"
            end
            moves = reshape(moves, 1, []);
            T = OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', moves, 'Role', "command");
            obj.setDeclarations_(options.ID, T, T, "calendar");
            obj.Moves = moves;
        end

        function s = exportState(obj)
            s = struct('Count', obj.Count);
        end

        function importState(obj, s)
            obj.Count = s.Count;
        end
    end

    methods (Access = protected)
        function [ok, reason] = verify(~, ~)
            ok = true;  reason = "";
        end

        function out = act(obj, signal)
            obj.Count = obj.Count + 1;
            out = OceanAgents.Minds.Signals.Signal.make(obj.Emits, signal.Value);
        end
    end
end
