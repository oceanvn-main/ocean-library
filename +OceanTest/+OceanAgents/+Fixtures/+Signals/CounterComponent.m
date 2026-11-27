classdef CounterComponent < OceanAgents.Components.IComponent
    % COUNTERCOMPONENT  Test fixture: accepts label commands, refuses the label "forbidden" (verify), counts accepted
    % commands (act) and replies with the running count. Supports the "calendar" and "formula" bindings so the
    % binding-equivalence rule can be exercised.
    properties (SetAccess = private)
        Count double = 0
        Last string = ""
    end
    methods
        function obj = CounterComponent(id, labels)
            arguments
                id (1,1) string = "counter"
                labels string = ["left" "right" "forbidden"]
            end
            T = OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', labels, 'Role', "command");
            obj.setDeclarations_(id, T, OceanAgents.Minds.Signals.SignalType.reply(), ["calendar" "formula"]);
        end
        function s = exportState(obj)
            s = struct('Count', obj.Count, 'Last', obj.Last);
        end
        function importState(obj, s)
            obj.Count = s.Count;  obj.Last = s.Last;
        end
    end

    methods (Access = protected)
        function [ok, reason] = verify(~, signal)
            ok = signal.Value ~= "forbidden";
            reason = "the label ""forbidden"" is refused";
        end

        function out = act(obj, signal)
            obj.Count = obj.Count + 1;
            obj.Last = signal.Value;
            out = OceanAgents.Minds.Signals.Signal.ok(struct('Count', obj.Count));
        end
    end
end
