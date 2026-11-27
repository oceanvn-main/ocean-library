classdef RecorderComponent < OceanAgents.Components.IComponent
    % RECORDERCOMPONENT  Test fixture: accepts record commands and keeps the last stamped signal it received, so a test
    % can try to replay that signal somewhere else.
    properties (SetAccess = private)
        Last = []
    end
    methods
        function obj = RecorderComponent(id)
            T = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "command");
            obj.setDeclarations_(id, T, OceanAgents.Minds.Signals.SignalType.empty(1, 0), "calendar");
        end
    end
    methods (Access = protected)
        function [ok, reason] = verify(~, ~)
            ok = true;  reason = "";
        end
        function out = act(obj, signal)
            obj.Last = signal;
            out = OceanAgents.Minds.Signals.Signal.empty(1, 0);
        end
    end
end
