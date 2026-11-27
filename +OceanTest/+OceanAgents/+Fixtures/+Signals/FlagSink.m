classdef FlagSink < handle
    % FLAGSINK  Test fixture: a receiver that accepts flag events in the sense role and records value, time, sender.
    properties (SetAccess = private)
        Accepts
        Values = false(1, 0)
        Times = zeros(1, 0)
        Senders = strings(1, 0)
    end
    methods
        function obj = FlagSink()
            obj.Accepts = OceanAgents.Minds.Signals.SignalType('ValueKind', "flag", 'Role', "sense");
        end
        function out = receive(obj, signal)
            obj.Values(end + 1) = signal.Value;
            obj.Times(end + 1) = signal.Time;
            obj.Senders(end + 1) = signal.Sender;
            out = OceanAgents.Minds.Signals.Signal.empty(1, 0);
        end
    end
end
