classdef (Sealed) Signal
    % SIGNAL  One signal: its type, its value, and the coordinates the router stamps on delivery (time, sender, route,
    % sequence number). A part creates a signal with Signal.make and sends it through its Port. Only the Router can
    % set the delivery coordinates, so a part can never claim to be another sender (AB1_Signal_Body_Contract.md
    % sections 3.2 and 3.4; critic AB1 finding 1).
    %
    %   T = OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', ["left" "right"], 'Role', "command");
    %   s = OceanAgents.Minds.Signals.Signal.make(T, "left");
    %   replies = port.send("motor", s);                     % stamped: Sender = the port's owner

    properties (SetAccess = immutable)
        Type = OceanAgents.Minds.Signals.SignalType()
        Value = struct()
    end

    properties (SetAccess = {?OceanAgents.Minds.Signals.Router})
        Time double = NaN
        Sender string = ""
        Route string = ""
        Seq double = 0
        Via double = 0                          % ID of the router that delivered it (0 = never delivered)
        To string = ""                          % the part it was delivered to (a copy replayed elsewhere is refused)
    end

    methods
        function obj = Signal(type, value)
            if nargin == 0, return; end
            if ~(isa(type, 'OceanAgents.Minds.Signals.SignalType') && isscalar(type))
                error('OceanAgents:Signals:BadType', 'Signal: Type must be one SignalType.');
            end
            type.checkValue(value);
            obj.Type = type;
            if type.ValueKind == "label", value = string(value); end
            if type.ValueKind == "flag" || type.ValueKind == "flags", value = logical(value); end
            obj.Value = value;
        end

        function tf = isStamped(obj)
            tf = strlength(obj.Sender) > 0;
        end

        function tf = isRefusal(obj)
            % ISREFUSAL  True for a receiver's reply that refused the signal it answers.
            tf = obj.Type.Role == "reply" && isstruct(obj.Value) && isfield(obj.Value, 'Refused') && obj.Value.Refused;
        end
    end

    methods (Static)
        function s = make(type, value)
            % MAKE  An unstamped signal, ready to be sent through a Port.
            s = OceanAgents.Minds.Signals.Signal(type, value);
        end

        function s = refusal(reason)
            % REFUSAL  The reply of a receiver that refused a signal; carries the reason and changes nothing.
            s = OceanAgents.Minds.Signals.Signal(OceanAgents.Minds.Signals.SignalType.reply(), ...
                struct('Refused', true, 'Reason', string(reason)));
        end

        function s = ok(payload)
            % OK  An accepting reply, optionally with a payload record.
            if nargin < 1, payload = struct(); end
            payload.Refused = false;
            s = OceanAgents.Minds.Signals.Signal(OceanAgents.Minds.Signals.SignalType.reply(), payload);
        end
    end
end
