classdef (Abstract) IDevice < OceanAgents.Components.IComponent
    % IDEVICE  A component on the world side: the world's rules live in devices (design D17; DV1 contract §3). Like
    % every component it verifies, then acts, and it reaches other parts only through the port the world gives it.

    properties (Access = protected)
        Port_ = []
    end

    methods
        function connect(obj, port)
            % CONNECT  Called once by the world that wires this device.
            if ~isempty(obj.Port_)
                error('OceanAgents:Signals:Sealed', 'Device "%s" is already connected to a world.', obj.ID);
            end
            obj.Port_ = port;
        end
    end

    methods (Static)
        function t = commandRecord()
            persistent cached
            if isempty(cached), cached = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "command"); end
            t = cached;
        end
    end

    methods (Access = protected)
        function v = ask_(obj, route, payload)
            % ASK_  Send a record on route and return the reply's value (throws on a refusal).
            if isempty(obj.Port_)
                error('OceanAgents:Embodied:Incompatible', 'Device "%s" is not connected to a world.', obj.ID);
            end
            [~, role] = obj.routeRole_(route);
            if role == "query", t = OceanAgents.Minds.Signals.SignalType.query();
            else, t = OceanAgents.Devices.IDevice.commandRecord(); end
            r = obj.Port_.send(route, OceanAgents.Minds.Signals.Signal.make(t, payload));
            if isempty(r) || r(1).isRefusal()
                why = "no reply";
                if ~isempty(r), why = r(1).Value.Reason; end
                error('OceanAgents:Devices:Refused', 'Device "%s": "%s" was refused: %s', obj.ID, route, why);
            end
            v = r(1).Value;
        end

        function [route, role] = routeRole_(~, route)
            role = "command";
            if any(route == ["where" "ask"]), role = "query"; end
        end
    end
end
