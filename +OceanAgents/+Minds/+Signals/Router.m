classdef (Sealed) Router < handle
    % ROUTER  One per agent. Knows the routes; the kind of signal on each route is decided by the sender, negotiated
    % once at join, and fixed until the router is resealed (design D4; AB1_Signal_Body_Contract.md section 3.5).
    % It delivers synchronously, stamps every signal with the sender's identity, never alters content (only
    % converters do), and stops a run when one sender floods a single instant (chatter limit).
    %
    %   r = OceanAgents.Minds.Signals.Router();
    %   minds = r.attach(mind, "mind");   r.attach(motor, "motor");
    %   r.addRoute("motor", "motor", "command");
    %   r.negotiate("motor", moveType);    % "direct" | "converter" | throws TypeMismatch
    %   r.seal();
    %   replies = minds.send("motor", OceanAgents.Minds.Signals.Signal.make(moveType, "left"));
    %
    % A receiver is any object with a receive(signal) method returning a Signal row and an Accepts property
    % (SignalType array). A converter additionally has InType and OutType.

    properties (SetAccess = private)
        ID double = 0                           % unique per router in this MATLAB session (stamped as Signal.Via)
        Now double = 0
        Sealed (1,1) logical = false
        MaxEventsPerInstant double = 1000
        Seq double = 0
    end

    properties (Access = private)
        Parts_                                  % containers.Map: id -> receiver
        Routes_ = struct('Name', {}, 'Receiver', {}, 'Role', {}, 'Offered', {}, 'Mode', {}, 'Converter', {})
        Converters_ = cell(0, 2)                % {id, converter} pairs, in registration order
        Owner_ = []                             % the mind this router serves (a handle, set only by that mind)
        ChatterTime_ = NaN
        ChatterCount_                           % containers.Map: sender -> signals at ChatterTime_
    end

    methods
        function obj = Router(options)
            arguments
                options.MaxEventsPerInstant (1,1) double {mustBePositive, mustBeInteger} = 1000
            end
            obj.MaxEventsPerInstant = options.MaxEventsPerInstant;
            persistent counter
            if isempty(counter), counter = 0; end
            counter = counter + 1;
            obj.ID = counter;
            obj.Parts_ = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.ChatterCount_ = containers.Map('KeyType', 'char', 'ValueType', 'double');
        end

        function port = attach(obj, part, id)
            % ATTACH  Register a part under id and return the Port that sends on its behalf.
            arguments
                obj
                part
                id (1,1) string {mustBeNonzeroLengthText}
            end
            obj.requireOpen_('attach');
            if isKey(obj.Parts_, char(id))
                error('OceanAgents:Signals:BadType', 'Router: a part is already attached as "%s".', id);
            end
            if ~ismethod(part, 'receive')
                error('OceanAgents:Signals:BadType', 'Router: part "%s" has no receive(signal) method.', id);
            end
            if any(string(obj.Converters_(:, 1)) == id)
                error('OceanAgents:Signals:BadType', 'Router: "%s" is already the id of a converter.', id);
            end
            if isa(part, 'OceanAgents.Memory.IMemory'), part.joinRouter_(obj.ID, obj.Owner_, id); end   % one agent only
            if isa(part, 'OceanAgents.Components.GridPosition'), part.joinWorld_(obj.ID); end           % one world only
            obj.Parts_(char(id)) = part;
            port = OceanAgents.Minds.Signals.Port(obj, id);
        end

        function registerConverter(obj, converter, id)
            % REGISTERCONVERTER  Make a converter available to negotiation (chosen by type only, in this order).
            arguments
                obj
                converter
                id (1,1) string {mustBeNonzeroLengthText}
            end
            obj.requireOpen_('registerConverter');
            if isKey(obj.Parts_, char(id)) || (~isempty(obj.Converters_) && any(string(obj.Converters_(:, 1)) == id))
                error('OceanAgents:Signals:BadType', 'Router: "%s" is already used by a part or a converter.', id);
            end
            if ~(isprop(converter, 'InType') && isprop(converter, 'OutType') && ismethod(converter, 'receive'))
                error('OceanAgents:Signals:BadType', 'Router: "%s" is not a converter (InType, OutType, receive).', id);
            end
            obj.Converters_(end + 1, :) = {id, converter};
        end

        function addRoute(obj, name, receiverID, role)
            arguments
                obj
                name (1,1) string {mustBeNonzeroLengthText}
                receiverID (1,1) string {mustBeNonzeroLengthText}
                role (1,1) string {mustBeMember(role, ["command" "sense" "query" "reply" "modulation" "reward"])}
            end
            obj.requireOpen_('addRoute');
            if any(string({obj.Routes_.Name}) == name)
                error('OceanAgents:Signals:BadType', 'Router: route "%s" already exists.', name);
            end
            obj.Routes_(end + 1) = struct('Name', name, 'Receiver', receiverID, 'Role', role, 'Offered', [], ...
                'Mode', "", 'Converter', []);
        end

        function mode = negotiate(obj, route, offered)
            % NEGOTIATE  Settle the type a sender will use on route: "direct" when the receiver accepts it, else
            % "converter" when a registered converter bridges the two, else OceanAgents:Signals:TypeMismatch.
            arguments
                obj
                route (1,1) string
                offered (1,1) OceanAgents.Minds.Signals.SignalType
            end
            obj.requireOpen_('negotiate');
            k = obj.routeIndex_(route);
            r = obj.Routes_(k);
            if offered.Role ~= r.Role
                error('OceanAgents:Signals:TypeMismatch', 'Route "%s" carries role "%s"; offered "%s".', ...
                    route, r.Role, offered.Role);
            end
            accepts = obj.acceptsOf_(r.Receiver);
            conv = [];
            if any(arrayfun(@(a) a.accepts(offered), accepts))
                mode = "direct";
            else
                mode = "";
                for c = 1:size(obj.Converters_, 1)
                    cv = obj.Converters_{c, 2};
                    if cv.InType.accepts(offered) && any(arrayfun(@(a) a.accepts(cv.OutType), accepts))
                        mode = "converter";  conv = c;
                        break
                    end
                end
                if mode == ""
                    why = "no accepted type";
                    if ~isempty(accepts), why = accepts(1).mismatch(offered); end
                    error('OceanAgents:Signals:TypeMismatch', ...
                        'Route "%s": receiver "%s" accepts no %s, and no converter bridges it (first mismatch: %s).', ...
                        route, r.Receiver, offered.describe(), why);
                end
            end
            obj.Routes_(k).Offered = offered;
            obj.Routes_(k).Mode = mode;
            obj.Routes_(k).Converter = conv;
        end

        function seal(obj)
            % SEAL  Fix routes and negotiated types (join is complete). Further attach/addRoute/negotiate throw.
            obj.Sealed = true;
        end

        function advanceTo(obj, t)
            % ADVANCETO  Set the current instant used to stamp signals (never backwards).
            arguments
                obj
                t (1,1) double {mustBeFinite, mustBeNonnegative}
            end
            if t < obj.Now
                error('OceanAgents:Signals:BadValue', 'Router: time cannot move backwards (%g -> %g).', obj.Now, t);
            end
            obj.Now = t;
        end

        function m = modeOf(obj, route)
            m = obj.Routes_(obj.routeIndex_(route)).Mode;
        end

        function d = describe(obj)
            d = struct('Parts', string(keys(obj.Parts_)), 'Routes', string({obj.Routes_.Name}), 'Sealed', obj.Sealed);
        end
    end

    methods (Access = {?OceanAgents.Minds.SignalMind})
        function setOwner_(obj, mind)
            % SETOWNER_  Only a SignalMind can say "this router is mine", and only before any part is attached:
            % ownership is the mind object itself, never a number someone could pass in (critic AB1 round 3).
            obj.requireOpen_('setOwner_');
            if obj.Parts_.Count > 0
                error('OceanAgents:Signals:Sealed', 'Router: the owner is set before any part is attached.');
            end
            obj.Owner_ = mind;
        end
    end

    methods (Access = {?OceanAgents.Minds.Signals.Port})
        function replies = deliver_(obj, senderID, route, signal)
            if signal.isStamped()
                error('OceanAgents:Signals:ForgedSender', ...
                    'A signal sent by "%s" already claims sender "%s"; only the router stamps senders.', ...
                    senderID, signal.Sender);
            end
            k = obj.routeIndex_(route);
            if obj.Routes_(k).Mode == ""
                if obj.Sealed
                    error('OceanAgents:Signals:NoRoute', 'Route "%s" was never negotiated before sealing.', route);
                end
                obj.negotiate(route, signal.Type);
            end
            r = obj.Routes_(k);
            if ~r.Offered.accepts(signal.Type)
                error('OceanAgents:Signals:TypeMismatch', 'Route "%s" was negotiated for %s; got %s (%s).', ...
                    route, r.Offered.describe(), signal.Type.describe(), r.Offered.mismatch(signal.Type));
            end
            obj.countChatter_(senderID);
            s = obj.stamp_(signal, senderID, route, r.Receiver);
            receiver = obj.Parts_(char(r.Receiver));
            if r.Mode == "converter"
                cid = obj.Converters_{r.Converter, 1};
                outs = obj.Converters_{r.Converter, 2}.receive(s);
                replies = OceanAgents.Minds.Signals.Signal.empty(1, 0);
                for o = outs
                    if o.Type.Role == "reply", continue; end            % a converter's own refusal stops here
                    so = obj.stamp_(o, cid, route, r.Receiver);
                    replies = [replies, obj.stampReplies_(receiver.receive(so), r)]; %#ok<AGROW>
                end
                refusals = outs(arrayfun(@(o) o.Type.Role == "reply", outs));
                replies = [obj.stampReplies_(refusals, struct('Receiver', cid, 'Name', route)), replies];
            else
                replies = obj.stampReplies_(receiver.receive(s), r);
            end
        end
    end

    methods (Access = private)
        function requireOpen_(obj, what)
            if obj.Sealed
                error('OceanAgents:Signals:Sealed', 'Router: %s after seal (types are fixed once the join is done).', what);
            end
        end

        function k = routeIndex_(obj, route)
            k = find(string({obj.Routes_.Name}) == string(route), 1);
            if isempty(k), error('OceanAgents:Signals:NoRoute', 'No route "%s".', route); end
        end

        function a = acceptsOf_(obj, id)
            if ~isKey(obj.Parts_, char(id))
                error('OceanAgents:Signals:NoRoute', 'Receiver "%s" is not attached.', id);
            end
            p = obj.Parts_(char(id));
            if ~isprop(p, 'Accepts')
                error('OceanAgents:Signals:BadType', 'Receiver "%s" declares no Accepts.', id);
            end
            a = p.Accepts;
        end

        function n = nextSeq_(obj)
            obj.Seq = obj.Seq + 1;
            n = obj.Seq;
        end

        function countChatter_(obj, senderID)
            if ~isequal(obj.ChatterTime_, obj.Now)
                obj.ChatterTime_ = obj.Now;
                obj.ChatterCount_ = containers.Map('KeyType', 'char', 'ValueType', 'double');
            end
            key = char(senderID);
            n = 1;
            if isKey(obj.ChatterCount_, key), n = obj.ChatterCount_(key) + 1; end
            obj.ChatterCount_(key) = n;
            if n > obj.MaxEventsPerInstant
                error('OceanAgents:Signals:Chatter', ...
                    'Sender "%s" sent more than %d signals at time %g (chatter limit).', senderID, ...
                    obj.MaxEventsPerInstant, obj.Now);
            end
        end

        function s = stamp_(obj, s, sender, route, to)
            % STAMP_  Set the delivery coordinates (Signal allows only the Router to set them).
            s.Time = obj.Now;
            s.Sender = string(sender);
            s.To = string(to);
            s.Route = string(route);
            s.Seq = obj.nextSeq_();
            s.Via = obj.ID;
        end

        function out = stampReplies_(obj, replies, r)
            out = OceanAgents.Minds.Signals.Signal.empty(1, 0);
            if ~(isempty(replies) || isa(replies, 'OceanAgents.Minds.Signals.Signal'))
                error('OceanAgents:Signals:BadValue', 'Receiver "%s" returned a %s instead of Signals.', ...
                    r.Receiver, class(replies));
            end
            for x = reshape(replies, 1, [])
                out(end + 1) = obj.stamp_(x, r.Receiver, r.Name + "/reply", ""); %#ok<AGROW> always restamped: a
                                                                          % receiver cannot forge its reply's sender
            end
        end
    end
end
