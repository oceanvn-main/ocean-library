classdef (Sealed) Port < handle
    % PORT  A part's only way to send signals. Issued by Router.attach and bound to that part's identity: everything
    % sent through it is stamped with that identity by the router (trusted sender, AB1 section 3.4).
    %
    %   port = router.attach(motor, "motor");
    %   replies = port.send("sense", signal);

    properties (SetAccess = private)
        ID string = ""
    end

    properties (Access = private)
        Router_
    end

    methods (Access = {?OceanAgents.Minds.Signals.Router})
        function obj = Port(router, id)
            obj.Router_ = router;
            obj.ID = string(id);
        end
    end

    methods
        function replies = send(obj, route, signal)
            % SEND  Deliver signal on route; returns the receiver's replies (stamped, Sender = the receiver).
            arguments
                obj
                route (1,1) string
                signal (1,1) OceanAgents.Minds.Signals.Signal
            end
            replies = obj.Router_.deliver_(obj.ID, route, signal);
        end
    end
end
