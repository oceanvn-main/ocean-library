classdef (Abstract) IBody < handle
    % IBODY  The agent's boundary with the world: it carries the agent's components, turns the mind's command signals
    % into the primitive action the world receives, and turns the world's view into sense signals (a transducer at the
    % boundary; design D5, D6; AB1 section 6). A body never decides; a behavior that needs a choice asks the mind.
    %
    %   sig = body.probe()                   % what the body is: the reply to the mind's detection signal
    %   replies = body.receive(command)      % from the router (or body.command(command))
    %   label = body.toAction()              % the primitive action for this step
    %   body.observeAll(view)                % its sensors send their observations to the mind (AB2)

    properties (SetAccess = protected)
        ID string = "body"
        Signature = OceanAgents.Body.BodySignature()
        Sensors = {}                             % OceanAgents.Body.Sensors.IObservation, the body's observation parts
        Parts = {}                               % other components the body carries (e.g. its GridPosition, DV1)
    end

    properties (Dependent)
        Accepts
    end

    properties (Access = private)
        Mind_ = []                               % the one mind this body is paired with (critic AB1 finding 8)
        Port_ = []                               % the body's port on its mind's router (set at join)
        Routed_ string = strings(1, 0)           % IDs of the sensors the mind perceives (sent on "sense:<id>")
    end

    methods (Abstract)
        replies = command(obj, signal)
        label = toAction(obj)
    end

    methods
        function a = get.Accepts(obj)
            a = obj.Signature.Commands;
        end

        function sig = probe(obj)
            sig = obj.Signature;
        end

        function out = receive(obj, signal)
            if signal.Type.Role ~= "command"
                out = OceanAgents.Minds.Signals.Signal.refusal("a body takes commands only");
                return
            end
            out = obj.command(signal);
        end

        function beginEpisode(~)
        end

        function tf = isPaired(obj)
            tf = ~isempty(obj.Mind_);
        end

        function observeAll(obj, view)
            % OBSERVEALL  Every sensor the mind perceives reads the view (or the body) and sends its observation to
            % the mind through the router (AB2 section 5). Sensors the mind does not perceive are not read.
            if isempty(obj.Port_)
                error('OceanAgents:Embodied:Incompatible', 'Body "%s" is not joined to a mind.', obj.ID);
            end
            for k = 1:numel(obj.Sensors)
                sensor = obj.Sensors{k};
                if ~any(obj.Routed_ == sensor.ID), continue; end
                obj.Port_.send("sense:" + sensor.ID, sensor.observe(view, obj));
            end
        end

        function label = nextBehaviorMove(~)
            % NEXTBEHAVIORMOVE  The running behavior's next move, without using it up ("" if none).
            label = "";
        end

        function st = behaviorStatus(~)
            % BEHAVIORSTATUS  What the behavior sensor reads: no behavior by default.
            st = struct('Running', false, 'ID', "", 'Remaining', 0);
        end

        function c = part(obj, id)
            % PART  A component the body carries, by ID (e.g. "position"), for wiring it into a world.
            k = find(cellfun(@(x) x.ID == string(id), obj.Parts), 1);
            if isempty(k), error('OceanAgents:Body:UnknownCommand', 'Body "%s" carries no part "%s".', obj.ID, id); end
            c = obj.Parts{k};
        end

        function ids = sensorIDs(obj)
            ids = string(cellfun(@(x) x.ID, obj.Sensors, 'UniformOutput', false));
            if isempty(ids), ids = strings(1, 0); end
        end

        function t = sensorType(obj, id)
            k = find(obj.sensorIDs() == id, 1);
            t = obj.Sensors{k}.OutType;
        end
    end

    methods (Access = {?OceanAgents.Minds.SignalMind})
        function claim_(obj, mind)
            if ~isempty(obj.Mind_) && obj.Mind_ ~= mind
                error('OceanAgents:Embodied:Incompatible', 'Body "%s" already belongs to another mind.', obj.ID);
            end
            obj.Mind_ = mind;
        end

        function release_(obj)
            obj.Mind_ = [];
            obj.Port_ = [];
            obj.Routed_ = strings(1, 0);
        end

        function bindPort_(obj, port, routed)
            obj.Port_ = port;
            obj.Routed_ = reshape(string(routed), 1, []);
        end
    end

    methods (Static)
        function t = senseType()
            % SENSETYPE  The generic sense carried to the mind: a record with the observation and the action mask.
            persistent cached
            if isempty(cached), cached = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "sense"); end
            t = cached;
        end

        function t = moveType(labels)
            t = OceanAgents.Minds.Signals.SignalType('ValueKind', "label", 'Labels', labels, 'Role', "command");
        end
    end

    methods (Access = protected)
        function sensors = checkSensors_(obj, sensors)
            % CHECKSENSORS_  Validate a sensor list (constructor helper): IObservation parts with unique IDs.
            if ~iscell(sensors), sensors = num2cell(sensors); end
            sensors = reshape(sensors, 1, []);
            if ~all(cellfun(@(x) isa(x, 'OceanAgents.Body.Sensors.IObservation') && isscalar(x), sensors))
                error('OceanAgents:Signals:BadType', 'Body "%s": every sensor must be an IObservation.', obj.ID);
            end
            ids = string(cellfun(@(x) x.ID, sensors, 'UniformOutput', false));
            if numel(unique(ids)) ~= numel(ids)
                error('OceanAgents:Signals:BadType', 'Body "%s": sensor IDs must be unique.', obj.ID);
            end
        end

        function [senses, parts] = sensorDeclarations_(~, sensors)
            senses = OceanAgents.Minds.Signals.SignalType.empty(1, 0);
            parts = strings(1, 0);
            for k = 1:numel(sensors)
                senses(end + 1) = sensors{k}.OutType; %#ok<AGROW>
                parts(end + 1) = sensors{k}.ID + ":" + class(sensors{k}); %#ok<AGROW>
            end
        end

        function noAction_(obj)
            error('OceanAgents:Body:NoAction', 'Body "%s" has no action for this step (no command was given).', obj.ID);
        end
    end
end
