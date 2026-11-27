classdef GridRobotBody < OceanAgents.Body.IBody
    % GRIDROBOTBODY  A robot for grid worlds: a Motor component for its moves, sensors (digit, moves, behavior by default), plus
    % behaviors started by a command record (AB1 section 6). The moves it has are its capacity: a four-move robot and a
    % king-move robot differ exactly in the diagonals (and "stay"), which the mind sees through probe().
    %
    %   body = OceanAgents.Body.GridRobotBody(["1" "2" "3" "4"]);
    %   body.command(Signal.make(OceanAgents.Body.GridRobotBody.behaviorType(), ...
    %       struct('Behavior', "repeat", 'Move', "2", 'Times', 3, 'Interruptible', true)));
    %   body.command(Signal.make(OceanAgents.Body.GridRobotBody.behaviorType(), struct('Behavior', "stop")));

    properties (SetAccess = private)
        Motor
        Behavior = []
    end

    properties (Access = private)
        Pending_ string = strings(0, 1)
    end

    methods
        function obj = GridRobotBody(moves, options)
            arguments
                moves string
                options.ID (1,1) string = "body"
                options.Sensors = []
                options.Parts = {}
            end
            obj.ID = options.ID;
            moves = reshape(moves, 1, []);
            obj.Parts = reshape(options.Parts, 1, []);
            obj.Motor = OceanAgents.Components.Motor(moves);
            sensors = options.Sensors;
            if isempty(sensors)                       % default: digit, what the Laws permit, the running behavior
                sensors = {OceanAgents.Body.Sensors.DigitObservation(), OceanAgents.Body.Sensors.MovesObservation(moves), ...
                    OceanAgents.Body.Sensors.BehaviorObservation()};
            end
            obj.Sensors = obj.checkSensors_(sensors);
            [senses, parts] = obj.sensorDeclarations_(obj.Sensors);
            obj.Signature = OceanAgents.Body.BodySignature( ...
                [OceanAgents.Body.IBody.moveType(moves), OceanAgents.Body.GridRobotBody.behaviorType()], senses, ...
                ["motor:" + class(obj.Motor), parts, ...
                 string(cellfun(@(x) x.ID + ":" + class(x), obj.Parts, 'UniformOutput', false))]);
        end

        function replies = command(obj, signal)
            Sig = OceanAgents.Minds.Signals.Signal;
            if ~isempty(obj.Behavior) && obj.Behavior.isDone(), obj.Behavior = []; end
            if signal.Type.ValueKind == "record"
                replies = obj.behaviorCommand_(signal.Value);
                return
            end
            if ~isempty(obj.Behavior) && ~obj.Behavior.Interruptible
                replies = Sig.refusal(sprintf('busy: behavior "%s" runs to completion', obj.Behavior.ID));
                return
            end
            replies = obj.drive_(signal);
            if ~replies(1).isRefusal(), obj.Behavior = []; end   % an ACCEPTED move interrupts the behavior
        end

        function label = toAction(obj)
            if isempty(obj.Pending_) && ~isempty(obj.Behavior)
                next = obj.Behavior.next();
                if strlength(next) == 0
                    obj.Behavior = [];
                else
                    obj.drive_(OceanAgents.Minds.Signals.Signal.make(obj.Motor.Accepts, next));
                end
            end
            if isempty(obj.Pending_), obj.noAction_(); end
            label = obj.Pending_;
            obj.Pending_ = strings(0, 1);
        end

        function beginEpisode(obj)
            obj.Pending_ = strings(0, 1);
            obj.Behavior = [];
        end

        function label = nextBehaviorMove(obj)
            label = "";
            if ~isempty(obj.Behavior) && ~obj.Behavior.isDone(), label = obj.Behavior.peek(); end
        end

        function st = behaviorStatus(obj)
            if isempty(obj.Behavior) || obj.Behavior.isDone()
                st = struct('Running', false, 'ID', "", 'Remaining', 0);
            else
                rem = NaN;
                if isprop(obj.Behavior, 'Remaining'), rem = obj.Behavior.Remaining; end
                st = struct('Running', true, 'ID', obj.Behavior.ID, 'Remaining', rem);
            end
        end
    end

    methods (Static)
        function t = behaviorType()
            t = OceanAgents.Minds.Signals.SignalType('ValueKind', "record", 'Role', "command");
        end
    end

    methods (Access = private)
        function replies = drive_(obj, signal)
            out = obj.Motor.receive(signal);
            replies = out(arrayfun(@(o) o.isRefusal(), out));
            act = out(arrayfun(@(o) ~o.isRefusal(), out));
            if ~isempty(act)
                obj.Pending_ = act(1).Value;
                replies = OceanAgents.Minds.Signals.Signal.ok();
            end
        end

        function replies = behaviorCommand_(obj, v)
            Sig = OceanAgents.Minds.Signals.Signal;
            if ~isfield(v, 'Behavior')
                replies = Sig.refusal("a command record needs a Behavior field");
                return
            end
            switch string(v.Behavior)
                case "repeat"
                    if ~isempty(obj.Behavior) && ~obj.Behavior.Interruptible
                        replies = Sig.refusal(sprintf('busy: behavior "%s" runs to completion', obj.Behavior.ID));
                        return
                    end
                    if ~(isfield(v, 'Times') && isnumeric(v.Times) && isscalar(v.Times) && isfinite(v.Times) && ...
                            v.Times >= 1 && v.Times == fix(v.Times))
                        replies = Sig.refusal("repeat needs a positive integer Times");
                        return
                    end
                    if ~(isfield(v, 'Move') && any(obj.Motor.Moves == string(v.Move)))
                        replies = Sig.refusal("repeat needs one of this body's moves");
                        return
                    end
                    if isfield(v, 'Interruptible') && ~(isscalar(v.Interruptible) && (islogical(v.Interruptible) || ...
                            (isnumeric(v.Interruptible) && any(v.Interruptible == [0 1]))))
                        replies = Sig.refusal("Interruptible must be true or false");
                        return
                    end
                    interruptible = ~isfield(v, 'Interruptible') || logical(v.Interruptible);
                    obj.Behavior = OceanAgents.Body.RepeatMove(string(v.Move), v.Times, 'Interruptible', interruptible);
                    replies = Sig.ok();
                case "stop"
                    if ~isempty(obj.Behavior)
                        if ~obj.Behavior.Interruptible
                            replies = Sig.refusal(sprintf('NotInterruptible: behavior "%s" runs to completion', ...
                                obj.Behavior.ID));
                            return
                        end
                        obj.Behavior.stop();
                        obj.Behavior = [];
                    end
                    replies = Sig.ok();
                otherwise
                    replies = Sig.refusal(sprintf('unknown behavior "%s"', string(v.Behavior)));
            end
        end
    end
end
