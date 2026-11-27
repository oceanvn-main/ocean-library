classdef RepeatMove < OceanAgents.Body.IBehavior
    % REPEATMOVE  Repeat one move a fixed number of times (the simplest multi-step behavior).
    %   b = OceanAgents.Body.RepeatMove("right", 3, 'Interruptible', false);

    properties (SetAccess = private)
        Move string
        Remaining double
    end

    methods
        function obj = RepeatMove(move, times, options)
            arguments
                move (1,1) string
                times (1,1) double {mustBeInteger, mustBePositive}
                options.Interruptible (1,1) logical = true
            end
            obj.ID = "repeat:" + move;
            obj.Move = move;
            obj.Remaining = times;
            obj.Interruptible = options.Interruptible;
        end

        function label = peek(obj)
            label = "";
            if obj.Remaining > 0, label = obj.Move; end
        end

        function tf = isDone(obj)
            tf = obj.Remaining <= 0;
        end

        function label = next(obj)
            if obj.Remaining <= 0
                label = "";
                return
            end
            obj.Remaining = obj.Remaining - 1;
            label = obj.Move;
        end
    end
end
