classdef (Abstract) IBehavior < handle
    % IBEHAVIOR  A routine a body runs once the mind starts it with a signal: one primitive command per step until it
    % is done or stopped (design D6; AB1 section 6). Whether it may be stopped part-way is declared, so committed and
    % interruptible execution can be compared. A behavior never starts itself.

    properties (SetAccess = protected)
        ID string = "behavior"
        Interruptible logical = true
    end

    methods (Abstract)
        label = next(obj)          % the next primitive command label, or "" when done
    end

    methods
        function label = peek(~)
            % PEEK  The next command label without using it up ("" when unknown or done).
            label = "";
        end

        function tf = isDone(~)
            % ISDONE  True once the behavior has nothing left to do (it then no longer blocks the body).
            tf = false;
        end

        function stop(obj)
            if ~obj.Interruptible
                error('OceanAgents:Body:NotInterruptible', 'Behavior "%s" runs to completion and cannot be stopped.', obj.ID);
            end
        end
    end
end
