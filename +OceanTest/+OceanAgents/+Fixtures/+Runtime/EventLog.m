classdef EventLog < handle
    % EVENTLOG  Test fixture: an ordered list of events ("decide:alice", "env:step", "update:step:alice", ...)
    % shared on purpose between fixture components of ONE test to check call ordering.
    properties (SetAccess = private)
        Events string = strings(1, 0)
    end
    methods
        function add(obj, event)
            obj.Events(end + 1) = string(event);
        end
        function k = indexOf(obj, event)
            % INDEXOF  Positions of an event in the log.
            k = find(obj.Events == string(event));
        end
    end
end
