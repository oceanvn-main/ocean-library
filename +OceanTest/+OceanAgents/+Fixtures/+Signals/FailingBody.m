classdef FailingBody < OceanAgents.Body.EmptyBody
    % FAILINGBODY  Test fixture: an empty body that breaks (throws) on command number FailAt, so a run fails part-way
    % and the learning gate can be checked (the failed step must never be learned).
    properties (SetAccess = private)
        FailAt double
        Commands double = 0
    end
    methods
        function obj = FailingBody(labels, failAt)
            obj@OceanAgents.Body.EmptyBody(labels);
            obj.FailAt = failAt;
        end
        function replies = command(obj, signal)
            obj.Commands = obj.Commands + 1;
            if obj.Commands >= obj.FailAt
                error('OceanTest:FailingBody:Broken', 'The body broke on command %d.', obj.Commands);
            end
            replies = command@OceanAgents.Body.EmptyBody(obj, signal);
        end
    end
end
