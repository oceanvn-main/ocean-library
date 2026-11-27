classdef FailingCommitLog < OceanAgents.Time.CommitLog
    % FAILINGCOMMITLOG  Test fixture: a history sink whose append fails from the FailFrom-th call on.
    properties
        FailFrom double = Inf
    end
    properties (Access = private)
        Calls_ double = 0
    end
    methods
        function append(obj, ctx, state)
            obj.Calls_ = obj.Calls_ + 1;
            if obj.Calls_ >= obj.FailFrom
                error('Test:DiskFull', 'recording failed');
            end
            append@OceanAgents.Time.CommitLog(obj, ctx, state);
        end
    end
end
