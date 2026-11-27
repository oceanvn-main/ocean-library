classdef ReturnLogger < OceanStats.RL.TD.Observers.IObserver
    % RETURNLOGGER  Logs, per episode, the undiscounted return, the length and whether it ended by termination or
    % truncation. A minimal example of a pluggable observer (one file, no edit to the learner):
    %   log = OceanStats.RL.TD.Observers.ReturnLogger();
    %   TDLearner(..., 'Observer', log).run();   log.Returns, log.Lengths, log.Truncated
    properties (SetAccess = private)
        Returns (:,1) double = zeros(0, 1)
        Lengths (:,1) double = zeros(0, 1)
        Truncated (:,1) logical = false(0, 1)
    end
    properties (Access = private)
        Ret_ (1,1) double = 0
        Len_ (1,1) double = 0
    end
    methods
        function startRun(obj), obj.Returns = zeros(0, 1);  obj.Lengths = zeros(0, 1);  obj.Truncated = false(0, 1); end
        function begin(obj, ~, ~), obj.Ret_ = 0;  obj.Len_ = 0; end
        function step(obj, r, ~, ~, ~), obj.Ret_ = obj.Ret_ + r;  obj.Len_ = obj.Len_ + 1; end
        function finish(obj, ~, truncated)
            obj.Returns(end+1, 1) = obj.Ret_;  obj.Lengths(end+1, 1) = obj.Len_;  obj.Truncated(end+1, 1) = truncated;
        end
        function finishArrays(obj, ~, ~, R, ~, truncated, ~)
            obj.Returns(end+1, 1) = sum(R);  obj.Lengths(end+1, 1) = numel(R);  obj.Truncated(end+1, 1) = truncated;
        end
    end
end
