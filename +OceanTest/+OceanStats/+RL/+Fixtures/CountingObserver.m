classdef CountingObserver < OceanStats.RL.TD.Observers.IObserver
    % COUNTINGOBSERVER  TEST FIXTURE: a brand-new observer written as ONE file, supplied through the 'Observer' option.
    properties (SetAccess = private)
        Steps = 0
        Episodes = 0
        Starts = 0
    end
    methods
        function startRun(obj), obj.Steps = 0;  obj.Episodes = 0;  obj.Starts = 0; end
        function begin(obj, ~, ~), obj.Starts = obj.Starts + 1; end
        function step(obj, ~, ~, ~, ~), obj.Steps = obj.Steps + 1; end
        function finish(obj, ~, ~), obj.Episodes = obj.Episodes + 1; end
        function finishArrays(obj, ~, ~, R, ~, ~, ~), obj.Steps = obj.Steps + numel(R);  obj.Episodes = obj.Episodes + 1;  obj.Starts = obj.Starts + 1; end
    end
end
