classdef Recorder < OceanStats.RL.Dyna.Observers.IObserver
    % RECORDER Inspectable causal event log for Dyna runs.
    properties (SetAccess = private)
        Transitions cell = {}
        ModelEvents cell = {}
        UpdateEvents cell = {}
        Episodes struct = struct('Return', {}, 'Length', {}, 'Terminated', {}, 'Truncated', {})
        Result struct = struct()
    end
    methods
        function startRun(obj)
            obj.Transitions={}; obj.ModelEvents={}; obj.UpdateEvents={};
            obj.Episodes=struct('Return',{},'Length',{},'Terminated',{},'Truncated',{}); obj.Result=struct();
        end
        function realTransition(obj,e), obj.Transitions{end+1,1}=e; end
        function modelUpdated(obj,e), obj.ModelEvents{end+1,1}=e; end
        function valueUpdated(obj,e), obj.UpdateEvents{end+1,1}=e; end
        function finishEpisode(obj,e), obj.Episodes(end+1,1)=e; end
        function finishRun(obj,r), obj.Result=r; end
    end
end
