classdef CompositeObserver < OceanStats.RL.Dyna.Observers.IObserver
    % COMPOSITEOBSERVER Broadcasts events in declaration order; failures are fail-stop.
    properties (SetAccess = immutable)
        Observers cell
    end
    methods
        function obj=CompositeObserver(observers)
            arguments, observers cell, end
            if ~all(cellfun(@(x) isa(x,'OceanStats.RL.Dyna.Observers.IObserver'),observers))
                error('OceanStats:RL:Dyna:Observers:BadObserver','Every item must implement IObserver.');
            end
            obj.Observers=observers;
        end
        function startRun(obj), obj.each_('startRun'); end
        function realTransition(obj,e), obj.each_('realTransition',e); end
        function modelUpdated(obj,e), obj.each_('modelUpdated',e); end
        function valueUpdated(obj,e), obj.each_('valueUpdated',e); end
        function finishEpisode(obj,e), obj.each_('finishEpisode',e); end
        function finishRun(obj,e), obj.each_('finishRun',e); end
    end
    methods (Access=private)
        function each_(obj,name,varargin)
            for k=1:numel(obj.Observers), obj.Observers{k}.(name)(varargin{:}); end
        end
    end
end
