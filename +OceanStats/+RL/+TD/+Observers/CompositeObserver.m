classdef CompositeObserver < OceanStats.RL.TD.Observers.IObserver
    % COMPOSITEOBSERVER  Fans every hook out to several observers, in order.
    properties (SetAccess = private)
        Observers cell = {}
    end
    methods
        function obj = CompositeObserver(observers)
            arguments
                observers cell
            end
            for k = 1:numel(observers)
                if ~isa(observers{k}, 'OceanStats.RL.TD.Observers.IObserver')
                    error('OceanStats:RL:TD:Observers:BadObserver', 'Observer %d is not an OceanStats.RL.TD.Observers.IObserver.', k);
                end
            end
            obj.Observers = observers;
        end
        function startRun(obj), for k = 1:numel(obj.Observers), obj.Observers{k}.startRun(); end, end
        function begin(obj, s0, a0), for k = 1:numel(obj.Observers), obj.Observers{k}.begin(s0, a0); end, end
        function step(obj, r, s2, a2, o), for k = 1:numel(obj.Observers), obj.Observers{k}.step(r, s2, a2, o); end, end
        function coin(obj, c), for k = 1:numel(obj.Observers), obj.Observers{k}.coin(c); end, end
        function action(obj, a), for k = 1:numel(obj.Observers), obj.Observers{k}.action(a); end, end
        function finish(obj, t, u), for k = 1:numel(obj.Observers), obj.Observers{k}.finish(t, u); end, end
        function finishArrays(obj, S, A, R, t, u, o), for k = 1:numel(obj.Observers), obj.Observers{k}.finishArrays(S, A, R, t, u, o); end, end
        function ep = episodes(obj)
            ep = [];
            for k = 1:numel(obj.Observers)
                e = obj.Observers{k}.episodes();
                if ~isempty(e), ep = e; return; end
            end
        end
    end
end
