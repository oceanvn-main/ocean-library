classdef MaxPriorityQueue < handle
    % MAXPRIORITYQUEUE Binary max heap with deterministic pair-ID tie breaking.
    properties (Access = private)
        Priorities (:,1) double = zeros(0, 1)
        PairIDs (:,1) double = zeros(0, 1)
        Generations (:,1) uint64 = zeros(0, 1, 'uint64')
        CurrentGeneration (:,1) uint64 = zeros(0, 1, 'uint64')
        CurrentPriority (:,1) double = zeros(0, 1)
        Active (:,1) logical = false(0, 1)
        StaleDiscardCount (1,1) uint64 = uint64(0)
    end
    methods
        function push(obj, pairID, priority)
            arguments
                obj
                pairID (1,1) double {mustBeInteger,mustBePositive}
                priority (1,1) double {mustBeFinite,mustBeNonnegative}
            end
            obj.ensureCapacity_(pairID);
            obj.CurrentGeneration(pairID) = obj.CurrentGeneration(pairID) + 1;
            obj.CurrentPriority(pairID) = priority; obj.Active(pairID) = true;
            obj.Priorities(end + 1, 1) = priority; obj.PairIDs(end + 1, 1) = pairID;
            obj.Generations(end + 1, 1) = obj.CurrentGeneration(pairID);
            obj.siftUp_(numel(obj.PairIDs));
            if numel(obj.PairIDs) > max(64, 4 * nnz(obj.Active)), obj.compact_(); end
        end
        function [pairID, priority, generation] = pop(obj)
            obj.discardStale_();
            if isempty(obj.PairIDs)
                error('OceanStats:RL:Dyna:Planning:EmptyQueue', 'Cannot pop an empty priority queue.');
            end
            pairID = obj.PairIDs(1); priority = obj.Priorities(1); generation = obj.Generations(1);
            obj.removeRoot_(); obj.Active(pairID) = false;
        end
        function tf = isEmpty(obj), obj.discardStale_(); tf = isempty(obj.PairIDs); end
        function n = count(obj), n = nnz(obj.Active); end
        function reset(obj)
            obj.Priorities = zeros(0,1); obj.PairIDs = zeros(0,1);
            obj.Generations = zeros(0,1,'uint64'); obj.CurrentGeneration = zeros(0,1,'uint64');
            obj.CurrentPriority = zeros(0,1); obj.Active = false(0,1); obj.StaleDiscardCount = uint64(0);
        end
        function s = inspect(obj)
            obj.discardStale_();
            s = struct('PairIDs', obj.PairIDs, 'Priorities', obj.Priorities, ...
                'Generations', obj.Generations, 'ActiveCount', nnz(obj.Active), ...
                'HeapEntryCount', numel(obj.PairIDs), 'StaleDiscardCount', double(obj.StaleDiscardCount));
        end
    end
    methods (Access = private)
        function ensureCapacity_(obj, id)
            n = numel(obj.Active); if id <= n, return; end
            obj.Active(n+1:id,1) = false; obj.CurrentPriority(n+1:id,1) = 0;
            obj.CurrentGeneration(n+1:id,1) = uint64(0);
        end
        function discardStale_(obj)
            while ~isempty(obj.PairIDs)
                id = obj.PairIDs(1); gen = obj.Generations(1);
                if obj.Active(id) && gen == obj.CurrentGeneration(id) && obj.Priorities(1) == obj.CurrentPriority(id), return; end
                obj.removeRoot_(); obj.StaleDiscardCount = obj.StaleDiscardCount + 1;
            end
        end
        function removeRoot_(obj)
            last = numel(obj.PairIDs);
            if last == 1
                obj.Priorities = zeros(0,1); obj.PairIDs = zeros(0,1); obj.Generations = zeros(0,1,'uint64'); return
            end
            obj.Priorities(1)=obj.Priorities(last); obj.PairIDs(1)=obj.PairIDs(last); obj.Generations(1)=obj.Generations(last);
            obj.Priorities(last)=[]; obj.PairIDs(last)=[]; obj.Generations(last)=[]; obj.siftDown_(1);
        end
        function siftUp_(obj, k)
            while k > 1
                p = floor(k/2); if ~obj.higher_(k,p), break; end
                obj.swap_(k,p); k=p;
            end
        end
        function siftDown_(obj, k)
            n=numel(obj.PairIDs);
            while true
                l=2*k; if l>n, return; end
                r=l+1; best=l; if r<=n && obj.higher_(r,l), best=r; end
                if ~obj.higher_(best,k), return; end
                obj.swap_(k,best); k=best;
            end
        end
        function tf = higher_(obj,i,j)
            tf = obj.Priorities(i)>obj.Priorities(j) || ...
                (obj.Priorities(i)==obj.Priorities(j) && obj.PairIDs(i)<obj.PairIDs(j));
        end
        function swap_(obj,i,j)
            [obj.Priorities(i),obj.Priorities(j)]=deal(obj.Priorities(j),obj.Priorities(i));
            [obj.PairIDs(i),obj.PairIDs(j)]=deal(obj.PairIDs(j),obj.PairIDs(i));
            [obj.Generations(i),obj.Generations(j)]=deal(obj.Generations(j),obj.Generations(i));
        end
        function compact_(obj)
            ids=find(obj.Active); priorities=obj.CurrentPriority(ids); generations=obj.CurrentGeneration(ids);
            obj.PairIDs=zeros(0,1); obj.Priorities=zeros(0,1); obj.Generations=zeros(0,1,'uint64');
            for k=1:numel(ids)
                obj.PairIDs(end+1,1)=ids(k); obj.Priorities(end+1,1)=priorities(k); obj.Generations(end+1,1)=generations(k);
                obj.siftUp_(numel(obj.PairIDs));
            end
        end
    end
end
