classdef ReplayBuffer < handle
    % REPLAYBUFFER  Circular experience replay memory buffer for deep RL.
    %   Mnih et al. (2015): Stores transitions (S, A, R, S', Terminated) and samples random minibatches.
    properties (SetAccess = protected)
        Capacity (1,1) double {mustBeInteger, mustBePositive} = 10000
        StateDim (1,1) double {mustBeInteger, mustBePositive} = 1
        States double = []
        Actions double = []
        Rewards double = []
        NextStates double = []
        Terminated logical = []
        Pointer (1,1) double = 1
        Size (1,1) double = 0
    end
    methods
        function obj = ReplayBuffer(capacity, stateDim)
            arguments
                capacity (1,1) double {mustBeInteger, mustBePositive} = 10000
                stateDim (1,1) double {mustBeInteger, mustBePositive} = 1
            end
            obj.Capacity = capacity;
            obj.StateDim = stateDim;
            obj.States = zeros(capacity, stateDim);
            obj.Actions = zeros(capacity, 1);
            obj.Rewards = zeros(capacity, 1);
            obj.NextStates = zeros(capacity, stateDim);
            obj.Terminated = false(capacity, 1);
        end
        
        function store(obj, s, a, r, sNext, term)
            % Store a single transition (s, a, r, sNext, term)
            ptr = obj.Pointer;
            if isscalar(s)
                sVec = zeros(1, obj.StateDim);
                if s <= obj.StateDim, sVec(s) = 1; else, sVec(1) = s; end
                obj.States(ptr, :) = sVec;
            else
                obj.States(ptr, :) = s;
            end
            
            obj.Actions(ptr) = a;
            obj.Rewards(ptr) = r;
            
            if isscalar(sNext)
                sNextVec = zeros(1, obj.StateDim);
                if sNext <= obj.StateDim, sNextVec(sNext) = 1; else, sNextVec(1) = sNext; end
                obj.NextStates(ptr, :) = sNextVec;
            else
                obj.NextStates(ptr, :) = sNextVec;
            end
            
            obj.Terminated(ptr) = term;
            
            obj.Pointer = mod(obj.Pointer, obj.Capacity) + 1;
            obj.Size = min(obj.Size + 1, obj.Capacity);
        end
        
        function batch = sampleBatch(obj, batchSize, stream)
            % Sample a random minibatch of size batchSize
            if nargin < 3, stream = RandStream.getGlobalStream(); end
            idx = randi(stream, obj.Size, [batchSize, 1]);
            batch = struct();
            batch.States = obj.States(idx, :);
            batch.Actions = obj.Actions(idx);
            batch.Rewards = obj.Rewards(idx);
            batch.NextStates = obj.NextStates(idx, :);
            batch.Terminated = obj.Terminated(idx);
        end
    end
end
