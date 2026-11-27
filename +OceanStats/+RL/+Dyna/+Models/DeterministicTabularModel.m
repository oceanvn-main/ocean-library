classdef DeterministicTabularModel < OceanStats.RL.Dyna.Models.IWorldModel
    % DETERMINISTICTABULARMODEL Latest-outcome tabular model for Dyna-Q.

    properties (Access = private)
        Observed (:,:) logical
        NextState (:,:) double
        Reward (:,:) double
        Terminated (:,:) logical
        ObservationCount (:,:) uint64
        PairIDs (:,1) double
        PredecessorPairIDs cell
    end
    properties (SetAccess = immutable)
        IncludeUnseenActions (1,1) logical
    end

    methods
        function obj = DeterministicTabularModel(numStates, numActions, validActionMask, options)
            arguments
                numStates
                numActions
                validActionMask
                options.IncludeUnseenActions (1,1) logical = false
            end
            obj@OceanStats.RL.Dyna.Models.IWorldModel(numStates, numActions, validActionMask);
            obj.IncludeUnseenActions = options.IncludeUnseenActions;
            obj.initialize_();
        end

        function event = update(obj, transition)
            t = obj.readTransition_(transition);
            obj.validatePair(t.StateIndex, t.ActionIndex, true);
            if t.NextStateIndex > obj.NumStates
                error('OceanStats:RL:Dyna:Models:BadTransition', 'NextStateIndex exceeds NumStates.');
            end
            id = sub2ind([obj.NumStates, obj.NumActions], t.StateIndex, t.ActionIndex);
            if obj.IncludeUnseenActions
                obj.addDefaultActions_(t.StateIndex);
            end
            wasObserved = obj.Observed(id);
            oldNext = obj.NextState(id);
            oldReward = obj.Reward(id);
            oldTerminated = obj.Terminated(id);
            removed = zeros(0,1); added = zeros(0,1);
            if wasObserved && oldNext > 0 && oldNext ~= t.NextStateIndex
                obj.removePredecessor_(oldNext, id);
                removed = oldNext;
            end
            if ~wasObserved
                obj.PairIDs(end + 1, 1) = id;
            end
            obj.Observed(id) = true;
            obj.NextState(id) = t.NextStateIndex;
            obj.Reward(id) = t.Reward;
            obj.Terminated(id) = t.Terminated;
            obj.ObservationCount(id) = obj.ObservationCount(id) + 1;
            if t.NextStateIndex > 0
                obj.addPredecessor_(t.NextStateIndex, id);
                if ~wasObserved || oldNext ~= t.NextStateIndex, added = t.NextStateIndex; end
            end
            obj.Version = obj.Version + 1;
            previous = repmat(obj.outcomeStruct_(0,0,false,0),0,1);
            if wasObserved, previous = obj.outcomeStruct_(oldNext,oldReward,oldTerminated,1); end
            event = OceanStats.RL.Dyna.Core.ModelEvent( ...
                'StateIndex',t.StateIndex,'ActionIndex',t.ActionIndex, ...
                'ObservationCount',double(obj.ObservationCount(id)), ...
                'PreviousOutcomes',previous,'NewOutcomes',obj.distribution(t.StateIndex,t.ActionIndex), ...
                'PredecessorChanges',struct('AddedNextStates',added,'RemovedNextStates',removed), ...
                'ModelVersion',double(obj.Version));
        end

        function tf = contains(obj, stateIndex, actionIndex)
            obj.validatePair(stateIndex, actionIndex, false);
            tf = obj.Observed(stateIndex, actionIndex);
        end

        function pairs = observedPairs(obj)
            pairs = obj.PairIDs;
        end

        function outcome = sample(obj, stateIndex, actionIndex, stream)
            arguments
                obj
                stateIndex (1,1) double {mustBeInteger,mustBePositive}
                actionIndex (1,1) double {mustBeInteger,mustBePositive}
                stream (1,1) RandStream
            end
            if isempty(stream)
                error('OceanStats:RL:Dyna:Models:BadStream', 'An owned stream is required.');
            end
            outcome = obj.distribution(stateIndex, actionIndex);
        end

        function outcomes = distribution(obj, stateIndex, actionIndex)
            obj.requireObserved_(stateIndex, actionIndex);
            id = sub2ind([obj.NumStates, obj.NumActions], stateIndex, actionIndex);
            outcomes = obj.outcomeStruct_(obj.NextState(id), obj.Reward(id), obj.Terminated(id), 1);
        end

        function pred = predecessors(obj, nextStateIndex)
            obj.validateState_(nextStateIndex);
            ids = obj.PredecessorPairIDs{nextStateIndex};
            pred = obj.predecessorRecords_(ids);
        end

        function snapshot = inspect(obj, stateIndex, actionIndex)
            obj.validatePair(stateIndex, actionIndex, false);
            id = sub2ind([obj.NumStates, obj.NumActions], stateIndex, actionIndex);
            outcomes = repmat(obj.outcomeStruct_(0, 0, false, 0), 0, 1);
            if obj.Observed(id), outcomes = obj.distribution(stateIndex, actionIndex); end
            snapshot = struct('StateIndex', stateIndex, 'ActionIndex', actionIndex, ...
                'PairID', id, 'Observed', obj.Observed(id), ...
                'ObservationCount', double(obj.ObservationCount(id)), ...
                'Outcomes', outcomes, 'Version', obj.Version);
        end

        function reset(obj)
            obj.initialize_();
            obj.Version = uint64(0);
        end
    end

    methods (Access = private)
        function initialize_(obj)
            dims = [obj.NumStates, obj.NumActions];
            obj.Observed = false(dims); obj.NextState = zeros(dims);
            obj.Reward = zeros(dims); obj.Terminated = false(dims);
            obj.ObservationCount = zeros(dims, 'uint64');
            obj.PairIDs = zeros(0, 1);
            obj.PredecessorPairIDs = cell(obj.NumStates, 1);
        end

        function requireObserved_(obj, s, a)
            obj.validatePair(s, a, true);
            if ~obj.Observed(s, a)
                error('OceanStats:RL:Dyna:Models:UnobservedPair', ...
                    'State/action pair (%d,%d) has not been observed.', s, a);
            end
        end

        function addPredecessor_(obj, nextState, id)
            ids = obj.PredecessorPairIDs{nextState};
            if ~any(ids == id), obj.PredecessorPairIDs{nextState}(end + 1, 1) = id; end
        end

        function removePredecessor_(obj, nextState, id)
            ids = obj.PredecessorPairIDs{nextState};
            obj.PredecessorPairIDs{nextState} = ids(ids ~= id);
        end

        function addDefaultActions_(obj, stateIndex)
            actions = find(obj.ValidActionMask(stateIndex, :));
            for actionIndex = actions
                id = sub2ind([obj.NumStates, obj.NumActions], stateIndex, actionIndex);
                if obj.Observed(id), continue; end
                obj.Observed(id) = true;
                obj.NextState(id) = stateIndex;
                obj.Reward(id) = 0;
                obj.Terminated(id) = false;
                obj.PairIDs(end + 1, 1) = id;
                obj.addPredecessor_(stateIndex, id);
            end
        end

        function records = predecessorRecords_(obj, ids)
            records = repmat(struct('StateIndex', 0, 'ActionIndex', 0, 'PairID', 0, ...
                'Probability', 0, 'MeanReward', 0), numel(ids), 1);
            for k = 1:numel(ids)
                [s, a] = ind2sub([obj.NumStates, obj.NumActions], ids(k));
                records(k) = struct('StateIndex', s, 'ActionIndex', a, 'PairID', ids(k), ...
                    'Probability', 1, 'MeanReward', obj.Reward(ids(k)));
            end
        end

        function validateState_(obj, s)
            if ~(isnumeric(s) && isscalar(s) && isfinite(s) && s == fix(s) && s >= 1 && s <= obj.NumStates)
                error('OceanStats:RL:Dyna:Models:StateOutOfRange', 'Next-state index is out of range.');
            end
        end
    end

    methods (Static, Access = private)
        function t = readTransition_(transition)
            names = {'StateIndex','ActionIndex','Reward','NextStateIndex','Terminated'};
            for k = 1:numel(names)
                if ~(isstruct(transition) && isfield(transition, names{k})) && ...
                        ~(isobject(transition) && isprop(transition, names{k}))
                    error('OceanStats:RL:Dyna:Models:BadTransition', 'Transition lacks field %s.', names{k});
                end
            end
            t = struct();
            for k = 1:numel(names), t.(names{k}) = transition.(names{k}); end
            if ~(isscalar(t.Reward) && isfinite(t.Reward)) || ...
                    ~(isscalar(t.NextStateIndex) && t.NextStateIndex >= 0 && t.NextStateIndex == fix(t.NextStateIndex)) || ...
                    ~(isscalar(t.Terminated) && (islogical(t.Terminated) || isnumeric(t.Terminated)))
                error('OceanStats:RL:Dyna:Models:BadTransition', 'Transition fields have invalid values.');
            end
            t.Terminated = logical(t.Terminated);
            if ~t.Terminated && t.NextStateIndex == 0
                error('OceanStats:RL:Dyna:Models:BadTransition', 'Only terminal transitions may use NextStateIndex zero.');
            end
        end

        function outcome = outcomeStruct_(nextState, reward, terminated, probability)
            outcome = struct('NextStateIndex', nextState, 'Reward', reward, ...
                'Terminated', logical(terminated), 'Probability', probability);
        end
    end
end
