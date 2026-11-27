classdef EmpiricalTabularModel < OceanStats.RL.Dyna.Models.IWorldModel
    % EMPIRICALTABULARMODEL Bounded sufficient-statistic stochastic model.

    properties (Access = private)
        OutcomeNext cell
        OutcomeTerminated cell
        OutcomeCount cell
        OutcomeRewardMean cell
        ObservationCount (:,:) uint64
        PairIDs (:,1) double
        PredecessorPairIDs cell
    end

    methods
        function mode = planningBackupMode(~), mode = "expectation"; end
        function obj = EmpiricalTabularModel(numStates, numActions, validActionMask)
            obj@OceanStats.RL.Dyna.Models.IWorldModel(numStates, numActions, validActionMask);
            obj.initialize_();
        end

        function event = update(obj, transition)
            t = obj.readTransition_(transition);
            obj.validatePair(t.StateIndex, t.ActionIndex, true);
            if t.NextStateIndex > obj.NumStates
                error('OceanStats:RL:Dyna:Models:BadTransition', 'NextStateIndex exceeds NumStates.');
            end
            id = sub2ind([obj.NumStates, obj.NumActions], t.StateIndex, t.ActionIndex);
            previous = obj.distributionIfObserved_(t.StateIndex, t.ActionIndex);
            predecessorWasPresent = t.NextStateIndex > 0 && any(obj.PredecessorPairIDs{t.NextStateIndex} == id);
            if obj.ObservationCount(id) == 0, obj.PairIDs(end + 1, 1) = id; end
            next = obj.OutcomeNext{id}; term = obj.OutcomeTerminated{id};
            k = find(next == t.NextStateIndex & term == t.Terminated, 1);
            if isempty(k)
                next(end + 1, 1) = t.NextStateIndex;
                term(end + 1, 1) = t.Terminated;
                count = obj.OutcomeCount{id}; count(end + 1, 1) = 0;
                meanReward = obj.OutcomeRewardMean{id}; meanReward(end + 1, 1) = 0;
                k = numel(next);
            else
                count = obj.OutcomeCount{id}; meanReward = obj.OutcomeRewardMean{id};
            end
            count(k) = count(k) + 1;
            meanReward(k) = meanReward(k) + (t.Reward - meanReward(k)) / count(k);
            obj.OutcomeNext{id} = next; obj.OutcomeTerminated{id} = term;
            obj.OutcomeCount{id} = count; obj.OutcomeRewardMean{id} = meanReward;
            obj.ObservationCount(id) = obj.ObservationCount(id) + 1;
            if t.NextStateIndex > 0
                ids = obj.PredecessorPairIDs{t.NextStateIndex};
                if ~any(ids == id), obj.PredecessorPairIDs{t.NextStateIndex}(end + 1, 1) = id; end
            end
            obj.Version = obj.Version + 1;
            added = zeros(0,1);
            if t.NextStateIndex > 0 && ~predecessorWasPresent, added=t.NextStateIndex; end
            event = OceanStats.RL.Dyna.Core.ModelEvent( ...
                'StateIndex',t.StateIndex,'ActionIndex',t.ActionIndex, ...
                'ObservationCount',double(obj.ObservationCount(id)), ...
                'PreviousOutcomes',previous,'NewOutcomes',obj.distribution(t.StateIndex,t.ActionIndex), ...
                'PredecessorChanges',struct('AddedNextStates',added,'RemovedNextStates',zeros(0,1)), ...
                'ModelVersion',double(obj.Version));
        end

        function tf = contains(obj, s, a)
            obj.validatePair(s, a, false); tf = obj.ObservationCount(s, a) > 0;
        end

        function pairs = observedPairs(obj), pairs = obj.PairIDs; end

        function outcome = sample(obj, s, a, stream)
            arguments
                obj
                s (1,1) double {mustBeInteger,mustBePositive}
                a (1,1) double {mustBeInteger,mustBePositive}
                stream (1,1) RandStream
            end
            outcomes = obj.distribution(s, a);
            cumulative = cumsum([outcomes.Probability]);
            k = find(rand(stream) <= cumulative, 1);
            outcome = outcomes(k);
        end

        function outcomes = distribution(obj, s, a)
            obj.validatePair(s, a, true);
            id = sub2ind([obj.NumStates, obj.NumActions], s, a);
            if obj.ObservationCount(id) == 0
                error('OceanStats:RL:Dyna:Models:UnobservedPair', 'State/action pair has not been observed.');
            end
            next = obj.OutcomeNext{id}; term = obj.OutcomeTerminated{id};
            count = obj.OutcomeCount{id}; rewards = obj.OutcomeRewardMean{id};
            probabilities = count / sum(count);
            outcomes = repmat(struct('NextStateIndex', 0, 'Reward', 0, ...
                'Terminated', false, 'Probability', 0), numel(next), 1);
            for k = 1:numel(next)
                outcomes(k) = struct('NextStateIndex', next(k), 'Reward', rewards(k), ...
                    'Terminated', term(k), 'Probability', probabilities(k));
            end
        end

        function pred = predecessors(obj, nextStateIndex)
            obj.validateState_(nextStateIndex);
            ids = obj.PredecessorPairIDs{nextStateIndex};
            pred = repmat(struct('StateIndex', 0, 'ActionIndex', 0, 'PairID', 0, ...
                'Probability', 0, 'MeanReward', 0), 0, 1);
            for j = 1:numel(ids)
                [s, a] = ind2sub([obj.NumStates, obj.NumActions], ids(j));
                d = obj.distribution(s, a);
                matches = [d.NextStateIndex] == nextStateIndex;
                if any(matches)
                    probability = sum([d(matches).Probability]);
                    meanReward = sum([d(matches).Probability] .* [d(matches).Reward]) / probability;
                    pred(end + 1, 1) = struct('StateIndex', s, 'ActionIndex', a, ...
                        'PairID', ids(j), 'Probability', probability, 'MeanReward', meanReward); %#ok<AGROW>
                end
            end
        end

        function snapshot = inspect(obj, s, a)
            obj.validatePair(s, a, false);
            id = sub2ind([obj.NumStates, obj.NumActions], s, a);
            d = obj.distributionIfObserved_(s, a);
            snapshot = struct('StateIndex', s, 'ActionIndex', a, 'PairID', id, ...
                'Observed', obj.ObservationCount(id) > 0, ...
                'ObservationCount', double(obj.ObservationCount(id)), ...
                'Outcomes', d, 'Version', obj.Version);
        end

        function reset(obj), obj.initialize_(); obj.Version = uint64(0); end
    end

    methods (Access = private)
        function initialize_(obj)
            dims = [obj.NumStates, obj.NumActions];
            obj.OutcomeNext = cell(dims); obj.OutcomeTerminated = cell(dims);
            obj.OutcomeCount = cell(dims); obj.OutcomeRewardMean = cell(dims);
            obj.ObservationCount = zeros(dims, 'uint64'); obj.PairIDs = zeros(0, 1);
            obj.PredecessorPairIDs = cell(obj.NumStates, 1);
        end
        function d = distributionIfObserved_(obj, s, a)
            id = sub2ind([obj.NumStates, obj.NumActions], s, a);
            d = repmat(struct('NextStateIndex', 0, 'Reward', 0, 'Terminated', false, 'Probability', 0), 0, 1);
            if obj.ObservationCount(id) > 0, d = obj.distribution(s, a); end
        end
        function validateState_(obj, s)
            if ~(isnumeric(s) && isscalar(s) && isfinite(s) && s == fix(s) && s >= 1 && s <= obj.NumStates)
                error('OceanStats:RL:Dyna:Models:StateOutOfRange', 'Next-state index is out of range.');
            end
        end
    end
    methods (Static, Access = private)
        function t = readTransition_(x)
            names = {'StateIndex','ActionIndex','Reward','NextStateIndex','Terminated'}; t = struct();
            for k = 1:numel(names)
                if isstruct(x) && isfield(x,names{k})
                    t.(names{k}) = x.(names{k});
                elseif isobject(x) && isprop(x,names{k})
                    t.(names{k}) = x.(names{k});
                else
                    error('OceanStats:RL:Dyna:Models:BadTransition','Transition lacks field %s.',names{k});
                end
            end
            if ~(isscalar(t.Reward) && isfinite(t.Reward) && isscalar(t.NextStateIndex) && ...
                    t.NextStateIndex >= 0 && t.NextStateIndex == fix(t.NextStateIndex) && isscalar(t.Terminated))
                error('OceanStats:RL:Dyna:Models:BadTransition','Transition fields have invalid values.');
            end
            t.Terminated = logical(t.Terminated);
            if ~t.Terminated && t.NextStateIndex == 0
                error('OceanStats:RL:Dyna:Models:BadTransition','Only terminal transitions may use NextStateIndex zero.');
            end
        end
    end
end
