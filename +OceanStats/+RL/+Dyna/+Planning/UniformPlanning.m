classdef UniformPlanning < OceanStats.RL.Dyna.Planning.IPlanningStrategy
    % UNIFORMPLANNING Uniform O(1) sampling from a compact eligible-pair list.
    properties (Access = private)
        PairIDs (:,1) double = zeros(0, 1)
        PairPresent (:,1) logical = false(0, 1)
        ModelVersion (1,1) uint64 = uint64(0)
        SelectionCount (1,1) uint64 = uint64(0)
    end
    methods
        function observeReal(obj, ~, model, ~, ~), obj.synchronize_(model); end
        function tf = canPlan(obj, model)
            obj.synchronize_(model); tf = ~isempty(obj.PairIDs);
        end
        function candidate = next(obj, model, values, gamma, stream) %#ok<INUSD>
            if ~(isa(stream, 'RandStream') && isscalar(stream))
                error('OceanStats:RL:Dyna:Planning:BadStream', 'Planning requires one owned RandStream.');
            end
            obj.synchronize_(model);
            if isempty(obj.PairIDs)
                error('OceanStats:RL:Dyna:Planning:NoCandidate', 'No modeled pair is eligible for planning.');
            end
            position = randi(stream, numel(obj.PairIDs));
            id = obj.PairIDs(position);
            [s, a] = ind2sub([model.NumStates, model.NumActions], id);
            obj.SelectionCount = obj.SelectionCount + 1;
            candidate = obj.makeCandidate_(s, a, id, 0, model.Version);
        end
        function afterUpdate(~, ~, ~, ~, ~), end
        function reset(obj)
            obj.PairIDs = zeros(0, 1); obj.PairPresent = false(0, 1);
            obj.ModelVersion = uint64(0); obj.SelectionCount = uint64(0);
        end
        function stats = inspect(obj)
            stats = struct('Name', "uniform", 'EligiblePairIDs', obj.PairIDs, ...
                'NumEligiblePairs', numel(obj.PairIDs), 'SelectionCount', double(obj.SelectionCount), ...
                'ModelVersion', obj.ModelVersion);
        end
    end
    methods (Access = private)
        function synchronize_(obj, model)
            if obj.ModelVersion == model.Version && numel(obj.PairPresent) == model.NumStates * model.NumActions, return; end
            raw = model.observedPairs();
            if isempty(raw)
                ids = zeros(0, 1);
            elseif size(raw, 2) == 2
                ids = sub2ind([model.NumStates, model.NumActions], raw(:, 1), raw(:, 2));
            else
                ids = raw(:);
            end
            ids = unique(ids, 'stable');
            if any(ids < 1 | ids > model.NumStates * model.NumActions | ids ~= fix(ids))
                error('OceanStats:RL:Dyna:Planning:BadPairID', 'The model returned an invalid pair ID.');
            end
            obj.PairIDs = ids;
            obj.PairPresent = false(model.NumStates * model.NumActions, 1);
            obj.PairPresent(ids) = true;
            obj.ModelVersion = model.Version;
        end
    end
    methods (Static, Access = private)
        function c = makeCandidate_(s, a, id, priority, version)
            c = struct('StateIndex', s, 'ActionIndex', a, 'PairID', id, ...
                'Priority', priority, 'ModelVersion', version);
        end
    end
end
