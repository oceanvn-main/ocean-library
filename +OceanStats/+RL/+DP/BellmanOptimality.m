classdef BellmanOptimality
    % BELLMANOPTIMALITY Exact expected backups for indexed world models.
    methods (Static)
        function [value, action, qValues, distributions] = backup(model, stateIndex, values, discount, tieTolerance)
            arguments
                model (1,1) OceanStats.RL.Dyna.Models.IWorldModel
                stateIndex (1,1) double {mustBeInteger,mustBePositive}
                values (:,1) double {mustBeFinite}
                discount (1,1) double {mustBeInRange(discount,0,1)}
                tieTolerance (1,1) double {mustBeNonnegative} = 1e-9
            end
            if stateIndex > model.NumStates || numel(values) ~= model.NumStates
                error('OceanStats:RL:DP:BadStateOrValues', 'State or value-vector size is invalid.');
            end
            actions = find(model.ValidActionMask(stateIndex,:));
            if isempty(actions)
                value = values(stateIndex); action = 0;
                qValues = nan(1,model.NumActions); distributions = cell(1,model.NumActions);
                return
            end
            qValues = nan(1,model.NumActions);
            distributions = cell(1,model.NumActions);
            for a = actions
                d = model.distribution(stateIndex,a);
                qValues(a) = OceanStats.RL.DP.BellmanOptimality.actionValue(d,values,discount);
                distributions{a} = d;
            end
            value = max(qValues(actions));
            action = actions(find(qValues(actions) >= value-tieTolerance,1,'first'));
        end

        function q = actionValue(outcomes, values, discount)
            arguments
                outcomes struct
                values (:,1) double {mustBeFinite}
                discount (1,1) double {mustBeInRange(discount,0,1)}
            end
            if isempty(outcomes) || ~all(isfield(outcomes,{'Probability','Reward','NextStateIndex','Terminated'}))
                error('OceanStats:RL:DP:BadDistribution','Distribution is empty or lacks required fields.');
            end
            p = [outcomes.Probability]; r = [outcomes.Reward];
            n = [outcomes.NextStateIndex]; t = [outcomes.Terminated];
            if ~all(isfinite(p)) || any(p < 0) || abs(sum(p)-1) > 1e-9 || ...
                    ~all(isfinite(r)) || ~all(isfinite(n)) || any(n ~= fix(n)) || ...
                    any(n < 0 | n > numel(values)) || ...
                    ~(islogical(t) || all(t == 0 | t == 1)) || any(~logical(t) & n == 0)
                error('OceanStats:RL:DP:BadDistribution','Invalid probabilities, rewards, terminal flags, or successor indices.');
            end
            continuation = zeros(size(n));
            active = ~logical(t);
            continuation(active) = values(n(active));
            q = sum(p .* (r + discount .* continuation));
        end
    end
end
