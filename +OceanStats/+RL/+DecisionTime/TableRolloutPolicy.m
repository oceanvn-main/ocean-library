classdef TableRolloutPolicy < OceanAlgo.StateSearch.IRolloutPolicy
    % TABLEROLLOUTPOLICY  A fixed tabular RL policy used as the rollout policy of RolloutSearch / MCTS.
    %
    %   policy = OceanStats.RL.DecisionTime.TableRolloutPolicy(table)
    %     table  NumStates x 1 action indices             deterministic π(s) (e.g. a PolicyVector from value
    %                                                     iteration, RTDP or a learned greedy policy)
    %            NumStates x NumActions probabilities      stochastic π(a|s); every used row sums to 1
    %   a = policy.act(model, s, stream)                   model: WorldModelSearchModel (state = index)
    %
    % The chosen action must be legal in s (model.actions(s)); otherwise an error is raised.
    properties (SetAccess = immutable)
        Table double
    end
    methods
        function obj = TableRolloutPolicy(table)
            arguments
                table double {mustBeNonempty, mustBeFinite, mustBeNonnegative}
            end
            if size(table, 2) == 1
                if any(table ~= fix(table))
                    error('OceanStats:RL:DecisionTime:BadPolicyTable', 'A deterministic table holds integer action indices.');
                end
            elseif any(abs(sum(table, 2) - 1) > 1e-9 & any(table > 0, 2))
                error('OceanStats:RL:DecisionTime:BadPolicyTable', 'Each used row of a stochastic table must sum to 1.');
            end
            obj.Table = table;
        end
        function a = act(obj, model, s, stream)
            arguments
                obj
                model (1,1) OceanAlgo.StateSearch.ISearchModel
                s (1,1) double {mustBeInteger, mustBePositive}
                stream (1,1) RandStream
            end
            if s > size(obj.Table, 1)
                error('OceanStats:RL:DecisionTime:BadPolicyTable', 'Policy table has no row for state %d.', s);
            end
            if size(obj.Table, 2) == 1
                a = obj.Table(s);
            else
                a = find(rand(stream) <= cumsum(obj.Table(s, :)), 1, 'first');
                if isempty(a), a = find(obj.Table(s, :) > 0, 1, 'last'); end
            end
            if isempty(a) || ~ismember(a, model.actions(s))
                error('OceanStats:RL:DecisionTime:InadmissibleAction', ...
                    'The rollout policy chose an action that is not legal in state %d.', s);
            end
        end
        function name = describe(obj)
            if size(obj.Table, 2) == 1, name = "table (deterministic)"; else, name = "table (stochastic)"; end
        end
    end
end
