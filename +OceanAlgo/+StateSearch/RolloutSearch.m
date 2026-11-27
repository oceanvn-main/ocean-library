classdef RolloutSearch < handle
    % ROLLOUTSEARCH  Rollout algorithm (Sutton & Barto 2nd ed., Section 8.10): Monte Carlo estimates of the
    % rollout policy's action values at the current state, then the greedy action.
    %
    %   engine = OceanAlgo.StateSearch.RolloutSearch('RolloutPolicy', OceanAlgo.StateSearch.UniformRandomRolloutPolicy(), ...
    %                'NumRollouts', 100, 'Horizon', 100, 'DiscountFactor', 1, 'LeafValue', @(s) 0, ...
    %                'Seed', 1, 'MaxSimulatedSteps', Inf, 'RecordReturns', false)
    %   result = engine.search(model, s0)     model: ISampleModel (an IDistributionModel is sampled through
    %                                         DistributionSampleModel)
    %
    % For every legal action a at S_t, NumRollouts simulated trajectories start with a and then follow the
    % rollout policy π (OceanAlgo.StateSearch.rolloutReturn):
    %
    %   Q̂(S_t, a) = (1/n) Σ_i G_i(a)          G = discounted return, truncated after Horizon steps and corrected
    %                                           by γ^H h(S_H) (h = LeafValue, a stored evaluation function)
    %   A_t       = argmax_a Q̂(S_t, a)        ties -> first in model.actions order
    %
    % By the policy improvement theorem (S&B 4.2) the policy that takes A_t at S_t and follows π afterwards is
    % as good as π, given accurate estimates: the aim is to improve on the rollout policy, not to find an
    % optimal one. Rollouts are run round-robin over the actions (rollout i of every action before rollout
    % i+1), so a budget stop leaves the per-action counts within one of each other.
    %
    % Randomness: the engine owns RandStream('mrg32k3a', Seed); successive search calls continue the stream.
    % Status: "ok", "no_legal_action", "at_goal", "budget_exhausted" (MaxSimulatedSteps reached; the greedy
    % action over the actions sampled so far).
    % ActionValues = Q̂ (NaN if never sampled); ActionVisits = rollouts per action.
    % Work: Rollouts, SimulatedSteps, StandardErrors (1 x K; NaN with fewer than 2 rollouts).
    % Trace (RecordReturns): NumRollouts x K matrix of returns (NaN where not run).
    properties (SetAccess = private)
        RolloutPolicy
        NumRollouts (1,1) double = 100
        Horizon (1,1) double = 100
        DiscountFactor (1,1) double = 1
        LeafValue (1,1) function_handle = @(s) 0
        MaxSimulatedSteps (1,1) double = Inf
        RecordReturns (1,1) logical = false
        Stream
    end

    methods
        function obj = RolloutSearch(opts)
            arguments
                opts.RolloutPolicy (1,1) OceanAlgo.StateSearch.IRolloutPolicy = OceanAlgo.StateSearch.UniformRandomRolloutPolicy()
                opts.NumRollouts (1,1) double {mustBeInteger, mustBePositive} = 100
                opts.Horizon (1,1) double {mustBeInteger, mustBePositive} = 100
                opts.DiscountFactor (1,1) double {mustBeInRange(opts.DiscountFactor, 0, 1)} = 1
                opts.LeafValue (1,1) function_handle = @(s) 0
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 1
                opts.MaxSimulatedSteps (1,1) double {mustBePositive} = Inf
                opts.RecordReturns (1,1) logical = false
            end
            obj.RolloutPolicy = opts.RolloutPolicy; obj.NumRollouts = opts.NumRollouts;
            obj.Horizon = opts.Horizon; obj.DiscountFactor = opts.DiscountFactor; obj.LeafValue = opts.LeafValue;
            obj.MaxSimulatedSteps = opts.MaxSimulatedSteps; obj.RecordReturns = opts.RecordReturns;
            obj.Stream = RandStream('mrg32k3a', 'Seed', opts.Seed);
        end

        function result = search(obj, model, s0)
            arguments
                obj
                model (1,1) OceanAlgo.StateSearch.ISearchModel
                s0
            end
            name = "rollout";
            model = OceanAlgo.StateSearch.RolloutSearch.asSampleModel(model);
            acts = model.actions(s0);
            if model.isGoal(s0)
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "at_goal", 'Value', 0, ...
                    'Actions', acts); return
            end
            if isempty(acts)
                result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', "no_legal_action", ...
                    'Actions', acts); return
            end
            K = numel(acts); total = zeros(1, K); totalSq = zeros(1, K); n = zeros(1, K);
            returns = nan(obj.NumRollouts * obj.RecordReturns, K);
            steps = 0; status = "ok";
            for i = 1:obj.NumRollouts
                for k = 1:K
                    if steps >= obj.MaxSimulatedSteps, status = "budget_exhausted"; break; end
                    [G, st] = OceanAlgo.StateSearch.rolloutReturn(model, s0, acts(k), obj.RolloutPolicy, ...
                        obj.Stream, obj.DiscountFactor, obj.Horizon, obj.LeafValue);
                    steps = steps + st; n(k) = n(k) + 1;
                    total(k) = total(k) + G; totalSq(k) = totalSq(k) + G^2;
                    if obj.RecordReturns, returns(i, k) = G; end
                end
                if status == "budget_exhausted", break; end
            end
            q = total ./ n; q(n == 0) = NaN;
            se = sqrt(max(totalSq ./ n - q.^2, 0) .* n ./ max(n - 1, 1) ./ n); se(n < 2) = NaN;
            sampled = find(n > 0);
            best = max(q(sampled));
            pick = sampled(find(q(sampled) >= best - 1e-12 * max(1, abs(best)), 1, 'first'));
            work = struct('Rollouts', sum(n), 'SimulatedSteps', steps, 'StandardErrors', se);
            tr = []; if obj.RecordReturns, tr = returns; end
            result = OceanAlgo.StateSearch.SearchResult('Algorithm', name, 'Status', status, 'Action', acts(pick), ...
                'Value', q(pick), 'Actions', acts, 'ActionValues', q, 'ActionVisits', n, 'Work', work, 'Trace', tr);
        end
    end

    methods (Static)
        function m = asSampleModel(model)
            % ASSAMPLEMODEL  The model itself if it samples; a DistributionSampleModel around a distribution model.
            if isa(model, 'OceanAlgo.StateSearch.ISampleModel')
                m = model;
            elseif isa(model, 'OceanAlgo.StateSearch.IDistributionModel')
                m = OceanAlgo.StateSearch.DistributionSampleModel(model);
            else
                error('OceanAlgo:StateSearch:NeedSampleModel', 'This engine needs an ISampleModel or IDistributionModel.');
            end
        end
    end
end
