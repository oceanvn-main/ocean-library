classdef DiscountingAwareControlStrategy < OceanStats.RL.MonteCarlo.Strategies.IMonteCarloStrategy
    % DISCOUNTINGAWARECONTROLSTRATEGY Discounting-Aware Off-Policy MC Control Strategy.
    %
    % Implements Exercise 5.14 from Sutton & Barto (2nd Ed., Chapter 5), adapting
    % the truncated weighted importance sampling estimator (Eq 5.10) to action
    % values Q(s,a) via efficient backward recursions:
    %
    %   B_t = (1 - \gamma) + \gamma r_{t+1} B_{t+1}
    %   U_t = R_{t+1} B_t + \gamma r_{t+1} U_{t+1}
    %
    % where r_k = \pi(A_k|S_k) / b(A_k|S_k).
    %
    % Unlike standard off-policy control, execution does not break immediately
    % on non-greedy actions when \gamma < 1, treating non-greedy actions as
    % partial terminations with weight (1 - \gamma).
    %
    % Package: OceanStats.RL.MonteCarlo.Strategies
    % Reference: Sutton & Barto (2018) Chapter 5, Exercise 5.14

    methods
        function [V, Q, pi, history] = execute(~, solver, env)
            solver.validate_off_policy_handles_('behavior_only');
            policy_fn  = solver.get_behavior_policy_fn_();
            gamma      = solver.Gamma;
            return_acc = 0;

            for ep_num = 1:solver.NumEpisodes
                if isempty(solver.ForceStartState)
                    ep = solver.generate_episode_fast_(policy_fn);
                else
                    ep = solver.generate_episode_fast_(policy_fn, 'ForceStartState', solver.ForceStartState);
                end
                T = ep.Length;
                if T == 0, continue; end

                % Step 1: Base case at terminal step T
                B = 1.0;
                U = ep.Rewards(T);

                s_T = ep.StateIndices(T);
                a_T = ep.ActionIndices(T);
                if s_T > 0 && a_T > 0
                    solver.QAccumulator.update(U / B, B, s_T, a_T);
                    solver.update_policy_at_state_(s_T, env);
                end

                % Step 2: Backward recursion from t = T-1 down to 1
                for t = T-1:-1:1
                    s_next = ep.StateIndices(t+1);
                    a_next = ep.ActionIndices(t+1);
                    if s_next == 0, continue; end

                    % Compute ratio r_{t+1} for next action A_{t+1} at S_{t+1}
                    target_a_next = solver.PolicyTable(s_next);
                    if a_next == target_a_next
                        b_prob = solver.get_behavior_prob_(s_next, a_next, env);
                        if b_prob > 0
                            r_next = 1.0 / b_prob;
                        else
                            r_next = 0;
                        end
                    else
                        r_next = 0;
                    end

                    % Backward recursions (S&B Exercise 5.14)
                    B = (1.0 - gamma) + gamma * r_next * B;
                    U = ep.Rewards(t) * B + gamma * r_next * U;

                    s_t = ep.StateIndices(t);
                    a_t = ep.ActionIndices(t);
                    if s_t == 0 || B <= 0, continue; end

                    solver.QAccumulator.update(U / B, B, s_t, a_t);
                    solver.update_policy_at_state_(s_t, env);
                end

                return_acc = return_acc + sum(ep.Rewards);
                solver.maybe_checkpoint_(ep_num, return_acc);
            end

            solver.finalize_tables_();
            V       = solver.VTable;
            Q       = solver.QTable;
            pi      = solver.PolicyTable;
            history = solver.History;
        end
    end

    methods (Static)
        function meta = getMetadata()
            meta = struct( ...
                'ClassName', "OceanStats.RL.MonteCarlo.Strategies.DiscountingAwareControlStrategy", ...
                'Package', "+OceanStats/+RL/+MonteCarlo/+Strategies/DiscountingAwareControlStrategy.m", ...
                'Category', "OceanStats.RL.Strategies", ...
                'TextbookRef', "Sutton & Barto (2018) Chapter 5, Exercise 5.14", ...
                'TextbookEq', "B_t = (1-\gamma) + \gamma r_{t+1} B_{t+1}, \quad U_t = R_{t+1} B_t + \gamma r_{t+1} U_{t+1}, \quad Q(s,a) = \frac{\sum U_t}{\sum B_t}", ...
                'SymbolicLatex', "B_t = (1-\gamma) + \gamma r_{t+1} B_{t+1}, \quad U_t = R_{t+1} B_t + \gamma r_{t+1} U_{t+1}, \quad Q(s,a) = \frac{\sum_t U_t}{\sum_t B_t}", ...
                'Symbols', [ ...
                    struct('Symbol', 'Q(s,a)', 'Description', 'Action-value estimate under target greedy policy pi'), ...
                    struct('Symbol', '\gamma', 'Description', 'Discount factor in [0, 1)'), ...
                    struct('Symbol', 'B_t', 'Description', 'Discounting-aware weight denominator recursion for step t'), ...
                    struct('Symbol', 'U_t', 'Description', 'Discounting-aware return numerator recursion for step t'), ...
                    struct('Symbol', 'r_{t+1}', 'Description', 'Action importance sampling ratio pi(A_{t+1}|S_{t+1}) / b(A_{t+1}|S_{t+1})'), ...
                    struct('Symbol', 'R_{t+1}', 'Description', 'Immediate reward received at step t+1') ...
                ], ...
                'MetadataText', "Discounting-aware off-policy Monte Carlo control strategy (Exercise 5.14) using efficient backward recursions for Q(s,a).", ...
                'Methods', ["execute(solver, env) -> runs discounting-aware MC control"] ...
            );
        end

        function spec = getPipelineSpec()
            spec = struct( ...
                'ComponentName', "DiscountingAwareControlStrategy", ...
                'Kind', "VIRTUAL", ...
                'Inputs', [ ...
                    struct('Name', "Environment", 'Domain', "RL-Environment", 'Type', "IEpisodeEnvironment"), ...
                    struct('Name', "BehaviorPolicy", 'Domain', "RL-Policy", 'Type', "function_handle") ...
                ], ...
                'Outputs', [ ...
                    struct('Name', "QTable", 'Domain', "Real", 'Type', "double"), ...
                    struct('Name', "PolicyTable", 'Domain', "Integer", 'Type', "double"), ...
                    struct('Name', "History", 'Domain', "Struct", 'Type', "struct") ...
                ] ...
            );
        end
    end
end
