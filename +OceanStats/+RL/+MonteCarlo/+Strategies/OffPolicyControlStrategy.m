classdef OffPolicyControlStrategy < OceanStats.RL.MonteCarlo.Strategies.IMonteCarloStrategy
    % OFFPOLICYCONTROLSTRATEGY Off-Policy MC Control Strategy.
    %
    % Finds optimal action values Q*(s,a) and policy pi* using Weighted
    % Importance Sampling from behavior policy b (Sutton & Barto 2nd Ed., page 111).
    %
    % Delegates importance weighting updates to an IImportanceWeighting scheme.
    %
    % Package: OceanStats.RL.MonteCarlo.Strategies
    % Reference: Sutton & Barto (2018) Section 5.7, page 111

    methods
        function [V, Q, pi, history] = execute(~, solver, env)
            solver.validate_off_policy_handles_('behavior_only');
            policy_fn  = solver.get_behavior_policy_fn_();
            return_acc = 0;
            weighting  = solver.get_weighting_scheme_();
            default_b  = isempty(solver.BehaviorPolicy) || ...
                         (isnumeric(solver.BehaviorPolicy) && isempty(solver.BehaviorPolicy));
            recorded_b = default_b && solver.TieBreak == "random";   % b(A|S) as it was when chosen
            solver.History.EpisodeReturns = zeros(1, solver.NumEpisodes);

            for ep_num = 1:solver.NumEpisodes
                if recorded_b
                    ep = solver.generate_episode_fast_(policy_fn, 'PolicyReturnsProb', true);
                else
                    ep = solver.generate_episode_fast_(policy_fn);
                end
                solver.History.EpisodeReturns(ep_num) = sum(ep.Rewards);
                G  = 0;
                W  = 1.0;
                if solver.TruncatedEpisodes == "skip" && ~ep.Terminated
                    % cut off at MaxEpisodeLen: no complete return, so no MC update (S&B Ch. 5)
                    return_acc = return_acc + sum(ep.Rewards);
                    solver.maybe_checkpoint_(ep_num, return_acc);
                    continue;
                end

                for t = ep.Length:-1:1
                    G     = solver.Gamma * G + ep.Rewards(t);
                    s_idx = ep.StateIndices(t);
                    a_idx = ep.ActionIndices(t);
                    if s_idx == 0, continue; end

                    weighting.update(solver.QAccumulator, G, W, s_idx, a_idx);

                    q_vals      = solver.QAccumulator.getValue();
                    all_states  = env.get_all_states();
                    s_vec       = all_states(s_idx, :);
                    valid_acts  = env.get_valid_actions(s_vec);
                    act_indices = env.action_to_index(valid_acts);
                    q_valid     = q_vals(s_idx, act_indices);
                    if solver.TieBreak == "random"
                        ties = act_indices(q_valid == max(q_valid));
                        if ismember(a_idx, ties)
                            best_a_idx = a_idx;                 % any maximiser is greedy: keep the taken one
                        else
                            best_a_idx = ties(randi(numel(ties)));
                        end
                    else
                        [~, best_i] = max(q_valid);
                        best_a_idx  = act_indices(best_i);
                    end
                    solver.PolicyTable(s_idx) = best_a_idx;

                    if a_idx ~= best_a_idx
                        break;
                    end

                    if recorded_b
                        b_prob  = ep.BehaviorProbs(t);
                    elseif default_b
                        m_valid = numel(valid_acts);
                        b_prob  = (1.0 - solver.Epsilon) + (solver.Epsilon / m_valid);
                    else
                        b_prob = solver.BehaviorPolicy(s_idx, a_idx);
                    end

                    if b_prob <= 0
                        break;
                    end
                    W = W / b_prob;
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
                'ClassName', "OceanStats.RL.MonteCarlo.Strategies.OffPolicyControlStrategy", ...
                'Package', "+OceanStats/+RL/+MonteCarlo/+Strategies/OffPolicyControlStrategy.m", ...
                'Category', "OceanStats.RL.Strategies", ...
                'TextbookRef', "Sutton & Barto (2018) Section 5.7, page 111", ...
                'TextbookEq', "Q(s,a) \leftarrow Q(s,a) + \frac{W}{C(s,a)} \left[ G - Q(s,a) \right]", ...
                'SymbolicLatex', "Q(s,a) = \frac{\sum W_t G_t}{\sum W_t}, \quad \pi(s) = \arg\max_a Q(s,a)", ...
                'Symbols', [ ...
                    struct('Symbol', 'Q(s,a)', 'Description', 'Action-value estimate for state-action pair (s,a)'), ...
                    struct('Symbol', 'W_t', 'Description', 'Importance sampling weight product'), ...
                    struct('Symbol', '\pi(s)', 'Description', 'Greedy action selection under target policy pi'), ...
                    struct('Symbol', 'G_t', 'Description', 'Sample return following action a in state s') ...
                ], ...
                'MetadataText', "Off-policy Monte Carlo control strategy delegating updates to IImportanceWeighting scheme.", ...
                'Methods', ["execute(solver, env) -> runs off-policy MC control"] ...
            );
        end

        function spec = getPipelineSpec()
            spec = struct( ...
                'ComponentName', "OffPolicyControlStrategy", ...
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
