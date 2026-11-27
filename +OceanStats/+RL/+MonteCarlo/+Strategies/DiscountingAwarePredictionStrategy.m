classdef DiscountingAwarePredictionStrategy < OceanStats.RL.MonteCarlo.Strategies.IMonteCarloStrategy
    % DISCOUNTINGAWAREPREDICTIONSTRATEGY Discounting-Aware Prediction Strategy.
    %
    % Implements Section 5.8 (Discounting-aware Importance Sampling) from
    % Sutton & Barto (2nd Ed.), treating discounting gamma as partial termination.
    %
    % Decomposes full returns into flat partial returns G_bar_{t:h} scaled by
    % truncated importance ratios rho_{t:h-1}, delegating to either:
    %   - DiscountingAwareOrdinaryISWeighting (Eq 5.9)
    %   - DiscountingAwareWeightedISWeighting (Eq 5.10)
    %
    % Package: OceanStats.RL.MonteCarlo.Strategies
    % Reference: Sutton & Barto (2018) Section 5.8, Equations 5.9-5.10

    methods
        function [V, Q, pi, history] = execute(~, solver, env)
            solver.validate_off_policy_handles_();
            gamma      = solver.Gamma;
            policy_fn  = @(s, env_) solver.pick_random_action_(s, env_);
            return_acc = 0;
            weighting  = solver.get_weighting_scheme_();

            for ep_num = 1:solver.NumEpisodes
                if isempty(solver.ForceStartState)
                    ep = solver.generate_episode_fast_(policy_fn);
                else
                    ep = solver.generate_episode_fast_(policy_fn, 'ForceStartState', solver.ForceStartState);
                end
                T = ep.Length;
                if T == 0, continue; end

                % Step 1: Compute per-step single importance ratios r_k = pi(A_k|S_k) / b(A_k|S_k)
                ratios = zeros(T, 1);
                for k = 1:T
                    s_k = ep.StateIndices(k);
                    a_k = ep.ActionIndices(k);
                    if s_k > 0 && a_k > 0
                        p_pi = solver.TargetPolicy(s_k, a_k);
                        p_b  = solver.BehaviorPolicy(s_k, a_k);
                        if p_b > 0
                            ratios(k) = p_pi / p_b;
                        else
                            ratios(k) = 0;
                        end
                    end
                end

                % Step 2: Compute flat partial returns and truncated weights (S&B §5.8)
                for t = 1:T
                    s_idx = ep.StateIndices(t);
                    if s_idx == 0, continue; end

                    g_sum      = 0;
                    w_sum      = 0;
                    rho        = 1.0;
                    cum_reward = 0;

                    for h = t:T
                        rho = rho * ratios(h);
                        if rho == 0, break; end
                        cum_reward = cum_reward + ep.Rewards(h);

                        if h < T
                            term_w = (1.0 - gamma) * (gamma^(h - t)) * rho;
                        else
                            term_w = (gamma^(T - t)) * rho;
                        end

                        g_sum = g_sum + term_w * cum_reward;
                        w_sum = w_sum + term_w;
                    end

                    weighting.update(solver.VAccumulator, g_sum, w_sum, s_idx, 0);
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
                'ClassName', "OceanStats.RL.MonteCarlo.Strategies.DiscountingAwarePredictionStrategy", ...
                'Package', "+OceanStats/+RL/+MonteCarlo/+Strategies/DiscountingAwarePredictionStrategy.m", ...
                'Category', "OceanStats.RL.Strategies", ...
                'TextbookRef', "Sutton & Barto (2018) Section 5.8, Equations 5.9 & 5.10", ...
                'TextbookEq', "G_t = (1-\gamma) \sum_{h=t+1}^{T-1} \gamma^{h-t-1} \bar{G}_{t:h} + \gamma^{T-t-1} \bar{G}_{t:T}", ...
                'SymbolicLatex', "V(s) = \frac{\sum_{t \in \mathcal{J}(s)} G_t^{\text{disc}}}{\sum_{t \in \mathcal{J}(s)} W_t^{\text{disc}}}", ...
                'Symbols', [ ...
                    struct('Symbol', 'V(s)', 'Description', 'State-value estimate for target policy pi'), ...
                    struct('Symbol', '\gamma', 'Description', 'Discount factor in [0, 1)'), ...
                    struct('Symbol', '\rho_{t:h-1}', 'Description', 'Truncated importance sampling ratio product'), ...
                    struct('Symbol', '\bar{G}_{t:h}', 'Description', 'Flat partial return sum of rewards from step t+1 to h'), ...
                    struct('Symbol', 'G_t^{\text{disc}}', 'Description', 'Discounting-aware composite return numerator for step t'), ...
                    struct('Symbol', 'W_t^{\text{disc}}', 'Description', 'Discounting-aware composite weight denominator for step t') ...
                ], ...
                'MetadataText', "Discounting-aware importance sampling prediction strategy providing variance reduction for gamma < 1.", ...
                'Methods', ["execute(solver, env) -> runs discounting-aware MC prediction"] ...
            );
        end

        function spec = getPipelineSpec()
            spec = struct( ...
                'ComponentName', "DiscountingAwarePredictionStrategy", ...
                'Kind', "VIRTUAL", ...
                'Inputs', [ ...
                    struct('Name', "Environment", 'Domain', "RL-Environment", 'Type', "IEpisodeEnvironment"), ...
                    struct('Name', "TargetPolicy", 'Domain', "RL-Policy", 'Type', "function_handle"), ...
                    struct('Name', "BehaviorPolicy", 'Domain', "RL-Policy", 'Type', "function_handle") ...
                ], ...
                'Outputs', [ ...
                    struct('Name', "VTable", 'Domain', "Real", 'Type', "double"), ...
                    struct('Name', "History", 'Domain', "Struct", 'Type', "struct") ...
                ] ...
            );
        end
    end
end
