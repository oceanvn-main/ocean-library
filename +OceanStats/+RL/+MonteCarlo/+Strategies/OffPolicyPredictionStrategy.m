classdef OffPolicyPredictionStrategy < OceanStats.RL.MonteCarlo.Strategies.IMonteCarloStrategy
    % OFFPOLICYPREDICTIONSTRATEGY Off-Policy Prediction Strategy.
    %
    % Estimates V_pi(s) under target policy pi using episodes generated under
    % behavior policy b (Sutton & Barto 2nd Ed., Sections 5.5-5.6).
    %
    % Delegates numeric ratio/weight updates to an IImportanceWeighting scheme
    % (OrdinaryISWeighting or WeightedISWeighting).
    %
    % Package: OceanStats.RL.MonteCarlo.Strategies
    % Reference: Sutton & Barto (2018) Sections 5.5-5.6

    methods
        function [V, Q, pi, history] = execute(~, solver, env)
            solver.validate_off_policy_handles_();

            policy_fn  = @(s, env_) solver.pick_random_action_(s, env_);
            return_acc = 0;
            weighting  = solver.get_weighting_scheme_();

            for ep_num = 1:solver.NumEpisodes
                if isempty(solver.ForceStartState)
                    ep = solver.generate_episode_fast_(policy_fn);
                else
                    ep = solver.generate_episode_fast_(policy_fn, 'ForceStartState', solver.ForceStartState);
                end
                G = 0;
                W = 1.0;

                for t = ep.Length:-1:1
                    G     = solver.Gamma * G + ep.Rewards(t);
                    s_idx = ep.StateIndices(t);
                    a_idx = ep.ActionIndices(t);
                    if s_idx == 0, continue; end

                    pi_prob = solver.TargetPolicy(s_idx, a_idx);
                    b_prob  = solver.BehaviorPolicy(s_idx, a_idx);

                    if b_prob <= 0
                        W = 0;
                    else
                        W = W * (pi_prob / b_prob);
                    end

                    weighting.update(solver.VAccumulator, G, W, s_idx, 0);

                    if W == 0 && solver.ISType == "Weighted"
                        break;
                    end
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
                'ClassName', "OceanStats.RL.MonteCarlo.Strategies.OffPolicyPredictionStrategy", ...
                'Package', "+OceanStats/+RL/+MonteCarlo/+Strategies/OffPolicyPredictionStrategy.m", ...
                'Category', "OceanStats.RL.Strategies", ...
                'TextbookRef', "Sutton & Barto (2018) Sections 5.5-5.6", ...
                'TextbookEq', "V(s) \doteq \frac{\sum_{t \in \mathcal{J}(s)} W_t G_t}{\sum_{t \in \mathcal{J}(s)} W_t}", ...
                'SymbolicLatex', "V(s) = \frac{\sum_{t \in \mathcal{J}(s)} \rho_{t:T-1} G_t}{\sum_{t \in \mathcal{J}(s)} \rho_{t:T-1}}", ...
                'Symbols', [ ...
                    struct('Symbol', 'V(s)', 'Description', 'State-value estimate for target policy pi'), ...
                    struct('Symbol', '\rho_{t:T-1}', 'Description', 'Importance sampling ratio product W_t'), ...
                    struct('Symbol', 'G_t', 'Description', 'Sample return following visit to state s at step t') ...
                ], ...
                'MetadataText', "Off-Policy Monte Carlo prediction strategy delegating updates to IImportanceWeighting scheme.", ...
                'Methods', ["execute(solver, env) -> runs off-policy MC prediction"] ...
            );
        end

        function spec = getPipelineSpec()
            spec = struct( ...
                'ComponentName', "OffPolicyPredictionStrategy", ...
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
