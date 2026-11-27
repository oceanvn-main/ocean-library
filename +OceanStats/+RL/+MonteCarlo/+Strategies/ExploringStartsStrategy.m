classdef ExploringStartsStrategy < OceanStats.RL.MonteCarlo.Strategies.IMonteCarloStrategy
    % EXPLORINGSTARTSSTRATEGY Monte Carlo Exploring Starts Control Strategy.
    %
    % Finds optimal action values Q*(s,a) and policy pi* using exploring starts
    % (Sutton & Barto 2nd Ed., page 99).
    %
    % Package: OceanStats.RL.MonteCarlo.Strategies
    % Reference: Sutton & Barto (2018) Section 5.3, page 99

    methods
        function [V, Q, pi, history] = execute(~, solver, env)
            n_states   = env.get_num_states();
            n_actions  = env.get_num_actions();
            all_states = env.get_all_states();
            all_acts   = env.get_all_actions();
            return_acc = 0;

            for ep_num = 1:solver.NumEpisodes
                % Random starting state and action
                s_rand_row = randi(n_states);
                a_rand_col = randi(n_actions);
                s0         = all_states(s_rand_row, :);
                a0         = all_acts(a_rand_col);

                policy_fn  = @(s, env_) solver.greedy_action_(s, env_);
                ep = solver.generate_episode_fast_(policy_fn, ...
                    'ForceStartState', s0, 'ForceStartAction', a0);

                G       = 0;
                % first visit of each (S_t, A_t) pair, found going FORWARD (Sutton & Barto pp. 99, 101: "unless the
                % pair S_t, A_t appears in S_0, A_0, ..., S_{t-1}, A_{t-1}")
                pairKeys = (ep.ActionIndices(1:ep.Length) - 1) * n_states + ep.StateIndices(1:ep.Length);
                pairKeys(ep.StateIndices(1:ep.Length) == 0) = 0;
                first = OceanStats.RL.MonteCarlo.Strategies.FirstVisitPredictionStrategy.firstVisits_( ...
                    pairKeys, n_states * n_actions);

                for t = ep.Length:-1:1
                    G     = solver.Gamma * G + ep.Rewards(t);
                    s_idx = ep.StateIndices(t);
                    a_idx = ep.ActionIndices(t);
                    if s_idx == 0, continue; end

                    if first(t)
                        solver.QAccumulator.update(G, 1.0, s_idx, a_idx);

                        q_vals = solver.QAccumulator.getValue();
                        [~, best_a] = max(q_vals(s_idx, :));
                        solver.PolicyTable(s_idx) = best_a;
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
                'ClassName', "OceanStats.RL.MonteCarlo.Strategies.ExploringStartsStrategy", ...
                'Package', "+OceanStats/+RL/+MonteCarlo/+Strategies/ExploringStartsStrategy.m", ...
                'Category', "OceanStats.RL.Strategies", ...
                'TextbookRef', "Sutton & Barto (2018) Section 5.3, page 99", ...
                'TextbookEq', "\pi(s) \doteq \arg\max_a Q(s,a)", ...
                'SymbolicLatex', "Q(s,a) = \frac{1}{|\mathcal{J}_1(s,a)|} \sum_{t \in \mathcal{J}_1(s,a)} G_t, \quad \pi(s) = \arg\max_a Q(s,a)", ...
                'Symbols', [ ...
                    struct('Symbol', 'Q(s,a)', 'Description', 'Action-value estimate for state-action pair (s,a)'), ...
                    struct('Symbol', '\pi(s)', 'Description', 'Greedy action selection under target policy pi'), ...
                    struct('Symbol', 'G_t', 'Description', 'Sample return following exploring start at step t') ...
                ], ...
                'MetadataText', "Monte Carlo Exploring Starts control algorithm finding optimal Q*(s,a) and pi*(s).", ...
                'Methods', ["execute(solver, env) -> runs MC Exploring Starts control"] ...
            );
        end

        function spec = getPipelineSpec()
            spec = struct( ...
                'ComponentName', "ExploringStartsStrategy", ...
                'Kind', "VIRTUAL", ...
                'Inputs', [ ...
                    struct('Name', "Environment", 'Domain', "RL-Environment", 'Type', "IEpisodeEnvironment") ...
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
