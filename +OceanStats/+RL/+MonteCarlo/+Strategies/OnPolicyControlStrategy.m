classdef OnPolicyControlStrategy < OceanStats.RL.MonteCarlo.Strategies.IMonteCarloStrategy
    % ONPOLICYCONTROLSTRATEGY On-Policy Epsilon-Greedy MC Control Strategy.
    %
    % Finds optimal epsilon-soft policy using epsilon-greedy action sampling
    % (Sutton & Barto 2nd Ed., page 101).
    %
    % Package: OceanStats.RL.MonteCarlo.Strategies
    % Reference: Sutton & Barto (2018) Section 5.4, page 101

    methods
        function [V, Q, pi, history] = execute(~, solver, env)
            n_states   = env.get_num_states();
            n_actions  = env.get_num_actions();
            return_acc = 0;

            policy_fn = @(s, env_) solver.epsilon_greedy_action_(s, env_);

            for ep_num = 1:solver.NumEpisodes
                if isempty(solver.ForceStartState)
                    ep = solver.generate_episode_fast_(policy_fn);
                else
                    ep = solver.generate_episode_fast_(policy_fn, 'ForceStartState', solver.ForceStartState);
                end
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
                'ClassName', "OceanStats.RL.MonteCarlo.Strategies.OnPolicyControlStrategy", ...
                'Package', "+OceanStats/+RL/+MonteCarlo/+Strategies/OnPolicyControlStrategy.m", ...
                'Category', "OceanStats.RL.Strategies", ...
                'TextbookRef', "Sutton & Barto (2018) Section 5.4, page 101", ...
                'TextbookEq', "\pi(a|s) = \frac{\epsilon}{|A|} + (1 - \epsilon) \mathbb{I}(a = \arg\max_a Q(s,a))", ...
                'SymbolicLatex', "\pi(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|A|}, & \text{if } a = \arg\max_a Q(s,a) \\ \frac{\epsilon}{|A|}, & \text{otherwise} \end{cases}", ...
                'Symbols', [ ...
                    struct('Symbol', '\pi(a|s)', 'Description', 'Epsilon-soft policy probability for action a in state s'), ...
                    struct('Symbol', '\epsilon', 'Description', 'Exploration probability parameter in [0, 1]'), ...
                    struct('Symbol', '|A|', 'Description', 'Number of valid actions in state s'), ...
                    struct('Symbol', 'Q(s,a)', 'Description', 'Action-value estimate for state-action pair (s,a)') ...
                ], ...
                'MetadataText', "On-policy epsilon-greedy Monte Carlo control algorithm.", ...
                'Methods', ["execute(solver, env) -> runs on-policy epsilon-greedy MC control"] ...
            );
        end

        function spec = getPipelineSpec()
            spec = struct( ...
                'ComponentName', "OnPolicyControlStrategy", ...
                'Kind', "VIRTUAL", ...
                'Inputs', [ ...
                    struct('Name', "Environment", 'Domain', "RL-Environment", 'Type', "IEpisodeEnvironment"), ...
                    struct('Name', "Epsilon", 'Domain', "Real", 'Type', "double") ...
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
