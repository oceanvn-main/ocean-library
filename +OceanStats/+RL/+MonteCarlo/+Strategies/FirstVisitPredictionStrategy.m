classdef FirstVisitPredictionStrategy < OceanStats.RL.MonteCarlo.Strategies.IMonteCarloStrategy
    % FIRSTVISITPREDICTIONSTRATEGY First-Visit MC Prediction Strategy.
    %
    % Estimates V_pi(s) for a given policy pi using sample episodes
    % (Sutton & Barto 2nd Ed., page 92).
    %
    % Package: OceanStats.RL.MonteCarlo.Strategies
    % Reference: Sutton & Barto (2018) Section 5.1, page 92

    methods
        function [V, Q, pi, history] = execute(~, solver, env)
            policy_fn  = solver.get_target_policy_fn_();
            return_acc = 0;
            n_states   = env.get_num_states();

            for ep_num = 1:solver.NumEpisodes
                if isempty(solver.ForceStartState)
                    ep = solver.generate_episode_fast_(policy_fn);
                else
                    ep = solver.generate_episode_fast_(policy_fn, 'ForceStartState', solver.ForceStartState);
                end
                G = 0;
                % "Unless S_t appears in S_0, ..., S_{t-1}" (Sutton & Barto p. 92): mark each state's FIRST time step
                % going forward; the backward loop updates only there. (A backward-scan "visited" flag would keep the
                % LAST visit instead - every-visit-like returns from the latest occurrence.)
                first = OceanStats.RL.MonteCarlo.Strategies.FirstVisitPredictionStrategy.firstVisits_( ...
                    ep.StateIndices(1:ep.Length), n_states);

                for t = ep.Length:-1:1
                    G     = solver.Gamma * G + ep.Rewards(t);
                    s_idx = ep.StateIndices(t);
                    if s_idx == 0, continue; end

                    if first(t)
                        solver.VAccumulator.update(G, 1.0, s_idx);
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
        function first = firstVisits_(keys, numKeys)
            % FIRSTVISITS_  first(t) is true when keys(t) did not occur at any earlier time step (keys 0 are ignored).
            first = false(numel(keys), 1);
            seen = false(numKeys, 1);
            for t = 1:numel(keys)
                k = keys(t);
                if k > 0 && ~seen(k)
                    seen(k) = true;
                    first(t) = true;
                end
            end
        end

        function meta = getMetadata()
            meta = struct( ...
                'ClassName', "OceanStats.RL.MonteCarlo.Strategies.FirstVisitPredictionStrategy", ...
                'Package', "+OceanStats/+RL/+MonteCarlo/+Strategies/FirstVisitPredictionStrategy.m", ...
                'Category', "OceanStats.RL.Strategies", ...
                'TextbookRef', "Sutton & Barto (2018) Section 5.1, page 92", ...
                'TextbookEq', "V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)} \left[ G - V(S_t) \right]", ...
                'SymbolicLatex', "V(s) = \frac{1}{|\mathcal{J}_1(s)|} \sum_{t \in \mathcal{J}_1(s)} G_t", ...
                'Symbols', [ ...
                    struct('Symbol', 'V(s)', 'Description', 'State-value estimate for state s under policy pi'), ...
                    struct('Symbol', '\mathcal{J}_1(s)', 'Description', 'Set of first-visit time steps for state s in episodes'), ...
                    struct('Symbol', 'G_t', 'Description', 'Sample return following first visit to state s at step t') ...
                ], ...
                'MetadataText', "First-Visit Monte Carlo prediction algorithm estimating state values V_pi(s).", ...
                'Methods', ["execute(solver, env) -> runs first-visit MC prediction"] ...
            );
        end

        function spec = getPipelineSpec()
            spec = struct( ...
                'ComponentName', "FirstVisitPredictionStrategy", ...
                'Kind', "VIRTUAL", ...
                'Inputs', [ ...
                    struct('Name', "Environment", 'Domain', "RL-Environment", 'Type', "IEpisodeEnvironment"), ...
                    struct('Name', "TargetPolicy", 'Domain', "RL-Policy", 'Type', "function_handle") ...
                ], ...
                'Outputs', [ ...
                    struct('Name', "VTable", 'Domain', "Real", 'Type', "double"), ...
                    struct('Name', "History", 'Domain', "Struct", 'Type', "struct") ...
                ] ...
            );
        end
    end
end
