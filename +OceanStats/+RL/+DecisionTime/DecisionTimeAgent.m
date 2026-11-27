classdef DecisionTimeAgent < handle
    % DECISIONTIMEAGENT  Decision-time planning in an RL episode (Sutton & Barto 2nd ed., Section 8.8): at
    % every state encountered, run a search on the planning model to select one action, take that action in
    % the real environment, and start planning anew from the next state.
    %
    %   agent = OceanStats.RL.DecisionTime.DecisionTimeAgent(engine, planningModel, 'ReuseTree', true)
    %     engine          any OceanAlgo.StateSearch engine (AStarSearch, IDAStarSearch, BeamSearch,
    %                     DepthLimitedSearch, RolloutSearch, MonteCarloTreeSearch)
    %     planningModel   OceanAlgo.StateSearch.ISearchModel, typically WorldModelSearchModel(env) (exact) or
    %                     WorldModelSearchModel(learnedDynaModel)
    %     ReuseTree       call engine.advance(action, nextState) after each real step when the engine has one
    %                     (MCTS subtree reuse, S&B 8.11)
    %
    %   ep = agent.runEpisode(env, 'Seed', 1, 'MaxSteps', 1000, 'DiscountFactor', 1)
    %     env   OceanStats.RL.Env.IEnvironment whose observation is the state index used by the planning model
    %           (tabular environments); for an IEnumerableEnvironment the observation is mapped with
    %           state_to_index and the planned action index with action_to_index.
    %     ep    struct: Return (Σ γ^t R_{t+1}), Steps, States, Actions, Rewards, Statuses (search status per
    %           decision), Expansions / SimulatedSteps (summed search work), Ended ("terminated" |
    %           "truncated" | "max_steps" | "no_action" | "at_goal" (planning model says the state is a goal) |
    %           "no_path" (a path engine found no way to a goal)).
    %
    % The planner only reads the planning model; the environment is stepped only here, once per decision.
    properties (SetAccess = private)
        Engine
        Model
        ReuseTree (1,1) logical = true
    end
    methods
        function obj = DecisionTimeAgent(engine, model, opts)
            arguments
                engine (1,1) handle
                model (1,1) OceanAlgo.StateSearch.ISearchModel
                opts.ReuseTree (1,1) logical = true
            end
            if ~ismethod(engine, 'search')
                error('OceanStats:RL:DecisionTime:BadEngine', 'The engine must provide search(model, state).');
            end
            obj.Engine = engine; obj.Model = model; obj.ReuseTree = opts.ReuseTree;
        end

        function ep = runEpisode(obj, env, opts)
            arguments
                obj
                env (1,1) OceanStats.RL.Env.IEnvironment
                opts.Seed double {mustBeScalarOrEmpty} = []
                opts.MaxSteps (1,1) double {mustBeInteger, mustBePositive} = 1000
                opts.DiscountFactor (1,1) double {mustBeInRange(opts.DiscountFactor, 0, 1)} = 1
            end
            [obs, ~] = env.reset('Seed', opts.Seed);
            s = obj.index_(env, obs);
            states = zeros(1, opts.MaxSteps + 1); acts = zeros(1, opts.MaxSteps); rews = zeros(1, opts.MaxSteps);
            statuses = strings(1, opts.MaxSteps); states(1) = s;
            G = 0; scale = 1; expansions = 0; simulated = 0; ended = "max_steps"; t = 0;
            reuse = obj.ReuseTree && ismethod(obj.Engine, 'advance');
            while t < opts.MaxSteps
                res = obj.Engine.search(obj.Model, s);
                if isnan(res.Action)                     % at_goal | no_path | no_legal_action
                    ended = res.Status; if ended == "no_legal_action", ended = "no_action"; end
                    break
                end
                expansions = expansions + obj.work_(res, 'Expansions');
                simulated = simulated + obj.work_(res, 'SimulatedSteps');
                a = res.Action;
                [obs, r, terminated, truncated] = env.step(obj.label_(env, s, a));
                t = t + 1;
                s2 = obj.index_(env, obs);
                acts(t) = a; rews(t) = r; states(t + 1) = s2; statuses(t) = res.Status;
                G = G + scale * r; scale = scale * opts.DiscountFactor;
                if reuse, obj.Engine.advance(a, s2); end
                s = s2;
                if terminated, ended = "terminated"; break; end
                if truncated, ended = "truncated"; break; end
            end
            ep = struct('Return', G, 'Steps', t, 'States', states(1:t + 1), 'Actions', acts(1:t), ...
                'Rewards', rews(1:t), 'Statuses', statuses(1:t), 'Expansions', expansions, ...
                'SimulatedSteps', simulated, 'Ended', ended);
        end
    end
    methods (Static, Access = private)
        function s = index_(env, obs)
            if isa(env, 'OceanStats.RL.Env.IEnumerableEnvironment'), s = env.state_to_index(obs); else, s = obs; end
        end
        function label = label_(env, s, a)
            % Planned action index -> environment action label.
            label = a;
            if isa(env, 'OceanStats.RL.Env.IEnumerableEnvironment')
                labels = env.get_valid_actions(env.index_to_state(s));
                hit = arrayfun(@(x) env.action_to_index(x), labels) == a;
                if nnz(hit) ~= 1
                    error('OceanStats:RL:DecisionTime:ActionMapping', 'Cannot map action index %d to an environment label.', a);
                end
                label = labels(hit);
            end
        end
        function v = work_(res, field)
            v = 0; if isfield(res.Work, field), v = res.Work.(field); end
        end
    end
end
