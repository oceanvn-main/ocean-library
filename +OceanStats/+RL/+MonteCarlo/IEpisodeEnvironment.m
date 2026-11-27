classdef (Abstract) IEpisodeEnvironment < handle
    % IEPISODEENVIRONMENT Abstract base class for episodic, sample-based environments.
    %
    % Defines the minimal interface contract required by Monte Carlo and other
    % model-free RL solvers (Chapter 5, Sutton & Barto 2018).
    %
    % KEY DISTINCTION from TabularEnvironment:
    %   TabularEnvironment  → exposes full P(s'|s,a) transition model (Ch.4 DP).
    %   IEpisodeEnvironment → exposes only step/reset (Ch.5+ MC/TD/FA).
    %
    % Package:    OceanStats.RL.MonteCarlo
    % Reference:  Sutton & Barto (2018) Chapter 5
    %
    % Abstract Methods Required:
    %   state              = reset(obj)
    %   [s',r,done]        = step(obj, action)
    %   actions            = get_valid_actions(obj, state)
    %   all_states         = get_all_states(obj)
    %   idx                = state_to_index(obj, state)
    %   action_idx         = action_to_index(obj, action)
    %   n_actions          = get_num_actions(obj)
    %   tf                 = is_terminal(obj, state)

    methods (Abstract)
        % RESET Begins a new episode and returns the initial state.
        %
        % Output:
        %   state - (1 x D) initial state vector
        state = reset(obj)

        % STEP Takes one action in the environment.
        %
        % Input:
        %   action     - scalar action value
        % Outputs:
        %   next_state - (1 x D) resulting state
        %   reward     - scalar immediate reward
        %   done       - logical; true if episode has ended
        [next_state, reward, done] = step(obj, action)

        % GET_VALID_ACTIONS Returns legally admissible actions for a given state.
        %
        % Input:
        %   state   - (1 x D) state vector
        % Output:
        %   actions - (1 x A) row vector of valid action values
        actions = get_valid_actions(obj, state)

        % GET_ALL_STATES Returns the complete finite state space.
        %
        % Output:
        %   all_states - (N x D) matrix; each row is one state vector
        all_states = get_all_states(obj)

        % STATE_TO_INDEX Maps a state vector to a unique integer index in [1, N].
        %
        % Should be O(1) (LUT-based) for performance in MC loops.
        %
        % Input:
        %   state - (1 x D) state vector
        % Output:
        %   idx   - scalar integer in [1, N]
        idx = state_to_index(obj, state)

        % ACTION_TO_INDEX Maps an action value to a unique integer index in [1, A].
        %
        % Used for Q-table column indexing.
        %
        % Input:
        %   action     - scalar action value
        % Output:
        %   action_idx - scalar integer in [1, A]
        action_idx = action_to_index(obj, action)

        % GET_NUM_ACTIONS Returns the total number of discrete actions (A).
        n_actions = get_num_actions(obj)

        % IS_TERMINAL Returns true if the given state is terminal.
        %
        % Input:
        %   state - (1 x D) state vector
        % Output:
        %   tf    - logical
        tf = is_terminal(obj, state)
    end

    methods
        function tf = was_truncated(~)
            % WAS_TRUNCATED  True if the LAST step ended the episode by a time limit / cut-off that is
            % outside the MDP (Gymnasium's `truncated`), as opposed to a true terminal state. Bootstrapping
            % (TD) must stop only on true termination, so learners call this after step() returned done=true.
            % Default: false (existing environments end episodes only by termination). Override to report cut-offs.
            tf = false;
        end

        function n = get_num_states(obj)
            % GET_NUM_STATES Returns total number of states N.
            n = size(obj.get_all_states(), 1);
        end

        function actions = get_all_actions(obj)
            % GET_ALL_ACTIONS Returns all discrete action values.
            %
            % Default implementation queries valid actions from the first state.
            % Override for O(1) lookup when action space is state-independent.
            states = obj.get_all_states();
            actions = obj.get_valid_actions(states(1, :));
        end

        function ep = generate_episode(obj, policy_fn, opts)
            % GENERATE_EPISODE Rolls out one full episode following policy_fn.
            %
            % Inputs:
            %   policy_fn  - function handle @(state, env) → action
            %   opts.MaxLen - maximum episode length (default 500)
            %   opts.ForceStartState - (1 x D) fixed start state ([] = use reset)
            %   opts.ForceStartAction- scalar fixed first action (NaN = use policy)
            %
            % Output:
            %   ep - struct with fields:
            %     .States        (T x D)   states visited
            %     .StateIndices  (T x 1)   linear state indices for table lookup
            %     .Actions       (T x 1)   actions taken
            %     .ActionIndices (T x 1)   action indices for Q-table column lookup
            %     .Rewards       (T x 1)   immediate rewards received
            %     .Length        scalar    episode length T
            %     .Terminated    logical   true if a terminal state was reached (false = cut off at MaxLen)
            %     .BehaviorProbs (T x 1)   b(A_t | S_t) when the policy returns it as a 2nd output
            %                              (opts.PolicyReturnsProb); NaN otherwise
            %
            % This method lives on the interface so any concrete environment
            % can be driven uniformly by the MonteCarloSolver.
            arguments
                obj
                policy_fn  function_handle
                opts.MaxLen              (1,1) double  {mustBePositive} = 500
                opts.ForceStartState     double = []
                opts.ForceStartAction    double = NaN
                opts.PolicyReturnsProb   (1,1) logical = false   % policy_fn -> [action, b(action | state)]
            end

            max_len = opts.MaxLen;

            % Pre-allocate episode buffers (avoids dynamic growth in hot loop)
            D = size(obj.get_all_states(), 2);
            states_buf   = zeros(max_len, D);
            s_idx_buf    = zeros(max_len, 1);
            actions_buf  = zeros(max_len, 1);
            a_idx_buf    = zeros(max_len, 1);
            rewards_buf  = zeros(max_len, 1);
            probs_buf    = nan(max_len, 1);

            % --- Episode Initialisation ---
            if isempty(opts.ForceStartState)
                s = obj.reset();
            else
                obj.reset();    % reset internal state (deck, hands, etc.)
                if ismethod(obj, 'set_state')
                    obj.set_state(opts.ForceStartState);
                end
                s = opts.ForceStartState;
            end

            t = 0;
            done = obj.is_terminal(s);

            while ~done && t < max_len
                t = t + 1;

                % Select action
                if t == 1 && ~isnan(opts.ForceStartAction)
                    a = opts.ForceStartAction;
                elseif opts.PolicyReturnsProb
                    [a, probs_buf(t)] = policy_fn(s, obj);          % probability at the moment of choice
                else
                    a = policy_fn(s, obj);
                end

                % Record (s, a) before transition
                states_buf(t, :)  = s;
                s_idx_buf(t)      = obj.state_to_index(s);
                actions_buf(t)    = a;
                a_idx_buf(t)      = obj.action_to_index(a);

                % Take step
                [s_next, r, done] = obj.step(a);
                rewards_buf(t)    = r;

                s = s_next;
            end

            % Trim to actual length
            ep.States        = states_buf(1:t, :);
            ep.StateIndices  = s_idx_buf(1:t);
            ep.Actions       = actions_buf(1:t);
            ep.ActionIndices = a_idx_buf(1:t);
            ep.Rewards       = rewards_buf(1:t);
            ep.Length        = t;
            ep.Terminated    = logical(done);
            ep.BehaviorProbs = probs_buf(1:t);
        end
    end
end
