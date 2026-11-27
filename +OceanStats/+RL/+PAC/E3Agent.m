classdef E3Agent < OceanStats.RL.PAC.IExplorationAgent
    % E3AGENT  Explicit Explore or Exploit (Kearns & Singh 2002, "Near-optimal reinforcement learning in
    % polynomial time", Machine Learning 49: Sections 5.5 and 6), discounted case, run online.
    %
    %   agent = OceanStats.RL.PAC.E3Agent(numStates, numActions, 'KnownCount', m, 'MaxReward', Rmax, ...
    %               'DiscountFactor', 0.95, 'Epsilon', 0.1, 'Horizon', [], 'EscapeThreshold', [], ...
    %               'MinReward', 0, 'Strategy', "exploration_biased", 'Seed', 1)
    %   tr = agent.run(env, 'Steps', 10000, 'Seed', 1)
    %
    % Known state (Definition 8): a state whose every admissible action has been executed at least m times.
    % (The paper's m_known is a polynomial bound; here it is the parameter KnownCount.)
    %   * Balanced wandering: in a state that is not known, take the action tried the fewest times (ties ->
    %     lowest index). LastMode = "wander".
    %   * In a known state, two off-line computations on the known-state MDP (knownStatePlans):
    %     the exploitation policy π̂ (discounted, absorbing s0 worth MinReward/(1-γ)) and the exploration policy
    %     π̂' maximising the probability of leaving the known set within T = Horizon steps.
    %   * Decision without knowing V* (Section 6, "exploration-biased"): if π̂' escapes with probability at least
    %     EscapeThreshold (default ε / (2 G^T_max), G^T_max = T Rmax, the Lemma 8 bound), execute π̂' for T
    %     steps ("explore"); otherwise π̂ is near-optimal by Lemma 8 and is executed ("exploit").
    %     Strategy "schedule" is the practical variant suggested in Section 6: with probability α(t) = 1/t
    %     explore (if escape is possible at all), else exploit -- drawn from the agent's own RandStream(Seed).
    %   * Leaving the known set during a plan resumes balanced wandering immediately; a new known state makes
    %     the plans stale, so they are recomputed at the next known state.
    % Deviations from the paper (labelled):
    %   - online: the paper halts and outputs π̂ in the discounted case; here exploitation is run for T steps,
    %     then the explore-or-exploit decision is taken again;
    %   - exploitation uses the stationary infinite-horizon discounted optimum of M̂_S (value iteration to
    %     convergence) instead of the paper's T-step non-stationary policy (its value is within the ε-horizon
    %     bound of Lemma 2);
    %   - rewards are per state-action pair, R̂(s,a), instead of the paper's state payoffs R(i);
    %   - episodic tasks: a transition that ends the episode carries no continuation value (knownStatePlans);
    %   - a known state becomes unknown again if its action mask later admits an action not yet tried m times;
    %   - an episode reset abandons the current T-step plan.
    % Horizon default: the ε-horizon time T = ceil( (1/(1-γ)) ln( Rmax / (ε (1-γ)) ) ) (Lemma 2, Eq. 9).
    % Rewards must lie in [MinReward, MaxReward] (K&S assume [0, Rmax]); otherwise an error is raised.
    properties (SetAccess = private)
        KnownCount (1,1) double
        Epsilon (1,1) double
        Horizon (1,1) double
        EscapeThreshold (1,1) double
        MinReward (1,1) double
        Strategy (1,1) string
        Known logical                 % S x 1
        Plans struct = struct([])     % last knownStatePlans output
        Stream
        Steps (1,1) double = 0
        Decisions double = zeros(0, 3)    % [step, state, 1 explore | 0 exploit]
    end
    properties (Access = private)
        Mode_ (1,1) string = "wander"
        Left_ (1,1) double = 0        % steps left in the current T-step plan
        Stale_ (1,1) logical = true
    end
    methods
        function obj = E3Agent(numStates, numActions, opts)
            arguments
                numStates (1,1) double {mustBeInteger, mustBePositive}
                numActions (1,1) double {mustBeInteger, mustBePositive}
                opts.KnownCount (1,1) double {mustBeInteger, mustBePositive} = 10
                opts.MaxReward (1,1) double {mustBePositive, mustBeFinite} = 1
                opts.DiscountFactor (1,1) double {mustBeGreaterThanOrEqual(opts.DiscountFactor, 0), mustBeLessThan(opts.DiscountFactor, 1)} = 0.95
                opts.Epsilon (1,1) double {mustBePositive} = 0.1
                opts.Horizon double {mustBeScalarOrEmpty} = []
                opts.EscapeThreshold double {mustBeScalarOrEmpty} = []
                opts.MinReward (1,1) double {mustBeFinite} = 0
                opts.Strategy (1,1) string {mustBeMember(opts.Strategy, ["exploration_biased", "schedule"])} = "exploration_biased"
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 1
            end
            if opts.MinReward > opts.MaxReward
                error('OceanStats:RL:PAC:BadRewardRange', 'MinReward must not exceed MaxReward.');
            end
            obj.init_(numStates, numActions, opts.DiscountFactor, opts.MaxReward, Inf);
            g = opts.DiscountFactor;
            T = opts.Horizon;
            if isempty(T), T = max(1, ceil(log(opts.MaxReward / (opts.Epsilon * (1 - g))) / (1 - g))); end
            th = opts.EscapeThreshold;
            if isempty(th), th = opts.Epsilon / (2 * T * opts.MaxReward); end
            obj.KnownCount = opts.KnownCount; obj.Epsilon = opts.Epsilon; obj.Horizon = T;
            obj.EscapeThreshold = th; obj.MinReward = opts.MinReward; obj.Strategy = opts.Strategy;
            obj.Known = false(numStates, 1);
            obj.Stream = RandStream('mrg32k3a', 'Seed', opts.Seed);
        end

        function a = act(obj, s, actionMask)
            obj.noteMask_(s, actionMask);
            obj.Steps = obj.Steps + 1;
            if obj.Known(s) && any(obj.Counts.Seen(s, obj.Mask(s, :)) < obj.KnownCount)
                obj.Known(s) = false; obj.Stale_ = true;          % a newly admissible action is untried
            end
            if ~obj.Known(s)                                      % balanced wandering
                obj.Mode_ = "wander"; obj.Left_ = 0;
                adm = find(obj.Mask(s, :));
                [~, k] = min(obj.Counts.Seen(s, adm));
                a = adm(k); obj.LastMode = "wander"; return
            end
            if obj.Mode_ == "wander" || obj.Left_ <= 0 || obj.Stale_
                obj.decide_(s);
            end
            if obj.Mode_ == "explore"
                a = obj.Plans.ExplorePolicy(s, obj.Left_);        % column t = action with t steps left
            else
                a = obj.Plans.ExploitPolicy(s);
            end
            obj.Left_ = obj.Left_ - 1;
            obj.LastMode = obj.Mode_;
        end

        function observe(obj, s, a, r, s2, terminated)
            obj.checkReward_(r, obj.MinReward);
            obj.Counts.record(s, a, r, s2, terminated);
            if ~obj.Known(s)
                adm = obj.Mask(s, :);
                if all(obj.Counts.Seen(s, adm) >= obj.KnownCount)
                    obj.Known(s) = true; obj.Stale_ = true;       % discovery of a new known state
                end
            end
        end

        function p = escapeProbability(obj)
            % Current maximal probability of leaving the known set within Horizon steps, per state.
            obj.plan_(); p = obj.Plans.Escape;
        end
        function pi = exploitationPolicy(obj)
            obj.plan_(); pi = obj.Plans.ExploitPolicy;
        end
    end
    methods (Access = protected)
        function episodeStart_(obj)
            obj.Left_ = 0;                                        % a plan never crosses an episode boundary
        end
    end
    methods (Access = private)
        function plan_(obj)
            if ~obj.Stale_ && ~isempty(obj.Plans), return; end
            [R, P, Pend] = obj.Counts.estimates();
            obj.Plans = OceanStats.RL.PAC.knownStatePlans(R, P, Pend, obj.Known, obj.Mask, ...
                obj.DiscountFactor, obj.Horizon, 'MinReward', obj.MinReward);
            obj.Stale_ = false;
        end
        function decide_(obj, s)
            obj.plan_();
            pEsc = obj.Plans.Escape(s);
            if obj.Strategy == "exploration_biased"
                explore = pEsc > obj.EscapeThreshold;          % "as long as this lower bound is exceeded"
            else
                explore = pEsc > 0 && rand(obj.Stream) < 1 / obj.Steps;
            end
            if explore, obj.Mode_ = "explore"; else, obj.Mode_ = "exploit"; end
            obj.Left_ = obj.Horizon;
            obj.Decisions(end+1, :) = [obj.Steps, s, explore];
        end
    end
end
