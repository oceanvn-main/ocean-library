classdef BlackjackEnv < OceanStats.RL.Compat.LegacyEnvironment
    % BLACKJACKENV  DEPRECATED name kept for existing callers: the environment now lives, algorithm-free, in
    % OceanStats.RL.Env.Classic.BlackjackEnv (Sutton & Barto 2018, Example 5.1). This class presents it through the
    % older three-output step / was_truncated interface and forwards the old public properties (read-only).
    % New code should use OceanStats.RL.Env.Classic.BlackjackEnv directly.
    properties (Constant)
        PLAYER_SUM_MIN = OceanStats.RL.Env.Classic.BlackjackEnv.PLAYER_SUM_MIN
        PLAYER_SUM_MAX = OceanStats.RL.Env.Classic.BlackjackEnv.PLAYER_SUM_MAX
        DEALER_MIN     = OceanStats.RL.Env.Classic.BlackjackEnv.DEALER_MIN
        DEALER_MAX     = OceanStats.RL.Env.Classic.BlackjackEnv.DEALER_MAX
        N_STATES       = OceanStats.RL.Env.Classic.BlackjackEnv.N_STATES
        N_ACTIONS      = OceanStats.RL.Env.Classic.BlackjackEnv.N_ACTIONS
    end
    properties (Dependent, SetAccess = private)
        StateIndexLUT
        AllStatesCache
        PlayerHand
        DealerHand
        EpisodeDone
    end
    methods
        function obj = BlackjackEnv()
            obj@OceanStats.RL.Compat.LegacyEnvironment(OceanStats.RL.Env.Classic.BlackjackEnv());
        end
        function action = index_to_action(obj, action_idx), action = obj.Base.index_to_action(action_idx); end
    end

    methods
        function v = get.StateIndexLUT(obj), v = obj.Base.StateIndexLUT; end
        function v = get.AllStatesCache(obj), v = obj.Base.AllStatesCache; end
        function v = get.PlayerHand(obj), v = obj.Base.PlayerHand; end
        function v = get.DealerHand(obj), v = obj.Base.DealerHand; end
        function v = get.EpisodeDone(obj), v = obj.Base.EpisodeDone; end
    end
end
