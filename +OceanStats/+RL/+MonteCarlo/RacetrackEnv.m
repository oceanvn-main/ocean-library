classdef RacetrackEnv < OceanStats.RL.Compat.LegacyEnvironment
    % RACETRACKENV  DEPRECATED name kept for existing callers: the environment now lives, algorithm-free, in
    % OceanStats.RL.Env.Classic.RacetrackEnv (Sutton & Barto 2018, Exercise 5.12). This class presents it through the older
    % three-output step / was_truncated interface and forwards the old public properties (read-only; use set_fail_prob).
    % New code should use the Classic class.
    properties (Constant)
        MAX_VELOCITY = OceanStats.RL.Env.Classic.RacetrackEnv.MAX_VELOCITY
        N_ACTIONS    = OceanStats.RL.Env.Classic.RacetrackEnv.N_ACTIONS
        ACTION_GRID  = OceanStats.RL.Env.Classic.RacetrackEnv.ACTION_GRID
    end
    properties (Dependent, SetAccess = private)
        TrackName
        Grid
        Height
        Width
        StartCells
        FinishCells
        StateIndexLUT
        ValidActionsLUT
        AllStatesCache
        TotalStates
        FailProb
        CurrentState
        EpisodeDone
    end
    methods
        function obj = RacetrackEnv(varargin)
            obj@OceanStats.RL.Compat.LegacyEnvironment(OceanStats.RL.Env.Classic.RacetrackEnv(varargin{:}));
        end
        function set_fail_prob(obj, prob), obj.Base.set_fail_prob(prob); end
        function action = index_to_action(obj, action_idx), action = obj.Base.index_to_action(action_idx); end
        function n = get_num_actions(obj), n = obj.Base.get_num_actions(); end
    end
    methods (Static)
        function meta = getMetadata()
            meta = OceanStats.RL.Env.Classic.RacetrackEnv.classMetadata();
            meta.ClassName = "OceanStats.RL.MonteCarlo.RacetrackEnv";
            meta.Package = "+OceanStats/+RL/+MonteCarlo/RacetrackEnv.m";
        end
        function spec = getPipelineSpec()
            spec = OceanStats.RL.Env.Classic.RacetrackEnv.classPipelineSpec();
        end
    end

    methods
        function v = get.TrackName(obj), v = obj.Base.TrackName; end
        function v = get.Grid(obj), v = obj.Base.Grid; end
        function v = get.Height(obj), v = obj.Base.Height; end
        function v = get.Width(obj), v = obj.Base.Width; end
        function v = get.StartCells(obj), v = obj.Base.StartCells; end
        function v = get.FinishCells(obj), v = obj.Base.FinishCells; end
        function v = get.StateIndexLUT(obj), v = obj.Base.StateIndexLUT; end
        function v = get.ValidActionsLUT(obj), v = obj.Base.ValidActionsLUT; end
        function v = get.AllStatesCache(obj), v = obj.Base.AllStatesCache; end
        function v = get.TotalStates(obj), v = obj.Base.TotalStates; end
        function v = get.FailProb(obj), v = obj.Base.FailProb; end
        function v = get.CurrentState(obj), v = obj.Base.CurrentState; end
        function v = get.EpisodeDone(obj), v = obj.Base.EpisodeDone; end
    end
end
