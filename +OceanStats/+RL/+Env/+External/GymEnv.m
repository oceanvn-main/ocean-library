classdef GymEnv < OceanStats.RL.Env.IEnvironment
    % GYMENV  A Gymnasium environment (Python) presented as an OceanStats.RL.Env.IEnvironment.
    %   env = OceanStats.RL.Env.External.GymEnv("CartPole-v1");   [obs, info] = env.reset('Seed', 7);   [obs, r, term, trunc, info] = env.step(1);
    % Conventions: actions are 1..A (Gymnasium 0..A-1); a Discrete observation s becomes s+1; a Box observation is a
    % 1 x D double row. info.action_mask is Gymnasium's own mask when it gives one, else all true; all false at a terminal.
    % 'Strict' (default true) refuses a step before reset, after the episode ended, or outside the ActionSpace, and
    % changes nothing. Strict = false forwards every call to Gymnasium as it is (which only warns after termination).
    % Randomness: Gymnasium owns the stream. reset('Seed', s) / setSeed(s) seed it; the first reset of an environment
    % that was never seeded draws its seed from MATLAB's global stream, so rng(k) still makes an unseeded run reproducible
    % (IEnvironment contract); later unseeded resets continue Gymnasium's stream.
    % Requires a Python environment visible to MATLAB (pyenv) with the gymnasium package; otherwise the constructor
    % raises OceanStats:RL:Env:External:GymEnv:NoGymnasium. The Python side is gym_bridge.py in this folder.
    % Conformance: with Strict = true it satisfies the environment axioms RL-1..RL-9; raw Gymnasium behaviour
    % (Strict = false) accepts steps after termination and violates RL-5.

    properties (SetAccess = protected)
        Name string
        Strict (1,1) logical = true
        IsDiscreteObs (1,1) logical = false
        NumActions double = 0
    end
    properties (Access = protected)
        Py
        PendingSeed = []
        EverSeeded (1,1) logical = false
        Live (1,1) logical = false
    end

    methods
        function obj = GymEnv(name, options)
            arguments
                name (1,1) string
                options.Strict (1,1) logical = true
            end
            obj.Name = name;  obj.Strict = options.Strict;
            here = fileparts(mfilename('fullpath'));
            try
                if count(py.sys.path, here) == 0, insert(py.sys.path, int32(0), here); end
                obj.Py = py.gym_bridge.Bridge(char(name));
            catch err
                error('OceanStats:RL:Env:External:GymEnv:NoGymnasium', ...
                    'Could not create the Gymnasium environment "%s" (is Python with gymnasium available to MATLAB?): %s', ...
                    name, err.message);
            end
            sp = cell(obj.Py.spaces());
            obj.NumActions = double(sp{5});
            obj.ActionSpace = OceanStats.RL.Env.Spaces.Discrete(obj.NumActions);
            obj.IsDiscreteObs = string(sp{1}) == "discrete";
            if obj.IsDiscreteObs
                obj.ObservationSpace = OceanStats.RL.Env.Spaces.Discrete(double(sp{2}));
            else
                obj.ObservationSpace = OceanStats.RL.Env.Spaces.Box(double(sp{3}), double(sp{4}));
            end
        end

        function setSeed(obj, seed)
            obj.PendingSeed = seed;
        end

        function [obs, reward, terminated, truncated, info] = step(obj, action)
            if obj.Strict
                if ~obj.Live
                    error('OceanStats:RL:Env:External:GymEnv:NotReset', 'step called before reset or after the episode ended.');
                end
                if ~obj.ActionSpace.contains(action)
                    error('OceanStats:RL:Env:External:GymEnv:InvalidAction', 'The action is not in %s.', obj.ActionSpace.describe());
                end
            end
            out = cell(obj.Py.step(action - 1));
            obs = obj.obs_(out{1});  reward = double(out{2});
            terminated = logical(out{3});  truncated = logical(out{4});
            mask = logical(double(out{5}));
            if terminated, mask = false(1, obj.NumActions); end
            info = struct('action_mask', reshape(mask, 1, []));
            obj.Live = ~(terminated || truncated);
        end
    end

    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            seed = -1;
            if ~isempty(obj.PendingSeed)
                seed = obj.PendingSeed;  obj.PendingSeed = [];
            elseif ~obj.EverSeeded
                seed = randi(2^31 - 1);                     % first unseeded reset: reproducible through rng(k)
            end
            obj.EverSeeded = true;
            out = cell(obj.Py.reset(seed));
            obs = obj.obs_(out{1});
            info = struct('action_mask', reshape(logical(double(out{2})), 1, []));
            obj.Live = true;
        end

        function obs = obs_(obj, o)
            obs = reshape(double(o), 1, []);
            if obj.IsDiscreteObs, obs = obs + 1; end
        end
    end
end
