classdef LabelledCounterEnv < OceanStats.RL.Env.IEnvironment
    % LABELLEDCOUNTERENV  Test fixture: a deterministic native environment with a Discrete(3, 'Start', 0) action space,
    % so action LABELS (0, 1, 2) differ from table COLUMNS (1, 2, 3). Used to prove the bridge passes labels, not columns.
    %
    %   env = OceanTest.OceanAgents.Fixtures.Env.LabelledCounterEnv('Horizon', 3);
    %   [obs, info] = env.reset('Seed', 0);        obs = 0
    %   [obs, r, term, trunc, info] = env.step(2); obs = 1, r = 2 (the label received), term after Horizon steps
    %
    % Dynamics: obs = number of steps taken; reward = the action label received; terminated when obs == Horizon.
    % Mask (changes every step): even obs -> [true false true]; odd obs -> [false true true]; terminal -> all false.
    % An illegal label errors (like TabularEnvironment). info also carries 'secret' and 'debug' fields so the bridge's
    % InfoAllow allowlist can be tested. 'ProvideMask', false omits info.action_mask (missing-mask default rule).
    % Public counters StepCalls / LastAction let a test prove that a rejected decision never reached step().

    properties (SetAccess = private)
        Horizon double = 3
        ProvideMask logical = true
        StepCalls double = 0
        LastAction double = []
        Count double = 0
        SeedSeen double = []
    end

    methods
        function obj = LabelledCounterEnv(options)
            arguments
                options.Horizon (1,1) double {mustBeInteger, mustBePositive} = 3
                options.ProvideMask (1,1) logical = true
            end
            obj.Horizon = options.Horizon;
            obj.ProvideMask = options.ProvideMask;
            obj.ObservationSpace = OceanStats.RL.Env.Spaces.Discrete(options.Horizon + 1, 'Start', 0);
            obj.ActionSpace = OceanStats.RL.Env.Spaces.Discrete(3, 'Start', 0);
        end

        function [obs, reward, terminated, truncated, info] = step(obj, action)
            obj.StepCalls = obj.StepCalls + 1;
            if ~any(obj.legalLabels_(obj.Count) == action)
                error('OceanTest:Fixtures:IllegalAction', 'Label %g is not legal at obs %d.', action, obj.Count);
            end
            obj.LastAction = action;
            obj.Count = obj.Count + 1;
            obs = obj.Count;
            reward = action;
            terminated = obj.Count >= obj.Horizon;
            truncated = false;
            info = obj.info_();
        end

        function setSeed(obj, seed)
            obj.SeedSeen = seed;
        end

        function m = maskAt(obj, count)
            % MASKAT  The mask this fixture reports at observation count.
            if count >= obj.Horizon
                m = false(1, 3);
            elseif mod(count, 2) == 0
                m = [true false true];
            else
                m = [false true true];
            end
        end
    end

    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            obj.Count = 0;
            obs = 0;
            info = obj.info_();
        end
    end

    methods (Access = private)
        function labels = legalLabels_(obj, count)
            labels = obj.ActionSpace.Start + find(obj.maskAt(count)) - 1;
        end

        function info = info_(obj)
            info = struct('secret', 42, 'debug', "internal");
            if obj.ProvideMask
                info.action_mask = obj.maskAt(obj.Count);
            end
        end
    end
end
