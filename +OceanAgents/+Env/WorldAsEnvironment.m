classdef WorldAsEnvironment < OceanStats.RL.Env.IEnvironment & OceanStats.RL.Env.IModelEnvironment
    % WORLDASENVIRONMENT  Reverse bridge: a door-built single-agent grid world presented exactly like a native
    % OceanStats.RL.Env.TabularEnvironment, so every existing solver, learner, wrapper and test can run on it unchanged.
    %
    %   gw  = OceanAgents.World.Doors.GridWorlds.cliffWalking();          % a GridWorlds factory struct
    %   env = OceanAgents.Env.WorldAsEnvironment(gw);
    %   s = env.reset('Seed', 7);                                         % AgentWorld.reset(7)
    %   [s2, r, terminated, truncated, info] = env.step(2);               % one AgentWorld.step = one World.advance
    %   [next, prob, reward] = env.get_transitions(37, 2);                % the factory's exact model (gw.Model.Fn)
    %   wrapped = OceanStats.RL.Env.Wrappers.TimeLimit(env, 20);          % or GridWorlds...('TimeLimit', 20)
    %
    % Seeding (native semantics): setSeed(s) makes the NEXT reset reseed the world with s; reset('Seed', s) does both.
    %   A reset with no pending seed continues the world's door streams (AgentWorld.reset([])), exactly as a native
    %   reset() continues the environment's stream. A reset before any seed raises OceanAgents:World:NoSeed (a world
    %   has no global-stream fallback).
    % Outputs: obs = the native state index (row-major cell index); info.action_mask = the native ValidMask row of the
    %   NEW state (all false at a terminal; after a truncation the successor's mask, as natively).
    % Refusals (nothing changes): step before reset, from a terminal state, after an ended episode, or with an action
    %   outside the ActionSpace raises the native IDs OceanStats:RL:Env:TabularEnvironment:NotReset / InvalidAction.
    % Model and enumeration come from the factory's Model (States, Terminal, ValidMask, Fn), identical to the native
    %   tables; NumStates, NumActions, TerminalMask and ValidMask mirror TabularEnvironment's properties.
    % set_state is not supported (OceanAgents:Env:NotSupported). unwrapped() is the bridge itself.
    % 3 pillars: getMetadata / getPipelineSpec (NONVIRTUAL, IsSymbolic = false).
    % Contract: Planning/System/Game Theory/W2_Grid_Doors_Contract.md section 5.

    properties (SetAccess = private)
        Name string = ""
        NumStates double = 0
        NumActions double = 0
        TerminalMask (:,1) logical
        ValidMask logical = logical([])
    end

    properties (Access = private)
        World_                          % OceanAgents.World.AgentWorld (one agent)
        ModelFn_                        % [next, prob, reward] = ModelFn_(s, a)
        AgentID_ string = "agent1"
        PendingSeed_ double = []        % seed for the next reset ([] = continue the streams)
        Frame_ = []                     % current DecisionFrame
        Current_ double = 0             % current state index
        Live_ logical = false           % an episode is running and has not ended
    end

    methods
        function obj = WorldAsEnvironment(gridWorld)
            arguments
                gridWorld (1,1) struct
            end
            need = {'Doors', 'Agency', 'Model', 'ActionLabels'};
            if ~all(isfield(gridWorld, need))
                error('OceanAgents:Env:NotSupported', ...
                    'WorldAsEnvironment needs a GridWorlds factory struct (Doors, Agency, Model, ActionLabels).');
            end
            name = "world";
            if isfield(gridWorld, 'Name'), name = string(gridWorld.Name); end
            m = gridWorld.Model;
            labels = reshape(gridWorld.ActionLabels, 1, []);
            if ~isequal(labels, 1:numel(labels)) || ~isequal(m.States(:), (1:numel(m.States))')
                error('OceanAgents:Env:NotSupported', ...
                    'WorldAsEnvironment presents states 1..S and action labels 1..A (a native tabular layout).');
            end
            obj.Name = name;
            obj.World_ = OceanAgents.World.AgentWorld(gridWorld.Doors, gridWorld.Agency, 'Name', name);
            if numel(obj.World_.Roster) ~= 1
                error('OceanAgents:Env:NotSupported', 'WorldAsEnvironment presents exactly one agent.');
            end
            obj.AgentID_ = obj.World_.Roster(1);
            obj.ModelFn_ = m.Fn;
            obj.NumStates = numel(m.States);
            obj.NumActions = numel(labels);
            obj.TerminalMask = logical(m.Terminal(:));
            obj.ValidMask = logical(m.ValidMask);
            obj.ObservationSpace = OceanStats.RL.Env.Spaces.Discrete(obj.NumStates);
            obj.ActionSpace = OceanStats.RL.Env.Spaces.Discrete(obj.NumActions);
        end

        function setSeed(obj, seed)
            % SETSEED  The next reset reseeds the world with seed.
            arguments
                obj
                seed (1,1) double {mustBeInteger, mustBeNonnegative}
            end
            obj.PendingSeed_ = seed;
        end

        function [obs, reward, terminated, truncated, info] = step(obj, action)
            if isempty(obj.Frame_)
                error('OceanStats:RL:Env:TabularEnvironment:NotReset', 'Call reset() before step().');
            end
            s = obj.Current_;
            if ~obj.Live_ || ~obj.ActionSpace.contains(action) || ~obj.ValidMask(s, action)
                error('OceanStats:RL:Env:TabularEnvironment:InvalidAction', ...
                    'Action %s is not admissible in state %d.', mat2str(action), s);
            end
            f = obj.Frame_;
            decision = OceanAgents.Core.Decision('AgentID', obj.AgentID_, 'EpisodeID', f.EpisodeID, ...
                'FrameID', f.FrameID, 'Action', action);
            f = obj.World_.step(OceanAgents.Core.ActionBatch('EpisodeID', f.EpisodeID, 'FrameID', f.FrameID, ...
                'Decisions', decision));
            obj.Frame_ = f;
            obs = f.Views{1}.Observation;
            reward = f.Rewards(1);
            terminated = f.Terminated(1);
            truncated = f.Truncated(1);
            obj.Current_ = obs;
            obj.Live_ = ~(terminated || truncated);
            info = struct('action_mask', obj.ValidMask(obs, :));
        end

        function set_state(~, ~)
            error('OceanAgents:Env:NotSupported', ...
                'WorldAsEnvironment does not support set_state: a world''s state is changed only by its doors.');
        end

        % ---- IModelEnvironment / IEnumerableEnvironment (the factory's exact model)
        function [next_states, probabilities, immediate_rewards] = get_transitions(obj, state, action)
            [next_states, probabilities, immediate_rewards] = obj.ModelFn_(state(1), action);
        end
        function actions = get_valid_actions(obj, state), actions = find(obj.ValidMask(state(1), :)); end
        function tf = is_terminal(obj, state), tf = obj.TerminalMask(state(1)); end
        function states = get_all_states(obj), states = (1:obj.NumStates)'; end
        function idx = state_to_index(~, state), idx = state(1); end
        function k = action_to_index(~, action), k = action; end

        function meta = getMetadata(obj)
            meta = getMetadata@OceanStats.RL.Env.IEnvironment(obj);
            meta.Assumptions = [meta.Assumptions, {sprintf(['door-built world "%s" behind the native tabular ' ...
                'interface: %d states, %d actions; exact model from the factory'], obj.Name, obj.NumStates, obj.NumActions)}];
            meta.Reference = "reverse bridge World -> IEnvironment (W2 contract section 5)";
        end
    end

    methods (Access = protected)
        function [obs, info] = doReset_(obj)
            % A pending seed reseeds the world; otherwise its door streams continue (native reset() semantics).
            seed = obj.PendingSeed_;
            obj.PendingSeed_ = [];
            obj.Live_ = false;
            obj.Frame_ = obj.World_.reset(seed);
            obs = obj.Frame_.Views{1}.Observation;
            obj.Current_ = obs;
            obj.Live_ = true;
            info = struct('action_mask', obj.ValidMask(obs, :));
        end
    end
end
