classdef WorldModelSearchModel < OceanAlgo.StateSearch.IDistributionModel & OceanAlgo.StateSearch.ISampleModel
    % WORLDMODELSEARCHMODEL  Adapter from an RL model to the planning-model contracts of OceanAlgo.StateSearch,
    % so every search engine (A*, IDA*, beam, depth-limited search, rollout, MCTS) can plan on it.
    %
    %   pm = OceanStats.RL.DecisionTime.WorldModelSearchModel(model)
    %     model   OceanStats.RL.Dyna.Models.IWorldModel (learned or exact, indexed states/actions), or
    %             OceanStats.RL.Env.IModelEnvironment (wrapped in ExactModelAdapter, as RTDPSolver does)
    %
    % Mapping: state = state index (also its key); actions(s) = admissible action indices
    % (ValidActionMask row); outcomes(s,a) = model.distribution(s,a) with NextStateIndex -> Next;
    % sample(s,a,stream) = model.sample(s,a,stream). A learned (non-read-only) model only knows the pairs it has
    % observed: actions(s) then lists only observed admissible actions (model.contains), so a state with none is
    % a leaf valued by the engine's LeafValue -- the search never asks the model about an unknown pair.
    %
    % Caching: a read-only model (isReadOnly(), e.g. ExactModelAdapter) never changes, so its distributions are
    % cached per (s, a) on first use and samples are drawn from the cache by the same inverse-CDF rule the
    % exact adapter uses (one uniform draw: first k with u <= cumsum(p)) -- identical draws, far fewer calls.
    % A learnable model is always queried live, so planning sees its latest update.
    properties (SetAccess = immutable)
        WorldModel      % OceanStats.RL.Dyna.Models.IWorldModel
        Cached (1,1) logical
    end
    properties (Access = private)
        Cache_ cell
    end
    methods
        function obj = WorldModelSearchModel(model)
            if isa(model, 'OceanStats.RL.Env.IModelEnvironment')
                model = OceanStats.RL.Dyna.Models.ExactModelAdapter(model);
            elseif ~isa(model, 'OceanStats.RL.Dyna.Models.IWorldModel') || ~isscalar(model)
                error('OceanStats:RL:DecisionTime:BadModel', 'Model must be an IModelEnvironment or an IWorldModel.');
            end
            obj.WorldModel = model;
            obj.Cached = model.isReadOnly();
            if obj.Cached, obj.Cache_ = cell(model.NumStates, model.NumActions); end
        end
        function a = actions(obj, s)
            % Admissible actions; for a learnable model only those it has observed (model.contains), so a
            % search plans over what the model actually knows. A state with no observed action is a leaf.
            a = find(obj.WorldModel.ValidActionMask(s, :));
            if ~obj.Cached
                a = a(arrayfun(@(x) obj.WorldModel.contains(s, x), a));
            end
        end
        function k = key(~, s), k = s; end
        function o = outcomes(obj, s, a)
            if obj.Cached && ~isempty(obj.Cache_{s, a}), o = obj.Cache_{s, a}; return; end
            d = obj.WorldModel.distribution(s, a);
            o = struct('Next', {d.NextStateIndex}, 'Reward', {d.Reward}, 'Terminated', {d.Terminated}, ...
                'Probability', {d.Probability});
            o = o(:);
            if obj.Cached, obj.Cache_{s, a} = o; end
        end
        function o = sample(obj, s, a, stream)
            if obj.Cached
                d = obj.outcomes(s, a);
                k = find(rand(stream) <= cumsum([d.Probability]), 1);
                if isempty(k), k = numel(d); end
                o = struct('Next', d(k).Next, 'Reward', d(k).Reward, 'Terminated', logical(d(k).Terminated));
                return
            end
            x = obj.WorldModel.sample(s, a, stream);
            o = struct('Next', x.NextStateIndex, 'Reward', x.Reward, 'Terminated', logical(x.Terminated));
        end
        function n = numStates(obj), n = obj.WorldModel.NumStates; end
    end
end
