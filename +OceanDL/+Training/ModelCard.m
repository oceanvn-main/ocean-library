classdef ModelCard
    % MODELCARD  A trained model saved WITH the recipe that builds it, so any
    %   tool (scripts, the Model Lab GUI) can rebuild it without knowing the
    %   training settings. A plain checkpoint stores only weights; loading
    %   those needs the exact constructor settings typed again.
    %
    %   RECIPE: struct('Class', "pkg.ClassName", 'Positional', {{...}}, 'Args', {{name, value, ...}})
    %     Positional / Args values may themselves be recipes (struct with a
    %     'Class' field) -- built first, recursively. So a DiffusionModel card
    %     holds the recipes of its U-Net and its noise schedule.
    %     r = ModelCard.recipe("OceanDL.Vision.UNet2D", {}, {'Widths', [32 64]})
    %
    %   ModelCard.save(file, model, recipe, 'Meta', meta)   % weights = model.stateDict()
    %   ModelCard.save(file, model, recipe, 'State', ema.Shadow)   % e.g. save EMA weights instead
    %   [model, card] = ModelCard.load(file)                % rebuilt, weights loaded, eval mode
    %   card.Meta: free-form (dataset, class names, data range, notes, training stats)
    %   card.DataType: "single" | "double" -- the model is cast before loading.

    properties (Constant)
        Version = 1
    end

    methods (Static)
        function r = recipe(className, positional, args)
            arguments
                className (1,1) string
                positional cell = {}
                args cell = {}
            end
            r = struct('Class', className, 'Positional', {positional}, 'Args', {args});
        end

        function save(file, model, recipe, opts)
            arguments
                file (1,1) string
                model OceanDL.nn.Module
                recipe struct
                opts.Meta struct = struct()
                opts.State = []
            end
            state = opts.State;
            if isempty(state), state = model.stateDict(); end
            state = OceanDL.Training.ModelCard.toHost(state);          % a card must load without a GPU
            card = struct('Version', OceanDL.Training.ModelCard.Version, 'Recipe', recipe, 'State', state, ...
                'DataType', model.dataType(), 'Meta', opts.Meta, 'Saved', string(datetime('now')));
            folder = fileparts(file);
            if strlength(folder) > 0 && ~isfolder(folder), mkdir(folder); end
            tmp = file + ".tmp";
            save(tmp, 'card', '-v7', '-mat');
            movefile(tmp, file, 'f');
        end

        function [model, card] = load(file)
            s = load(file, 'card');
            card = s.card;
            model = OceanDL.Training.ModelCard.build(card.Recipe);
            if ~isempty(card.DataType) && card.DataType ~= ""
                model.to('DataType', card.DataType);
            end
            model.loadStateDict(card.State);
            model.eval();
        end

        function obj = build(r)
            % Construct a recipe (recursively building nested recipes).
            B = @OceanDL.Training.ModelCard.buildValue;
            pos = cellfun(B, r.Positional, 'UniformOutput', false);
            args = r.Args;
            for k = 2:2:numel(args), args{k} = B(args{k}); end
            obj = feval(char(r.Class), pos{:}, args{:});
        end

        function v = toHost(v)
            % Copy of a state (containers.Map / struct / cell of arrays) with every
            % gpuArray gathered. Maps are COPIED -- the caller's (e.g. a live
            % ModelEMA.Shadow, a handle) is left on its device.
            H = @OceanDL.Training.ModelCard.toHost;
            if isa(v, 'containers.Map')
                k = keys(v);
                if isempty(k), v = containers.Map(); return; end
                v = containers.Map(k, cellfun(H, values(v, k), 'UniformOutput', false));
            elseif isstruct(v)
                f = fieldnames(v);
                for i = 1:numel(v)
                    for j = 1:numel(f), v(i).(f{j}) = H(v(i).(f{j})); end
                end
            elseif iscell(v)
                v = cellfun(H, v, 'UniformOutput', false);
            elseif isa(v, 'gpuArray')
                v = gather(v);
            end
        end

        function v = buildValue(v)
            if isstruct(v) && isscalar(v) && isfield(v, 'Class') && isfield(v, 'Positional')
                v = OceanDL.Training.ModelCard.build(v);
            end
        end
    end
end
