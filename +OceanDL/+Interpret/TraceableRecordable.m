classdef TraceableRecordable < OceanReplay.Record.IRecordable
    % TRACEABLERECORDABLE  Generic Replay adapter for ANY network that implements
    %   OceanDL.Interpret.ITraceable (CNNs, VAE, hierarchical VAE, ...): records a
    %   set of input images and the activations of chosen layers, one point per
    %   image, for OceanViz.App.Replay.ReplayStudio. No family semantics -- model
    %   families add theirs in a subclass (e.g. OceanDL.Generative.Inspect.VAERecordable).
    %
    %   Axis "image": the images' own identifiers (e.g. test-set indices), in the
    %   in ascending order (images and labels are reordered to match). Channels:
    %     x               the input image                         image
    %     label           its class (when labels are given)       scalar
    %     act_<layer>     activation of <layer> (H x W x C)       image (one tile per channel)
    %   Display ranges of activations = [min max] of that layer over the given images
    %   (display only; stored values are untouched).
    %
    %   Example:
    %     r = OceanDL.Interpret.TraceableRecordable(model, X, 'Ids', idx, 'Labels', y, ...
    %         'Layers', ["enc.L1.block1" "dec.out"]);
    %     OceanReplay.Record.Recorder.run(r, spec, "acts.h5");

    properties (SetAccess = protected)
        Model
        X                                    % H x W x C x N (model's data range)
        Ids (1,:) double
        Labels = []
        ClassNames (1,:) string = strings(1, 0)
        Layers (1,:) string                  % traced layers recorded as act_<layer>
        DataRange (1,2) double
        ModelName (1,1) string
        DatasetName (1,1) string
        ActRange                             % layer -> [lo hi] (display)
        ActShape                             % layer -> [H W C]
    end

    methods
        function obj = TraceableRecordable(model, X, opts)
            arguments
                model (1,1) OceanDL.Interpret.ITraceable
                X {mustBeNumeric}
                opts.Ids (1,:) double = []
                opts.Labels = []
                opts.ClassNames (1,:) string = strings(1, 0)
                opts.Layers = []                         % [] = every traced layer (see defaultLayers)
                opts.DataRange (1,2) double = [0 1]
                opts.ModelName (1,1) string = ""
                opts.DatasetName (1,1) string = ""
            end
            N = size(X, 4);
            ids = opts.Ids; if isempty(ids), ids = 1:N; end
            if numel(ids) ~= N || numel(unique(ids)) ~= N
                error('OceanDL:Interpret:TraceableRecordable:ids', 'Ids must be %d unique values (one per image).', N);
            end
            if ~isempty(opts.Labels) && numel(opts.Labels) ~= N
                error('OceanDL:Interpret:TraceableRecordable:labels', 'Labels must have one entry per image.');
            end
            [ids, ord] = sort(ids);                              % the axis is monotone: images in id order
            X = X(:, :, :, ord);
            labels = opts.Labels(:).'; if ~isempty(labels), labels = labels(ord); end
            obj.Model = model; obj.X = X; obj.Ids = ids; obj.Labels = labels;
            obj.ClassNames = opts.ClassNames; obj.DataRange = opts.DataRange;
            obj.ModelName = opts.ModelName; obj.DatasetName = opts.DatasetName;
            % one pass over all images: layer names, shapes and display ranges
            [acts, order] = OceanDL.Autograd.GradMode.noGrad(@() model.trace(X));
            if isempty(opts.Layers), obj.Layers = obj.defaultLayers(order); else, obj.Layers = string(opts.Layers); end
            obj.ActRange = containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.ActShape = containers.Map('KeyType', 'char', 'ValueType', 'any');
            for L = obj.Layers
                if ~isKey(acts, char(L))
                    error('OceanDL:Interpret:TraceableRecordable:layer', 'Layer "%s" is not traced (available: %s).', L, strjoin(order, ', '));
                end
                a = double(gather(OceanDL.Autograd.Tensor.raw(acts(char(L)))));
                lo = min(a(:)); hi = max(a(:)); if hi <= lo, hi = lo + 1; end
                obj.ActRange(char(L)) = [lo hi];
                obj.ActShape(char(L)) = size(a, 1:3);
            end
        end

        function axes = axesInfo(obj)
            axes = OceanReplay.Core.Axis("image", obj.Ids, 'Symbol', "n", ...
                'Description', "identifier of the input image (e.g. its index in the test set)");
        end

        function [c, v] = catalog(obj)
            C = @OceanReplay.Core.ChannelInfo;
            c = C("x", "image", 'Axes', "image", 'ValueShape', size(obj.X, 1:3), 'DisplayRange', obj.DataRange, ...
                'Symbol', "x", 'SymbolLatex', '$x$', 'Description', "the input image");
            if ~isempty(obj.Labels)
                d = "class label of the image";
                if ~isempty(obj.ClassNames), d = d + " (" + strjoin(compose("%d = %s", (1:numel(obj.ClassNames)).', obj.ClassNames(:)), ", ") + ")"; end
                c(end+1) = C("label", "scalar", 'Axes', "image", 'Symbol', "y", 'SymbolLatex', '$y$', 'Description', d);
            end
            for L = obj.Layers
                c(end+1) = C(obj.actName(L), "image", 'Axes', "image", 'ValueShape', obj.ActShape(char(L)), ...
                    'DisplayRange', obj.ActRange(char(L)), 'Symbol', L, ...
                    'Description', "activation of layer " + L + " (H x W x C, one tile per channel); display range = " + ...
                    "min..max of this layer over the recorded images"); %#ok<AGROW>
            end
            v = OceanReplay.Core.ViewInfo.empty;
        end

        function out = run(obj, rec)
            N = numel(obj.Ids); out = zeros(1, N);
            for k = 1:N
                x = obj.X(:, :, :, k);
                cache = containers.Map('KeyType', 'char', 'ValueType', 'any');   % one trace per image, only if needed
                state = struct('x', @() double(gather(x)));
                if ~isempty(obj.Labels), lab = obj.Labels(k); state.label = @() double(lab); end
                for L = obj.Layers
                    layer = L;
                    state.(obj.actName(layer)) = @() obj.activation(cache, x, layer);
                end
                [state, out(k)] = obj.addState(state, x, k);
                rec.point(struct('image', obj.Ids(k)), state);
            end
        end

        function d = describe(obj)
            d = struct('model', struct('name', obj.ModelName, 'class', string(class(obj.Model)), 'dataset', obj.DatasetName, ...
                    'layers', obj.Layers), ...
                'references', {{'Activations recorded with OceanDL.Interpret.ITraceable.trace (Szeliski 2nd ed. Sec. 5.4)'}}, ...
                'notation', struct(), 'layout', "default");
        end
    end

    methods (Access = protected)
        function [state, value] = addState(~, state, ~, ~)
            % Hook for families: add providers for image k; value = one number for the run's output.
            value = 0;
        end

        function names = defaultLayers(~, order)
            names = order;
        end

        function a = activation(obj, cache, x, layer)
            if ~isKey(cache, 'acts')
                cache('acts') = OceanDL.Autograd.GradMode.noGrad(@() obj.Model.trace(x));
            end
            acts = cache('acts');
            a = double(gather(OceanDL.Autograd.Tensor.raw(acts(char(layer)))));
        end
    end

    methods (Static)
        function n = actName(layer)
            % Channel name of a traced layer (a valid identifier): "enc.L1.block1" -> "act_enc_L1_block1".
            n = "act_" + regexprep(string(layer), '[^A-Za-z0-9]', '_');
        end
    end
end
