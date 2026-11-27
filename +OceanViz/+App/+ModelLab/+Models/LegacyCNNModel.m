classdef LegacyCNNModel < OceanViz.App.ModelLab.Models.IModelHandle
    % LEGACYCNNMODEL  A classifier on the legacy OceanDL.Conv.CNNEngine (the
    %   models of the former CNNInterpretabilityStudio), for the CNN panels.

    properties (SetAccess = private)
        Engine
        Spec
    end

    methods
        function obj = LegacyCNNModel(spec, weights, imageSize, name)
            if nargin < 3 || isempty(imageSize), imageSize = [32 32]; end
            if nargin < 4, name = "CNN"; end
            obj.Spec = spec;
            obj.Engine = OceanDL.Conv.CNNEngine(spec, weights);
            obj.ImageSize = [imageSize(1:2), spec.InputChannels];
            obj.Name = name;
            obj.Capabilities = ["cnn-legacy", "classify"];
        end

        function img = prepare(obj, img01)
            % (H x W x C) -> (H x W x C x 1) and a forward pass to fill the caches.
            img = reshape(img01, obj.ImageSize(1), obj.ImageSize(2), obj.ImageSize(3), 1);
            obj.Engine.forward(img);
        end

        function names = layers(obj, types)
            names = obj.Spec.LayerNames;
            if nargin > 1
                t = cellfun(@(s) s.Type, obj.Spec.LayerSpecs, 'UniformOutput', false);
                names = names(ismember(t, types));
            end
        end

        function name = logitsLayer(obj)
            names = obj.Spec.LayerNames;
            t = cellfun(@(s) s.Type, obj.Spec.LayerSpecs, 'UniformOutput', false);
            if strcmp(t{end}, 'softmax') && numel(names) >= 2, name = names{end-1}; else, name = names{end}; end
        end
    end
end
