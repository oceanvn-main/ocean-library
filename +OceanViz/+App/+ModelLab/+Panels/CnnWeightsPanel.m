classdef CnnWeightsPanel < OceanViz.App.ModelLab.Panels.CnnPanelBase
    % CNNWEIGHTSPANEL  Conv-layer filters as images (WeightVisualizer).
    %   (Former CNNInterpretabilityStudio tab "1. Weights"; same module and renderer.)

    properties (SetAccess = private)
        Layer
    end

    methods
        function t = title(~), t = "1. Weights"; end

        function build(obj, parent, ctx)
            obj.setup(parent, ctx);
            obj.Layer = obj.layerDropdown('Conv layer:', {'conv2d'});
            obj.runButton(@() obj.run());
        end

        function run(obj)
            conv = obj.engine().getLayerObject(obj.Layer.Value);
            data = struct('Tiles', {OceanDL.Interpretability.WeightVisualizer.extractTiles(conv)});
            OceanViz.Strategies.CNN.FilterGridStrategy().plot(obj.Ax, data, struct());
        end
    end
end
