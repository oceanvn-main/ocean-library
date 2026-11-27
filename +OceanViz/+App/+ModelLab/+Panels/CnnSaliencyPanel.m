classdef CnnSaliencyPanel < OceanViz.App.ModelLab.Panels.CnnPanelBase
    % CNNSALIENCYPANEL  Gradient of the class score with respect to the input pixels.
    %   (Former CNNInterpretabilityStudio tab "6. Saliency"; same module and renderer.)

    properties (SetAccess = private)
        Class
        Layer
    end

    methods
        function t = title(~), t = "6. Saliency"; end

        function build(obj, parent, ctx)
            obj.setup(parent, ctx);
            obj.Class = obj.Side.number('Target class (1-based):', 1, [1 Inf]);
            obj.Layer = obj.layerDropdown('Output layer (prefer pre-softmax):', {}, obj.Ctx.Model.logitsLayer());
            obj.runButton(@() obj.run());
        end

        function run(obj)
            img = obj.input();
            r = OceanDL.Interpretability.SaliencyMap.compute(obj.engine(), img, round(obj.Class.Value), 'OutputLayer', obj.Layer.Value);
            OceanViz.Strategies.CNN.SaliencyOverlayStrategy().plot(obj.Ax, r, struct('Image', img));
        end
    end
end
