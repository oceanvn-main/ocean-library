classdef CnnGradCAMPanel < OceanViz.App.ModelLab.Panels.CnnPanelBase
    % CNNGRADCAMPANEL  Class-activation map from a conv layer's gradients.
    %   (Former CNNInterpretabilityStudio tab "7. Grad-CAM"; same module and renderer.)

    properties (SetAccess = private)
        Class
        Layer
        OutLayer
    end

    methods
        function t = title(~), t = "7. Grad-CAM"; end

        function build(obj, parent, ctx)
            obj.setup(parent, ctx);
            obj.Class = obj.Side.number('Target class (1-based):', 1, [1 Inf]);
            convs = obj.Ctx.Model.layers({'conv2d'});
            obj.Layer = obj.layerDropdown('Conv layer:', {'conv2d'}, convs{end});
            obj.OutLayer = obj.layerDropdown('Output layer:', {}, obj.Ctx.Model.logitsLayer());
            obj.runButton(@() obj.run());
        end

        function run(obj)
            img = obj.input();
            r = OceanDL.Interpretability.GradCAM.compute(obj.engine(), img, round(obj.Class.Value), obj.Layer.Value, ...
                'OutputLayer', obj.OutLayer.Value);
            OceanViz.Strategies.CNN.GradCAMOverlayStrategy().plot(obj.Ax, r, struct('Image', img));
        end
    end
end
