classdef CnnOcclusionPanel < OceanViz.App.ModelLab.Panels.CnnPanelBase
    % CNNOCCLUSIONPANEL  How the class score drops when a patch is hidden.
    %   (Former CNNInterpretabilityStudio tab "5. Occlusion"; same module and renderer.)

    properties (SetAccess = private)
        Class
        Patch
        Stride
    end

    methods
        function t = title(~), t = "5. Occlusion"; end

        function build(obj, parent, ctx)
            obj.setup(parent, ctx);
            obj.Class = obj.Side.number('Target class (1-based):', 1, [1 Inf]);
            obj.Patch = obj.Side.number('Patch size (px):', 6, [1 Inf]);
            obj.Stride = obj.Side.number('Stride (px):', 3, [1 Inf]);
            obj.runButton(@() obj.run());
        end

        function run(obj)
            img = obj.input(); ps = round(obj.Patch.Value); st = round(obj.Stride.Value);
            r = OceanDL.Interpretability.OcclusionSensitivity.run(obj.engine(), img, round(obj.Class.Value), ...
                'PatchSize', [ps ps], 'Stride', [st st]);
            OceanViz.Strategies.CNN.OcclusionHeatmapStrategy().plot(obj.Ax, r, struct('Image', img));
        end
    end
end
