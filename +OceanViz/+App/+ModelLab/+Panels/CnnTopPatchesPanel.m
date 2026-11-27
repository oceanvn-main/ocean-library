classdef CnnTopPatchesPanel < OceanViz.App.ModelLab.Panels.CnnPanelBase
    % CNNTOPPATCHESPANEL  Input patches that most activate a channel (current image + 5 random candidates).
    %   (Former CNNInterpretabilityStudio tab "3. Top patches"; same module and renderer.)

    properties (SetAccess = private)
        Layer
        Channel
        TopK
    end

    methods
        function t = title(~), t = "3. Top patches"; end

        function build(obj, parent, ctx)
            obj.setup(parent, ctx);
            obj.Layer = obj.layerDropdown('Conv layer:', {'conv2d'});
            obj.Channel = obj.Side.number('Channel:', 1, [1 Inf]);
            obj.TopK = obj.Side.number('Top K:', 6, [1 64]);
            obj.runButton(@() obj.run());
        end

        function run(obj)
            img = obj.input(); sz = obj.Ctx.Model.ImageSize;
            candidates = img;
            for k = 1:5, candidates = cat(4, candidates, rand(sz(1), sz(2), sz(3), 1)); end
            r = OceanDL.Interpretability.TopActivatingPatches.find(obj.engine(), candidates, obj.Layer.Value, ...
                round(obj.Channel.Value), 'TopK', round(obj.TopK.Value));
            OceanViz.Strategies.CNN.PatchGridStrategy().plot(obj.Ax, r, struct());
        end
    end
end
