classdef CnnActivationPanel < OceanViz.App.ModelLab.Panels.CnnPanelBase
    % CNNACTIVATIONPANEL  One channel's activation map for the current image.
    %   (Former CNNInterpretabilityStudio tab "2. Activation map"; same module and renderer.)

    properties (SetAccess = private)
        Layer
        Channel
    end

    methods
        function t = title(~), t = "2. Activation map"; end

        function build(obj, parent, ctx)
            obj.setup(parent, ctx);
            obj.Layer = obj.layerDropdown('Layer:', {});
            obj.Channel = obj.Side.number('Channel:', 1, [1 Inf]);
            obj.runButton(@() obj.run());
        end

        function run(obj)
            img = obj.input();
            maps = OceanDL.Interpretability.ActivationMapExtractor.extract(obj.engine(), img, obj.Layer.Value);
            ch = min(max(1, round(obj.Channel.Value)), size(maps, 3));
            a = OceanDL.Interpretability.ActivationMapExtractor.channelToImage(maps, ch, 1);
            OceanViz.Strategies.CNN.ActivationMapStrategy().plot(obj.Ax, a, struct('Title', sprintf('%s / channel %d', obj.Layer.Value, ch)));
        end
    end
end
