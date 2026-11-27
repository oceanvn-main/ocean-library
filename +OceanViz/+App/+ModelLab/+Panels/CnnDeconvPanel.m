classdef CnnDeconvPanel < OceanViz.App.ModelLab.Panels.CnnPanelBase
    % CNNDECONVPANEL  Zeiler & Fergus deconvolution of one channel back to pixel space.
    %   (Former CNNInterpretabilityStudio tab "4. Deconv"; same module and renderer.)

    properties (SetAccess = private)
        Layer
        Channel
    end

    methods
        function t = title(~), t = "4. Deconv"; end

        function build(obj, parent, ctx)
            obj.setup(parent, ctx);
            obj.Layer = obj.layerDropdown('Layer (ReLU-only path):', {'conv2d', 'relu', 'maxpool2d', 'avgpool2d'});
            obj.Channel = obj.Side.number('Channel:', 1, [1 Inf]);
            obj.runButton(@() obj.run());
        end

        function run(obj)
            img = obj.input();
            r = OceanDL.Interpretability.DeconvVisualizer.reconstruct(obj.engine(), img, obj.Layer.Value, round(obj.Channel.Value));
            OceanViz.Strategies.CNN.DeconvStrategy().plot(obj.Ax, r, struct());
        end
    end
end
