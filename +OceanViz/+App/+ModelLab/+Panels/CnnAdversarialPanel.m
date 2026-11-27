classdef CnnAdversarialPanel < OceanViz.App.ModelLab.Panels.CnnPanelBase
    % CNNADVERSARIALPANEL  FGSM: the smallest-sign perturbation that changes the prediction.
    %   (Former CNNInterpretabilityStudio tab "8. Adversarial"; same module and renderer.)

    properties (SetAccess = private)
        Class
        Epsilon
    end

    methods
        function t = title(~), t = "8. Adversarial"; end

        function build(obj, parent, ctx)
            obj.setup(parent, ctx);
            obj.Class = obj.Side.number('True class (1-based):', 1, [1 Inf]);
            obj.Epsilon = obj.Side.number('Epsilon:', 0.03, [0 1]);
            obj.runButton(@() obj.run());
        end

        function run(obj)
            img = obj.input();
            r = OceanDL.Interpretability.AdversarialPerturbation.fgsm(obj.engine(), img, round(obj.Class.Value), obj.Epsilon.Value);
            OceanViz.Strategies.CNN.AdversarialCompareStrategy().plot(obj.Ax, r, struct('Image', img));
        end
    end
end
