classdef PanelRegistry
    % PANELREGISTRY  The list of Model Lab panels, in tab order. Adding a
    %   feature = writing one IPanel class and listing its constructor here
    %   (or calling register() at run time, e.g. from an experiment script).

    methods (Static)
        function ctors = all()
            P = 'OceanViz.App.ModelLab.Panels.';
            % generative models (Lai et al. Ch. 2)
            generative = {'ForwardProcessPanel', 'ReverseProcessPanel', 'GeneratePanel', 'TrainingStepPanel', ...
                'VAEExplorerPanel', 'InsideVAEPanel'};
            % CNN interpretability (the former CNNInterpretabilityStudio tabs)
            cnn = {'CnnWeightsPanel', 'CnnActivationPanel', 'CnnTopPatchesPanel', 'CnnDeconvPanel', ...
                'CnnOcclusionPanel', 'CnnSaliencyPanel', 'CnnGradCAMPanel', 'CnnAdversarialPanel'};
            names = [generative, cnn];
            ctors = cellfun(@(n) str2func([P n]), names, 'UniformOutput', false);
            ctors = [ctors, OceanViz.App.ModelLab.PanelRegistry.extra()];
        end

        function register(ctor)
            % Add a panel constructor for the rest of the MATLAB session.
            OceanViz.App.ModelLab.PanelRegistry.extra(ctor);
        end
    end

    methods (Static, Access = private)
        function list = extra(add)
            persistent registered
            if isempty(registered), registered = {}; end
            if nargin > 0, registered{end+1} = add; end
            list = registered;
        end
    end
end
