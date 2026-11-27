classdef CNNInterpretabilityStudio < handle
    % CNNINTERPRETABILITYSTUDIO  Backward-compatible launcher. The 8 CNN
    %   interpretability tabs (Weights, Activation map, Top patches, Deconv,
    %   Occlusion, Saliency, Grad-CAM, Adversarial) now live in the Model Lab
    %   (OceanViz.App.ModelLab), which also hosts the diffusion and VAE panels.
    %
    %   USAGE (unchanged):
    %     studio = CNNInterpretabilityStudio();       % dummy random-weight CNN
    %     studio.Lab                                  % the OceanViz.App.ModelLab.Shell
    %
    %   Equivalent:
    %     lab = OceanViz.App.ModelLab.launch("Dummy CNN (random weights)");
    %
    % See also: OceanViz.App.ModelLab.launch, OceanViz.App.ModelLab.PanelRegistry

    properties (SetAccess = private)
        Lab             % OceanViz.App.ModelLab.Shell
    end

    methods
        function obj = CNNInterpretabilityStudio(varargin)
            obj.Lab = OceanViz.App.ModelLab.launch("Dummy CNN (random weights)", varargin{:});
        end

        function show(obj)
            figure(obj.Lab.Figure);
        end
    end
end
