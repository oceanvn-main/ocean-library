classdef DiffusionHandle < OceanViz.App.ModelLab.Models.IModelHandle
    % DIFFUSIONHANDLE  A diffusion model for the Model Lab: either a trained
    %   OceanDL.Generative.Diffusion.DiffusionModel (from a ModelCard) or an
    %   "ideal" model -- the exact EmpiricalDenoiser of a dataset (one per
    %   class for conditional use), which needs no training.
    %
    %   predictor(labels, guidance) -> @(x, t) [x_phi, eps_phi]   for any sampler
    %   defaultSampler()            -> DDPM ancestral with all L steps for LinearVP

    properties (SetAccess = private)
        Model = []              % DiffusionModel, or [] for an ideal model
        Schedule
        Denoisers = {}          % ideal model: {all, class 1, ..., class K}
    end

    methods
        function obj = DiffusionHandle(schedule, opts)
            arguments
                schedule
                opts.Model = []
                opts.Denoisers = {}
                opts.ImageSize (1,3) double
                opts.DataRange (1,2) double = [-1 1]
                opts.ClassNames (1,:) string = strings(1, 0)
                opts.Name (1,1) string = "Diffusion model"
                opts.Meta struct = struct()
            end
            obj.Schedule = schedule; obj.Model = opts.Model; obj.Denoisers = opts.Denoisers;
            obj.ImageSize = opts.ImageSize; obj.DataRange = opts.DataRange;
            obj.ClassNames = opts.ClassNames; obj.Name = opts.Name; obj.Meta = opts.Meta;
            obj.Capabilities = ["diffusion", "noise-schedule", "sample"];
            if obj.numClasses() > 0, obj.Capabilities(end+1) = "conditional"; end
        end

        function k = numClasses(obj)
            if ~isempty(obj.Model), k = obj.Model.NumClasses; else, k = max(0, numel(obj.Denoisers) - 1); end
            if k > 0 && isempty(obj.ClassNames), obj.ClassNames = "class " + string(1:k); end
        end

        function tf = isIdeal(obj)
            tf = isempty(obj.Model);
        end

        function fn = predictor(obj, labels, guidance)
            % labels: [] (unconditional) or one class per sample; guidance w (trained models).
            if nargin < 2, labels = []; end
            if nargin < 3, guidance = 1; end
            if ~obj.isIdeal()
                fn = obj.Model.predictor('Labels', labels, 'Guidance', guidance);
            elseif isempty(labels)
                fn = obj.Denoisers{1}.handle();
            else
                labels = labels(:);
                if all(labels == labels(1))
                    fn = obj.Denoisers{labels(1) + 1}.handle();
                else                                        % mixed classes: per-sample denoiser
                    fn = @(x, t) OceanViz.App.ModelLab.Models.DiffusionHandle.perSample(obj.Denoisers, labels, x, t);
                end
            end
        end

        function smp = defaultSampler(obj)
            if isa(obj.Schedule, 'OceanDL.Generative.Diffusion.Schedules.LinearVP')
                smp = OceanDL.Generative.Diffusion.Samplers.DDPMAncestral('Steps', obj.Schedule.L);
            else
                smp = OceanDL.Generative.Diffusion.Samplers.DDIM('Steps', 50);
            end
        end

        function L = numLevels(obj)
            % Number of discrete steps for sliders (LinearVP: L; otherwise 100).
            if isa(obj.Schedule, 'OceanDL.Generative.Diffusion.Schedules.LinearVP'), L = obj.Schedule.L; else, L = 100; end
        end
    end

    methods (Static)
        function [x0, eps] = perSample(denoisers, labels, x, t)
            x0 = zeros(size(x), 'like', x); eps = x0;
            for c = unique(labels).'
                k = labels == c;
                [x0(:, :, :, k), eps(:, :, :, k)] = denoisers{c + 1}.predict(x(:, :, :, k), t);
            end
        end
    end
end
