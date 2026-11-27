classdef ModelFactory
    % MODELFACTORY  Build Model Lab model handles from what a user can pick:
    %     fromCard(file)          any OceanDL.Training.ModelCard (.mat)
    %     idealDDPM(dataset)      exact ideal DDPM of a dataset (no training)
    %     dummyCNN()              random-weight legacy CNN (interpretability demo)
    %     legacyCNN(jsonBase)     a legacy OceanDL.Conv.Checkpoint
    %     builtins()              names offered in the app's model menu

    methods (Static)
        function names = builtins()
            names = ["Ideal DDPM - shapes 16x16 (no training)", "Ideal DDPM - USPS digits (no training)", ...
                "Dummy CNN (random weights)"];
        end

        function h = builtin(name)
            F = @OceanViz.App.ModelLab.Models.ModelFactory.idealDDPM;
            switch string(name)
                case "Ideal DDPM - shapes 16x16 (no training)", h = F("shapes16");
                case "Ideal DDPM - USPS digits (no training)",  h = F("usps");
                case "Dummy CNN (random weights)",              h = OceanViz.App.ModelLab.Models.ModelFactory.dummyCNN();
                otherwise, error('OceanViz:App:ModelLab:ModelFactory:Unknown', 'Unknown built-in model "%s".', name);
            end
        end

        function h = fromCard(file)
            [model, card] = OceanDL.Training.ModelCard.load(file);
            meta = card.Meta;
            [~, base] = fileparts(file);
            names = strings(1, 0); if isfield(meta, 'ClassNames'), names = string(meta.ClassNames); end
            if isa(model, 'OceanDL.Generative.Diffusion.DiffusionModel')
                range = [-1 1]; if isfield(meta, 'DataRange'), range = meta.DataRange; end
                h = OceanViz.App.ModelLab.Models.DiffusionHandle(model.Schedule, 'Model', model, ...
                    'ImageSize', meta.ImageSize, 'DataRange', range, 'ClassNames', names, ...
                    'Name', string(base), 'Meta', meta);
            elseif isa(model, 'OceanDL.Generative.VAE') || isa(model, 'OceanDL.Generative.HierarchicalVAE')
                h = OceanViz.App.ModelLab.Models.VAEHandle(model, 'ImageSize', meta.ImageSize, ...
                    'Name', string(base), 'Meta', meta, 'ClassNames', names);
            else
                error('OceanViz:App:ModelLab:ModelFactory:Unsupported', 'No Model Lab handle for %s.', class(model));
            end
        end

        function h = idealDDPM(datasetName)
            D = OceanViz.App.ModelLab.Datasets.get(datasetName);
            sch = OceanDL.Generative.Diffusion.Schedules.LinearVP();
            X = 2 * double(D.X) - 1;                                   % [0, 1] -> [-1, 1]
            den = {OceanDL.Generative.Diffusion.EmpiricalDenoiser(X, sch)};
            for c = 1:numel(D.ClassNames)
                den{end+1} = OceanDL.Generative.Diffusion.EmpiricalDenoiser(X(:, :, :, D.y == c), sch); %#ok<AGROW>
            end
            meta = struct('Dataset', D.Name, 'Book', "Lai et al. Sec. 2.2 (exact ideal denoiser)");
            h = OceanViz.App.ModelLab.Models.DiffusionHandle(sch, 'Denoisers', den, 'ImageSize', size(D.X, 1:3), ...
                'DataRange', [-1 1], 'ClassNames', D.ClassNames, 'Name', "Ideal DDPM (" + D.Name + ")", 'Meta', meta);
        end

        function h = dummyCNN()
            [spec, weights, imageSize] = OceanDL.Conv.Checkpoint.makeDummyCheckpoint();
            h = OceanViz.App.ModelLab.Models.LegacyCNNModel(spec, weights, imageSize, "Dummy CNN (random weights)");
        end

        function h = legacyCNN(jsonBase)
            [spec, weights, imageSize] = OceanDL.Conv.Checkpoint.load(jsonBase);
            [~, name] = fileparts(jsonBase);
            h = OceanViz.App.ModelLab.Models.LegacyCNNModel(spec, weights, imageSize, string(name));
        end
    end
end
