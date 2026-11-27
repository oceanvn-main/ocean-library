classdef Datasets
    % DATASETS  Named datasets for the Model Lab (display range [0, 1],
    %   "SSCB"), cached for the session. A model card's Meta.Dataset names the
    %   one its model was trained on, so panels can offer its images.
    %     "shapes16" / "shapes32"   ComplexShapeImages (train seed 1, test seed 2)
    %     "usps"                    USPS handwritten digits 16 x 16
    %     "affectnet16/32/48"       AffectNet faces, clean 6-class protocol, grayscale

    methods (Static)
        function D = get(name)
            persistent cache
            if isempty(cache), cache = containers.Map(); end
            name = char(name);
            if isKey(cache, name), D = cache(name); return; end
            switch name
                case {'shapes16', 'shapes32'}
                    G = OceanData.Generators.Shapes2D.ComplexShapeImages;
                    sz = sscanf(name, 'shapes%d');
                    [Xtr, ytr] = G.generate(4000, 'Size', sz, 'Seed', 1);
                    [Xte, yte] = G.generate(600, 'Size', sz, 'Seed', 2);
                    D = struct('Name', string(name), 'X', single(Xtr), 'y', ytr, 'Xte', single(Xte), 'yte', yte, ...
                        'ClassNames', G.ClassNames);
                case 'usps'
                    [Xtr, ytr] = OceanViz.App.ModelLab.Datasets.usps("train");
                    [Xte, yte] = OceanViz.App.ModelLab.Datasets.usps("test");
                    D = struct('Name', "usps", 'X', single(Xtr), 'y', ytr, 'Xte', single(Xte), 'yte', yte, ...
                        'ClassNames', string(0:9));
                case {'affectnet16', 'affectnet32', 'affectnet48'}
                    % Clean 6-class AffectNet protocol, grayscale (the face DDPM's data).
                    sz = sscanf(name, 'affectnet%d');
                    D = OceanViz.App.ModelLab.Datasets.affectnet(sz);
                otherwise
                    error('OceanViz:App:ModelLab:Datasets:Unknown', 'Unknown dataset "%s".', name);
            end
            cache(name) = D;
        end
    end

    methods (Static)
        function D = affectnet(sz, opts)
            % Faces at sz x sz grayscale, clean protocol; the train/test split is the
            % reader's (so the lab's test faces were never seen by a model trained on train).
            arguments
                sz (1,1) double
                opts.CacheDir (1,1) string = OceanViz.App.ModelLab.Datasets.affectnetCache()
            end
            p = OceanData.Adapters.AffectNetReader.cleanProtocol();
            common = [p, {'ImageSize', [sz sz], 'Grayscale', true, 'CacheDir', opts.CacheDir}];
            [Xtr, ytr, info] = OceanData.Adapters.AffectNetReader.read("train", common{:});
            [Xte, yte] = OceanData.Adapters.AffectNetReader.read("test", common{:});
            D = struct('Name', "affectnet" + sz, 'X', single(Xtr), 'y', ytr(:), 'Xte', single(Xte), 'yte', yte(:), ...
                'ClassNames', string(info.Classes));
        end

        function d = affectnetCache()
            repo = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
            d = string(fullfile(repo, 'Experiments', 'Scripts', 'Computer Vision', 'Results', 'vit_cv', 'cache'));
        end
    end

    methods (Static, Access = private)
        function [X, y] = usps(split)
            [rows, y, fmt] = OceanData.Adapters.UspsDigitsReader.read(split);
            X = OceanDL.Layout.DimFormat.toCanonical(OceanDL.Layout.DimFormat.fromRowMajor(reshape(rows.', 1, []), fmt.Shape), fmt.Format);
            X = (X + 1) / 2;
            y = y + 1;
        end
    end
end
