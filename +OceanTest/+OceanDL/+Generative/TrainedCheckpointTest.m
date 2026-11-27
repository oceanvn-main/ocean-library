classdef TrainedCheckpointTest < OceanTest.BaseTestCase
    % TRAINEDCHECKPOINTTEST  The models trained by exp_vae_vs_hvae_usps are
    %   saved as ModelCards and can be rebuilt and used without retraining --
    %   the documented route:
    %     [net, card] = OceanDL.Training.ModelCard.load('.../cards/usps_hvae.mat');
    %     X = net.sample(n, [16 16]);
    %   Also checks that the weights do not load into a model built with
    %   different settings (shape mismatch is an error, not silent reuse).
    %   Skipped (not failed) when the full USPS run has not been done yet.

    methods (Test)
        function testCardsRebuildAndGenerate(testCase)
            dirC = OceanTest.OceanDL.Generative.TrainedCheckpointTest.cardDir(testCase);
            models = {"VAE", 'usps_vae.mat', 1; "HVAE", 'usps_hvae.mat', 2};
            for k = 1:size(models, 1)
                [name, file, layers] = models{k, :};
                f = fullfile(dirC, file);
                testCase.assumeTrue(isfile(f), sprintf('%s card not found (run exp_vae_vs_hvae_usps first)', name));
                [net, card] = OceanDL.Training.ModelCard.load(f);
                testCase.verifyClass(net, 'OceanDL.Generative.HierarchicalVAE', name);
                testCase.verifyEqual(net.numLatentLayers(), layers, name);
                testCase.verifyEqual(card.Meta.ImageSize, [16 16 1], name);
                testCase.verifyTrue(card.Meta.Binarized, name);
                X = net.sample(16, [16 16], 'Temperature', 1, 'Stream', RandStream('mt19937ar', 'Seed', 5));
                testCase.verifySize(X, [16 16 1 16], name);
                testCase.verifyTrue(all(X >= 0 & X <= 1, 'all'), name);                       % Bernoulli means
                X2 = net.sample(16, [16 16], 'Temperature', 1, 'Stream', RandStream('mt19937ar', 'Seed', 5));
                testCase.verifyEqual(X2, X, name);                                              % same seed, same images
                testCase.verifySize(net.reconstruct(X(:, :, :, 1:4)), [16 16 1 4], name);
            end
        end

        function testCardRefusesDifferentSettings(testCase)
            f = fullfile(OceanTest.OceanDL.Generative.TrainedCheckpointTest.cardDir(testCase), 'usps_hvae.mat');
            testCase.assumeTrue(isfile(f), 'HVAE card not found (run exp_vae_vs_hvae_usps first)');
            s = load(f, 'card');
            wrong = OceanDL.Generative.HierarchicalVAE('Widths', [16 32 32], 'LatentsPerLevel', [0 1 1], ...
                'LatentChannels', 4, 'Likelihood', "bernoulli", 'Groups', 8, 'Seed', 1);
            testCase.verifyError(@() wrong.loadStateDict(s.card.State), 'OceanDL:nn:Module:ShapeMismatch');
        end
    end

    methods (Static)
        function d = cardDir(testCase)
            d = fullfile(testCase.LibraryRoot, 'Experiments', 'Scripts', 'Computer Vision', 'Results', ...
                'generative', 'cards');
        end
    end
end
