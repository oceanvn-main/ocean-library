classdef DenseEncoderFactoryTest < OceanTest.BaseTestCase
    % DENSEENCODERFACTORYTEST  Unit tests for
    % OceanNLP.Embedding.DenseEncoderFactory and the
    % OceanNLP.Embedding.IDenseEncoder abstract contract.

    properties
        VocabFile
        Weights
    end

    methods (TestClassSetup)
        function buildFixture(testCase)
            tokens = ["[PAD]"; "[UNK]"; "[CLS]"; "[SEP]"; "cat"; "dog"];
            vocabPath = [tempname() '.txt'];
            fid = fopen(vocabPath, 'w');
            testCase.assertNotEqual(fid, -1, 'Failed to create temp vocab file.');
            for i = 1:numel(tokens)
                fprintf(fid, '%s\n', tokens(i));
            end
            fclose(fid);
            testCase.addTeardown(@() delete(vocabPath));
            testCase.VocabFile = vocabPath;

            W = struct();
            W.TokEmbedding.WordEmbeddings      = [0 0; 0 0; 1 1; 2 2; 3 0; 0 3];
            W.TokEmbedding.PositionEmbeddings  = [10 0; 0 10; 10 10; 5 5; 1 1];
            W.TokEmbedding.TokenTypeEmbeddings = [100 0; 0 100];
            W.TokEmbedding.LayerNormGamma = [1; 1];
            W.TokEmbedding.LayerNormBeta  = [0; 0];
            L.WQ = eye(2); L.bQ = [0;0]; L.WK = eye(2); L.bK = [0;0];
            L.WV = eye(2); L.bV = [0;0]; L.WO = eye(2); L.bO = [0;0];
            L.NumHeads = 1;
            L.W1 = eye(2); L.b1 = [0;0]; L.W2 = eye(2); L.b2 = [0;0];
            L.LN1Gamma = [1;1]; L.LN1Beta = [0;0];
            L.LN2Gamma = [1;1]; L.LN2Beta = [0;0];
            W.Layers = L;
            testCase.Weights = W;
        end
    end

    methods (Test)
        function testValidNamesListsAllFourProfiles(testCase)
            import OceanNLP.Embedding.DenseEncoderFactory
            names = DenseEncoderFactory.validNames();
            testCase.verifyEqual(sort(string(names)), ...
                sort(["bert-tiny", "bert-mini", "bert-base-uncased", "custom"]));
        end

        function testCreateReturnsBertEncoder(testCase)
            import OceanNLP.Embedding.DenseEncoderFactory
            enc = DenseEncoderFactory.create('bert-tiny', ...
                'Weights', testCase.Weights, 'VocabFile', testCase.VocabFile);
            testCase.verifyClass(enc, 'OceanNLP.Embedding.BertEncoder');
        end

        function testMultipleNamedProfilesRouteToTheSameClass(testCase)
            % NOT scope creep: the axis of variation this factory selects
            % between is CHECKPOINT, not class -- distinct profile names
            % are expected/documented to all currently resolve to
            % BertEncoder (see class header's "WHY THIS FACTORY IS NOT THE
            % SAME MISTAKE AS SimilarityFactory'S ORIGINAL SITUATION" note).
            import OceanNLP.Embedding.DenseEncoderFactory
            encTiny = DenseEncoderFactory.create('bert-tiny', ...
                'Weights', testCase.Weights, 'VocabFile', testCase.VocabFile);
            encMini = DenseEncoderFactory.create('bert-mini', ...
                'Weights', testCase.Weights, 'VocabFile', testCase.VocabFile);
            encCustom = DenseEncoderFactory.create('custom', ...
                'Weights', testCase.Weights, 'VocabFile', testCase.VocabFile);

            testCase.verifyClass(encTiny, 'OceanNLP.Embedding.BertEncoder');
            testCase.verifyClass(encMini, 'OceanNLP.Embedding.BertEncoder');
            testCase.verifyClass(encCustom, 'OceanNLP.Embedding.BertEncoder');
        end

        function testModelNameDefaultsToProfileNameWhenOmitted(testCase)
            import OceanNLP.Embedding.DenseEncoderFactory
            enc = DenseEncoderFactory.create('bert-tiny', ...
                'Weights', testCase.Weights, 'VocabFile', testCase.VocabFile);
            testCase.verifyEqual(enc.name(), "bert:bert-tiny");
        end

        function testExplicitModelNameOverridesProfileDefault(testCase)
            import OceanNLP.Embedding.DenseEncoderFactory
            enc = DenseEncoderFactory.create('bert-tiny', ...
                'Weights', testCase.Weights, 'VocabFile', testCase.VocabFile, ...
                'ModelName', "my-checkpoint");
            testCase.verifyEqual(enc.name(), "bert:my-checkpoint");
        end

        function testUnknownProfileErrors(testCase)
            import OceanNLP.Embedding.DenseEncoderFactory
            testCase.verifyError(@() DenseEncoderFactory.create('roberta-base', ...
                'Weights', testCase.Weights, 'VocabFile', testCase.VocabFile), ...
                'OceanNLP:Embedding:DenseEncoderFactory:UnknownProfile');
        end

        function testAbstractInterfaceCannotBeInstantiated(testCase)
            testCase.verifyError(@() OceanNLP.Embedding.IDenseEncoder(), ...
                'MATLAB:class:abstract');
        end
    end
end
