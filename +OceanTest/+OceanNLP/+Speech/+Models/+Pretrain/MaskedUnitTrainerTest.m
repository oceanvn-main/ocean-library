classdef MaskedUnitTrainerTest < OceanTest.BaseTestCase
    % MASKEDUNITTRAINERTEST  Pretraining on real FSDD log-mel with codebook targets: loss falls,
    %   runs are reproducible, checkpoints round-trip, iteration-2 re-clustering of hidden
    %   features, target validation.

    properties
        Seqs
        Val
        Codebook
    end

    methods (TestClassSetup)
        function build(testCase)
            names = ["0_jackson_0", "1_jackson_0", "2_nicolas_0", "3_nicolas_0", "4_yweweler_0", ...
                     "5_yweweler_0", "6_jackson_1", "7_nicolas_1", "8_yweweler_1", "9_jackson_2"];
            testCase.Seqs = OceanTest.OceanNLP.Speech.Models.Pretrain.UnitCodebookTest.fsddFeatures(names);
            testCase.Val = OceanTest.OceanNLP.Speech.Models.Pretrain.UnitCodebookTest.fsddFeatures(["3_jackson_46", "5_nicolas_47"]);
            testCase.Codebook = OceanNLP.Speech.Models.Pretrain.UnitCodebook(6, 'Seed', 0).fit(testCase.Seqs);
        end
    end

    methods (Test)
        function testTrainingReducesLossAndIsReproducible(testCase)
            [tr, ~] = testCase.makeTrainer(6);
            r0 = tr.evaluate("train");
            tr.fit();
            r1 = tr.evaluate("train");
            testCase.verifyLessThan(r1.Loss, r0.Loss, 'masked-unit loss on the training set must fall');
            testCase.verifyEqual(numel(tr.History.Epochs), 6);
            testCase.verifyTrue(all(isfinite(tr.History.StepLoss)));
            testCase.verifyTrue(isfinite(tr.History.Epochs(end).ValMaskedAcc));
            testCase.verifyEqual(tr.Model.CodebookVersion, testCase.Codebook.Version);
            % identical seed -> identical run, even with unrelated global RNG use in between
            [tr2, ~] = testCase.makeTrainer(6);
            rand(1, 17);
            tr2.fit();
            testCase.verifyEqual(tr2.History.StepLoss, tr.History.StepLoss);
            % mask fraction statistic of training batches is near the configured expectation
            testCase.verifyGreaterThan(mean(tr.History.StepMaskFraction), 0.2);
            testCase.verifyLessThan(mean(tr.History.StepMaskFraction), 0.8);
        end

        function testCheckpointRoundTrip(testCase)
            [tr, m] = testCase.makeTrainer(1);
            tr.fit();
            f = string(tempname) + ".mat";
            cleanup = onCleanup(@() delete(f));
            tr.save(f);
            ck = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer.load(f);
            testCase.verifyEqual(ck.Model.config(), m.config());
            X = randn(m.InputDim, 1, 30);
            m.eval(); ck.Model.eval();
            testCase.verifyEqual(ck.Model.forward(X, 30).Data, m.forward(X, 30).Data);
            testCase.verifyEqual(ck.History.StepLoss, tr.History.StepLoss);
        end

        function testIterationTwoReclustering(testCase)
            [tr, m] = testCase.makeTrainer(1);
            tr.fit();
            hid = tr.hiddenFeatures(1, "train");
            testCase.verifyEqual(numel(hid), numel(testCase.Seqs));
            testCase.verifyEqual(hid{1}.numBins(), m.Dim);
            testCase.verifyEqual(hid{1}.numFrames(), m.outputLength(testCase.Seqs{1}.numFrames()));
            cb2 = OceanNLP.Speech.Models.Pretrain.UnitCodebook(6, 'Seed', 0).fit(hid);
            testCase.verifyNotEqual(cb2.Version, testCase.Codebook.Version);
            testCase.verifyEqual(cb2.FeatureConfig.Type, "HuBERTHidden");
            m2 = OceanNLP.Speech.Models.Pretrain.HuBERTModel.fromConfig(m.config());
            tr2 = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer(m2, testCase.Seqs, 'Epochs', 1, 'BatchSize', 4, 'Verbose', false);
            tr2.setTargets(cb2.assign(hid), 'Rate', "output", 'Codebook', cb2);
            tr2.fit();
            testCase.verifyEqual(m2.CodebookVersion, cb2.Version);
            testCase.verifyTrue(all(isfinite(tr2.History.StepLoss)));
        end

        function testTargetValidation(testCase)
            [tr, ~] = testCase.makeTrainer(1);
            u = testCase.Codebook.assign(testCase.Seqs);
            bad = u; bad{1} = bad{1}(1:end-1);
            testCase.verifyError(@() tr.setTargets(bad), 'OceanNLP:Speech:Models:MaskedUnitTrainer:TargetLength');
            testCase.verifyError(@() tr.setTargets(u(1:end-1)), 'OceanNLP:Speech:Models:MaskedUnitTrainer:TargetCount');
            testCase.verifyError(@() tr.setTargets(u, 'Rate', "output"), 'OceanNLP:Speech:Models:MaskedUnitTrainer:TargetLength');
            m = OceanNLP.Speech.Models.Pretrain.HuBERTModel(40, 'Dim', 8, 'Depth', 1, 'Heads', 2, 'FFNDim', 8, 'ConvChannels', [8 8]);
            t3 = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer(m, testCase.Seqs, 'Verbose', false);
            testCase.verifyError(@() t3.fit(), 'OceanNLP:Speech:Models:MaskedUnitTrainer:NoTargets');
        end
    end

    methods
        function [tr, m] = makeTrainer(testCase, epochs)
            m = OceanNLP.Speech.Models.Pretrain.HuBERTModel(40, 'ConvChannels', [16 16], 'ConvKernels', [5 3], ...
                'ConvStrides', [2 1], 'Dim', 16, 'Depth', 1, 'Heads', 2, 'FFNDim', 32, 'Dropout', 0.1, ...
                'NumUnits', 6, 'ProjDim', 8, 'Seed', 1);
            tr = OceanNLP.Speech.Models.Pretrain.MaskedUnitTrainer(m, testCase.Seqs, 'Epochs', epochs, 'BatchSize', 4, ...
                'LR', 3e-3, 'Seed', 0, 'ValInputs', testCase.Val, 'Verbose', false);
            tr.setTargets(testCase.Codebook.assign(testCase.Seqs), 'Codebook', testCase.Codebook);
            tr.setValTargets(testCase.Codebook.assign(testCase.Val));
        end
    end
end
