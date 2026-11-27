classdef SpeechTrainerTest < OceanTest.BaseTestCase
    % SPEECHTRAINERTEST  Exact resume (N + save/load + N == 2N, bitwise), validation /
    %   best checkpoint, checkpoint round trip, RNG isolation -- on a small real FSDD subset.

    properties
        Manifest        % small train subset (real FSDD audio)
        ValManifest
        Vocab
        TmpDir
    end

    methods (TestClassSetup)
        function build(testCase)
            s = OceanTest.OceanNLP.Speech.Data.fsddTestManifest().canonicalFSDDSplit();
            tr = s.bySplit("train");
            va = s.bySplit("validation");
            testCase.Vocab = OceanNLP.Speech.Data.SpeechVocabulary.fit(tr.Entries.Transcript, 'Unit', "char");
            testCase.Manifest = tr.subset(round(linspace(1, tr.height(), 10)));
            testCase.ValManifest = va.subset(round(linspace(1, va.height(), 6)));
            testCase.TmpDir = string(tempname);
            mkdir(testCase.TmpDir);
            testCase.addTeardown(@() rmdir(testCase.TmpDir, 's'));
        end
    end

    methods (Test)
        function testExactResume(testCase)
            N = 5;   % 10 items, batch 4 -> 3 batches/epoch: the resume point is mid-epoch 2
            [trA, ~] = testCase.makeTrainer();
            trA.trainSteps(N);
            file = fullfile(testCase.TmpDir, "resume.mat");
            OceanNLP.Speech.Training.SpeechCheckpoint.save(file, trA.Model, trA.state());

            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(file);       % fresh model objects
            [~, bB] = testCase.makeData(ck.Features, ck.Vocabulary);         % fresh dataset + batcher
            rng(12345);                                                      % unrelated global RNG use
            trB = OceanNLP.Speech.Training.SpeechTrainer.resume(ck, bB);
            testCase.verifyEqual(trB.Step, N);
            trB.trainSteps(N);

            [trC, ~] = testCase.makeTrainer();                                 % 2N straight
            trC.trainSteps(2 * N);

            testCase.verifyEqual(trB.History.StepLoss, trC.History.StepLoss);  % bitwise
            testCase.verifyEqual(trB.History.StepLR, trC.History.StepLR);
            testCase.verifyEqual(trB.History.StepGradNorm, trC.History.StepGradNorm);
            sdB = trB.Model.stateDict(); sdC = trC.Model.stateDict();
            testCase.verifyEqual(sdB.keys(), sdC.keys());
            for k = sdC.keys()
                testCase.verifyEqual(sdB(k{1}), sdC(k{1}), sprintf('parameter %s differs', k{1}));
            end
            testCase.verifyEqual(trB.RngState, trC.RngState);
            testCase.verifyEqual([trB.Epoch trB.NextBatch], [trC.Epoch trC.NextBatch]);
            % the dropout path really drew random numbers (otherwise RNG restore is untested)
            testCase.verifyNotEqual(trC.RngState.State, testCase.initialRngState(0));
        end

        function testTrainerDoesNotTouchGlobalRng(testCase)
            tr = testCase.makeTrainer();
            rng(77);
            before = rng;
            tr.trainSteps(1);
            testCase.verifyEqual(rng, before);
        end

        function testFitValidationAndBestCheckpoint(testCase)
            best = fullfile(testCase.TmpDir, "best.mat");
            [tr, ~, vb] = testCase.makeTrainer('Epochs', 2, 'BestCheckpoint', best);
            tr.fit();
            E = tr.History.Epochs;
            testCase.verifyNumElements(E, 2);
            testCase.verifyEqual([E.Steps], [3 6]);
            testCase.verifyTrue(all(isfinite([E.ValLoss])) && all(isfinite([E.ValWER])));
            testCase.verifyTrue(isfile(best));
            testCase.verifyEqual(tr.BestScore(1), min([E.ValWER]));
            % the validation numbers are reproducible from the evaluator
            tr.restoreBest();
            v = tr.validate();
            testCase.verifyEqual([v.WER v.Loss], tr.BestScore, 'AbsTol', 1e-12);
            % checkpoint -> reload -> identical transcripts and logits
            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(best);
            b = vb.batch(1);
            t1 = tr.Model.transcribe(b, 'Decoder', "beam", 'BeamWidth', 4);
            t2 = ck.Model.transcribe(b, 'Decoder', "beam", 'BeamWidth', 4);
            testCase.verifyEqual(t2.texts(), t1.texts());
            testCase.verifyEqual([t2.Score], [t1.Score]);
            ck.Model.eval(); tr.Model.eval();
            testCase.verifyEqual(ck.Model.forward(b).Logits.Data, tr.Model.forward(b).Logits.Data);
        end

        function testTrainingReducesLossOnTinySet(testCase)
            % debugging sanity check only (NOT the experiment's result)
            tr = testCase.makeTrainer('Epochs', 8, 'LR', 3e-3);
            tr.trainSteps(24);
            L = tr.History.StepLoss;
            testCase.verifyLessThan(mean(L(end-2:end)), mean(L(1:3)));
        end

        function testCheckpointRoundTripAndErrors(testCase)
            tr = testCase.makeTrainer();
            file = fullfile(testCase.TmpDir, "rt.mat");
            OceanNLP.Speech.Training.SpeechCheckpoint.save(file, tr.Model, struct(), 'Extra', struct('Note', "x"));
            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(file);
            testCase.verifyEqual(ck.ModelConfig, tr.Model.config());
            testCase.verifyEqual(ck.Model.config(), tr.Model.config());
            testCase.verifyEqual(ck.Features.config(), tr.Model.Features.config());
            testCase.verifyEqual(ck.Vocabulary.stateDict(), tr.Model.Vocabulary.stateDict());
            testCase.verifyEqual(ck.Extra.Note, "x");
            testCase.verifyError(@() OceanNLP.Speech.Training.SpeechCheckpoint.load(fullfile(testCase.TmpDir, "none.mat")), ...
                'OceanNLP:Speech:Training:SpeechCheckpoint:NoFile');
            junk = fullfile(testCase.TmpDir, "junk.mat");
            a = 1; save(junk, 'a');
            testCase.verifyError(@() OceanNLP.Speech.Training.SpeechCheckpoint.load(junk), ...
                'OceanNLP:Speech:Training:SpeechCheckpoint:Format');
        end

        function testResumeRejectsDifferentBatcher(testCase)
            tr = testCase.makeTrainer();
            tr.trainSteps(1);
            file = fullfile(testCase.TmpDir, "bm.mat");
            OceanNLP.Speech.Training.SpeechCheckpoint.save(file, tr.Model, tr.state());
            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(file);
            ds = OceanNLP.Speech.Data.AudioDataset(testCase.Manifest, ck.Features, 'Vocabulary', ck.Vocabulary);
            other = OceanNLP.Speech.Data.SpeechBatcher(ds, 'BatchSize', 4, 'Seed', 99);
            testCase.verifyError(@() OceanNLP.Speech.Training.SpeechTrainer.resume(ck, other), ...
                'OceanNLP:Speech:Training:SpeechTrainer:BatcherMismatch');
        end

        function testResumeFromBestCheckpointKeepsBestWeights(testCase)
            % critic round 1, finding 2: restoreBest after resume used to throw NoBest
            best = fullfile(testCase.TmpDir, "best_resume.mat");
            [tr, ~, vb] = testCase.makeTrainer('Epochs', 2, 'BestCheckpoint', best);
            tr.trainSteps(3);                                   % one full epoch -> validated, best saved
            testCase.verifyTrue(isfile(best));
            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(best);
            [~, b2] = testCase.makeData(ck.Features, ck.Vocabulary);
            tr2 = OceanNLP.Speech.Training.SpeechTrainer.resume(ck, b2, 'ValBatcher', vb);
            saved = ck.Model.stateDict();
            tr2.trainSteps(1);                                  % weights move away from the best
            tr2.restoreBest();
            now = tr2.Model.stateDict();
            for k = saved.keys()
                testCase.verifyEqual(now(k{1}), saved(k{1}), sprintf('parameter %s', k{1}));
            end
        end

        function testBestCheckpointRequiresValidation(testCase)
            % critic round 1, finding 3: a BestCheckpoint without validation was silently never written
            fe = OceanNLP.Speech.Features.LogMel(8000, 'NumMels', 20);
            m = OceanNLP.Speech.Models.CTCRecognizer(fe, testCase.Vocab, ...
                {OceanNLP.Speech.Models.Encoders.ConvEncoder(20, 'Channels', 8, 'KernelSizes', 3, 'Strides', 1, 'Seed', 1)});
            [~, b] = testCase.makeData(fe, testCase.Vocab);
            testCase.verifyError(@() OceanNLP.Speech.Training.SpeechTrainer(m, b, 'BestCheckpoint', ...
                fullfile(testCase.TmpDir, "never.mat"), 'Verbose', false), ...
                'OceanNLP:Speech:Training:SpeechTrainer:BestNeedsValidation');
        end

        function testTrainStepsValidatesAtEpochBoundary(testCase)
            % critic round 1, finding 4: trainSteps(numBatches) skipped the epoch's validation row
            tr = testCase.makeTrainer();
            tr.trainSteps(3);                                   % exactly one epoch (3 batches)
            testCase.verifyNumElements(tr.History.Epochs, 1);
            testCase.verifyTrue(isfinite(tr.History.Epochs(1).ValWER));
            tr.trainSteps(1);                                   % continuing does not duplicate the row
            testCase.verifyNumElements(tr.History.Epochs, 1);
        end
    end

    methods
        function [tr, b, vb] = makeTrainer(testCase, varargin)
            fe = OceanNLP.Speech.Features.LogMel(8000, 'NumMels', 20);
            enc = {OceanNLP.Speech.Models.Encoders.ConvEncoder(20, 'Channels', 12, 'KernelSizes', 5, ...
                        'Strides', 2, 'Dropout', 0.1, 'Seed', 1), ...
                   OceanNLP.Speech.Models.Encoders.RecurrentEncoder(12, 'HiddenSize', 8, 'NumLayers', 2, ...
                        'Dropout', 0.2, 'Seed', 2)};
            m = OceanNLP.Speech.Models.CTCRecognizer(fe, testCase.Vocab, enc, 'Dropout', 0.1, 'Seed', 3, ...
                'ModelId', "test-ctc");
            [~, b] = testCase.makeData(fe, testCase.Vocab);
            vds = OceanNLP.Speech.Data.AudioDataset(testCase.ValManifest, fe, 'Vocabulary', testCase.Vocab);
            vb = OceanNLP.Speech.Data.SpeechBatcher(vds, 'BatchSize', 3, 'Shuffle', false);
            tr = OceanNLP.Speech.Training.SpeechTrainer(m, b, 'ValBatcher', vb, 'Epochs', 4, 'LR', 2e-3, ...
                'Seed', 5, 'Verbose', false, varargin{:});
        end

        function [ds, b] = makeData(testCase, fe, vocab)
            ds = OceanNLP.Speech.Data.AudioDataset(testCase.Manifest, fe, 'Vocabulary', vocab);
            b = OceanNLP.Speech.Data.SpeechBatcher(ds, 'BatchSize', 4, 'Seed', 1);
        end
    end

    methods (Static)
        function s = initialRngState(seed)
            prev = rng; rng(seed + 5, 'twister'); r = rng; s = r.State; rng(prev);
        end
    end
end
