classdef ModelCatalogTest < OceanTest.BaseTestCase
    % MODELCATALOGTEST  Discovery and metadata of the real checkpoints in Results/Speech/checkpoints.

    properties
        Dir string
        Cat
    end

    methods (TestClassSetup)
        function setup(tc)
            tc.Dir = string(fullfile(tc.LibraryRoot, 'Results', 'Speech', 'checkpoints'));
            tc.assumeTrue(isfile(fullfile(tc.Dir, 'ctc_fsdd_best.mat')), 'Checkpoints not present.');
            tc.Cat = OceanNLP.Speech.Inference.ModelCatalog('Root', tc.Dir);
        end
    end

    methods (Test)
        function listsRecognisersWithFrozenColumns(tc)
            rng(9, 'twister'); s0 = rng;
            t = tic;
            T = tc.Cat.list();
            tc.log(1, sprintf('list(): %d recognisers in %.2f s', height(T), toc(t)));
            tc.verifyEqual(rng, s0);
            cols = ["Name", "File", "Type", "ModelId", "Units", "SampleRate", "NumParams", "BestEpoch", "BestValWER"];
            tc.verifyTrue(all(ismember(cols, string(T.Properties.VariableNames))));
            tc.verifyTrue(all(T.Loadable));
            tc.verifyTrue(any(T.Name == "ctc_fsdd_best"));
            tc.verifyTrue(any(T.Name == "encdec_fsdd_best"));
            r = T(T.Name == "encdec_fsdd_best", :);
            raw = load(r.File, 'ModelConfig');
            tc.verifyEqual(r.Type, "EncoderDecoderRecognizer");
            tc.verifyEqual(r.ModelId, string(raw.ModelConfig.ModelId));
            tc.verifyEqual(r.Units, string(raw.ModelConfig.Vocabulary.Unit));
            tc.verifyEqual(r.SampleRate, raw.ModelConfig.Features.SampleRate);
        end

        function skippedFilesCarryAReason(tc)
            T = tc.Cat.list('IncludeSkipped', true);
            pre = T(startsWith(T.Name, "hubert_fsdd_pretrain"), :);
            tc.assumeNotEmpty(pre);
            tc.verifyFalse(any(pre.Loadable));
            tc.verifyTrue(all(contains(pre.Reason, "pretraining")));
            tc.verifyError(@() tc.Cat.load(pre.Name(1)), 'OceanNLP:Speech:Inference:ModelCatalog:NotRecognizer');
            tc.verifyTrue(all(T.Loadable == (T.Reason == "")));
            tc.verifyError(@() tc.Cat.load("no_such_model"), 'OceanNLP:Speech:Inference:ModelCatalog:UnknownName');
        end

        function hubertFineTunedCheckpointsAreRecognisers(tc)
            T = tc.Cat.list();
            ft = T(startsWith(T.Name, "hubert_ft/"), :);
            tc.assumeNotEmpty(ft);
            tc.verifyTrue(all(ft.Type == "CTCRecognizer"));
            m = tc.Cat.load(ft.Name(1));
            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(ft.File(1));
            w = OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/5_theo_7.wav", ...
                'Root', OceanNLP.Speech.Data.SpeechManifest.defaultRoot());
            tc.verifyEqual(m.transcribe(w).Text, ck.Model.transcribe(w).Text);
        end

        function numParamsEqualsLoadedModel(tc)
            T = tc.Cat.list();
            for name = ["ctc_fsdd_best", "encdec_fsdd_best"]
                m = tc.Cat.load(name);
                tc.verifyEqual(T.NumParams(T.Name == name), m.numParameters());
            end
        end

        function bestEpochAndWerComeFromHistory(tc)
            T = tc.Cat.list();
            for name = ["ctc_fsdd_best", "encdec_fsdd_best"]
                r = T(T.Name == name, :);
                h = tc.Cat.history(name);
                e = h.Epochs([h.Epochs.Epoch] == r.BestEpoch);
                tc.verifyNumElements(e, 1);
                tc.verifyEqual(r.BestValWER, e.ValWER);
                tc.verifyTrue(isfield(h, 'StepLoss'));
            end
            hp = tc.Cat.history("hubert_fsdd_pretrain");
            tc.verifyTrue(isfield(hp, 'StepLoss'));
        end

        function loadIsCached(tc)
            c = OceanNLP.Speech.Inference.ModelCatalog('Root', tc.Dir);
            tc.verifyFalse(c.isLoaded("ctc_fsdd_best"));
            a = c.load("ctc_fsdd_best");
            tc.verifyTrue(c.isLoaded("ctc_fsdd_best"));
            b = c.load("ctc_fsdd_best");
            tc.verifyTrue(a == b);
            c.clearCache();
            tc.verifyFalse(c.isLoaded("ctc_fsdd_best"));
        end

        function listNeverReadsTheWeights(tc)
            % a copy of a real checkpoint whose weights are removed lists identically
            tmp = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            src = load(fullfile(tc.Dir, 'ctc_fsdd_best.mat'));
            src.WeightValues = {};
            src.WeightNames = {};
            save(fullfile(tmp.Folder, 'stripped.mat'), '-struct', 'src', '-v7');
            c = OceanNLP.Speech.Inference.ModelCatalog('Root', string(tmp.Folder));
            T = c.list();
            ref = tc.Cat.list();
            ref = ref(ref.Name == "ctc_fsdd_best", :);
            tc.verifyEqual(T.NumParams, ref.NumParams);
            tc.verifyEqual(T.BestValWER, ref.BestValWER);
            tc.verifyEqual(T.ModelId, ref.ModelId);
            tc.verifyError(@() c.load("stripped"), ?MException);   % loading needs the weights
        end

        function corruptFileDoesNotBreakTheCatalog(tc)
            % critic round 1, F9: one unreadable .mat used to make list() (and app launch) fail
            tmp = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            copyfile(fullfile(tc.Dir, 'ctc_fsdd_best.mat'), fullfile(tmp.Folder, 'good.mat'));
            fid = fopen(fullfile(tmp.Folder, 'bad.mat'), 'w');
            fwrite(fid, uint8('this is not a MAT-file'));
            fclose(fid);
            c = OceanNLP.Speech.Inference.ModelCatalog('Root', string(tmp.Folder));
            T = c.list();
            tc.verifyEqual(T.Name, "good");
            A = c.list('IncludeSkipped', true);
            bad = A(A.Name == "bad", :);
            tc.verifyEqual(height(bad), 1);
            tc.verifyFalse(bad.Loadable);
            tc.verifyTrue(startsWith(bad.Reason, "unreadable file:"));
        end
    end
end
