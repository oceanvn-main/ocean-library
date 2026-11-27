classdef SpeechManifestTest < OceanTest.BaseTestCase
    % SPEECHMANIFESTTEST  FSDD manifest on the real tree: metadata, split, validation, I/O, audit.

    properties
        M          % full FSDD manifest (shared, read-only)
        RecDir
        Names      % independent listing of the recordings folder
        Digit
        Speaker
        Take
    end

    methods (TestClassSetup)
        function build(testCase)
            testCase.M = OceanTest.OceanNLP.Speech.Data.fsddTestManifest();
            testCase.RecDir = fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech', 'FSDD', 'recordings');
            d = dir(fullfile(testCase.RecDir, '*.wav'));
            testCase.Names = string({d.name})';
            parts = split(extractBefore(testCase.Names, ".wav"), "_");
            testCase.Digit = str2double(parts(:, 1));
            testCase.Speaker = parts(:, 2);
            testCase.Take = str2double(parts(:, 3));
        end
    end

    methods (Test)
        function testFromFSDDMatchesTree(testCase)
            m = testCase.M;
            E = m.Entries;
            testCase.verifyEqual(m.height(), numel(testCase.Names));
            testCase.verifyEqual(sort(E.Id), sort(extractBefore(testCase.Names, ".wav")));
            testCase.verifyTrue(all(startsWith(E.AudioPath, "FSDD/recordings/")));
            testCase.verifyEqual(m.Name, "FSDD");
            testCase.verifyTrue(all(E.Language == "en" & E.Session == "" & E.Split == ""));
            testCase.verifyTrue(all(contains(E.License, "CC BY-SA 4.0")));
            for s = unique(testCase.Speaker)'
                testCase.verifyEqual(nnz(E.Speaker == s), nnz(testCase.Speaker == s), s);
            end
            % transcript = English word of the file-name digit
            digit = str2double(extractBefore(E.Id, "_"));
            english = ["zero" "one" "two" "three" "four" "five" "six" "seven" "eight" "nine"];
            testCase.verifyEqual(E.Transcript, reshape(english(digit + 1), [], 1));
            % durations / rates against audioread (independent of audioinfo) on a sample
            pick = round(linspace(1, m.height(), 25));
            for i = pick
                [x, fs] = audioread(m.resolve(i));
                testCase.verifyEqual(E.SampleRate(i), fs);
                testCase.verifyEqual(E.Duration(i), size(x, 1) / fs, 'AbsTol', 1e-12, E.Id(i));
            end
            testCase.verifyTrue(isfile(m.resolve(1)));
        end

        function testCanonicalSplit(testCase)
            s = testCase.M.canonicalFSDDSplit();
            E = s.Entries;
            % independent expectation from the file names
            key = extractBefore(testCase.Names, ".wav");
            isTest = testCase.Speaker == "theo";
            isVal = ~isTest & testCase.Take >= 45 & testCase.Take <= 49;
            [~, loc] = ismember(E.Id, key);
            expected = repmat("train", numel(key), 1);
            expected(isVal) = "validation"; expected(isTest) = "test";
            testCase.verifyEqual(E.Split, expected(loc));
            counts = [nnz(E.Split == "train"), nnz(E.Split == "validation"), nnz(E.Split == "test")];
            testCase.verifyEqual(counts, [nnz(~isTest & ~isVal), nnz(isVal), nnz(isTest)]);

            tr = s.bySplit("train"); va = s.bySplit("validation"); te = s.bySplit("test");
            tr.assertDisjoint(te, 'By', "speaker");
            tr.assertDisjoint(va, 'By', "id");
            testCase.verifyError(@() tr.assertDisjoint(va, 'By', "speaker"), ...
                'OceanNLP:Speech:Data:SpeechManifest:Overlap');
        end

        function testSplitBySpeakerLeaveOneOut(testCase)
            s = testCase.M.splitBySpeaker(["jackson", "nicolas"], "yweweler", "theo");
            E = s.Entries;
            testCase.verifyEqual(nnz(E.Split == "train"), nnz(ismember(testCase.Speaker, ["jackson", "nicolas"])));
            testCase.verifyEqual(unique(E.Speaker(E.Split == "validation")), "yweweler");
            s.bySplit("train").assertDisjoint(s.bySplit("test"), 'By', "speaker");
            s2 = testCase.M.splitBySpeaker("jackson", strings(0, 1), "theo");          % others dropped
            testCase.verifyEqual(s2.height(), nnz(ismember(testCase.Speaker, ["jackson", "theo"])));
            testCase.verifyError(@() testCase.M.splitBySpeaker("theo", "theo", "jackson"), ...
                'OceanNLP:Speech:Data:SpeechManifest:Overlap');
            testCase.verifyError(@() testCase.M.splitBySpeaker("nobody", "theo", "jackson"), ...
                'OceanNLP:Speech:Data:SpeechManifest:UnknownSpeaker');
        end

        function testValidateRealCorpusAndListOffenders(testCase)
            rep = testCase.M.validate();
            testCase.verifyEqual(rep.NumFiles, numel(testCase.Names));
            E = testCase.M.Entries(1:4, :);
            E.AudioPath(2) = "FSDD/recordings/does_not_exist.wav";
            E.Duration(3) = E.Duration(3) + 0.5;
            bad = OceanNLP.Speech.Data.SpeechManifest(E, 'Root', testCase.M.Root, 'Name', "broken");
            try
                bad.validate();
                testCase.verifyFail('validate should have failed');
            catch ME
                testCase.verifyEqual(ME.identifier, 'OceanNLP:Speech:Data:SpeechManifest:Invalid');
                testCase.verifyTrue(contains(ME.message, E.Id(2)) && contains(ME.message, E.Id(3)));
                testCase.verifyFalse(contains(ME.message, E.Id(1) + ":"));
            end
        end

        function testSaveLoadRoundTripIsRootRelative(testCase)
            s = testCase.M.canonicalFSDDSplit().subset(1:37);
            f = string(tempname) + ".json";
            cleaner = onCleanup(@() delete(f));
            s.save(f);
            txt = fileread(f);
            testCase.verifyFalse(contains(txt, testCase.LibraryRoot));
            testCase.verifyFalse(contains(lower(txt), "c:\\"));
            back = OceanNLP.Speech.Data.SpeechManifest.load(f, 'Root', s.Root);
            testCase.verifyEqual(back.Entries, s.Entries);
            testCase.verifyEqual(back.Name, s.Name);
            empty = s.subset(false(37, 1));
            empty.save(f);
            testCase.verifyEqual(OceanNLP.Speech.Data.SpeechManifest.load(f).height(), 0);
        end

        function testAuditMatchesIndependentStatistics(testCase)
            s = testCase.M.canonicalFSDDSplit();
            t = s.audit();
            row = t(t.Split == "test" & t.Speaker == "theo", :);
            files = testCase.Names(testCase.Speaker == "theo");
            dur = zeros(numel(files), 1);
            for k = 1:numel(files)
                info = audioread(fullfile(testCase.RecDir, files(k)));
                dur(k) = size(info, 1) / 8000;
            end
            testCase.verifyEqual(row.NumUtterances, numel(files));
            testCase.verifyEqual(row.TotalDuration, sum(dur), 'RelTol', 1e-12);
            testCase.verifyEqual([row.MinDuration row.MedianDuration row.MaxDuration], ...
                [min(dur) median(dur) max(dur)], 'AbsTol', 1e-12);
            testCase.verifyEqual(row.SampleRates, "8000");
            all = t(t.Split == "ALL", :);
            testCase.verifyEqual(all.NumUtterances, numel(testCase.Names));
            testCase.verifyEqual(all.NumWords, numel(testCase.Names));    % one word per FSDD file
        end

        function testConstructorRejectsBadEntries(testCase)
            E = testCase.M.Entries(1:3, :);
            root = testCase.M.Root;
            Ctor = @(e) OceanNLP.Speech.Data.SpeechManifest(e, 'Root', root);
            Eabs = E; Eabs.AudioPath(1) = "C:/data/x.wav";
            testCase.verifyError(@() Ctor(Eabs), 'OceanNLP:Speech:Data:SpeechManifest:AbsolutePath');
            Edup = E; Edup.Id(2) = Edup.Id(1);
            testCase.verifyError(@() Ctor(Edup), 'OceanNLP:Speech:Data:SpeechManifest:DuplicateId');
            testCase.verifyError(@() Ctor(removevars(E, "Speaker")), 'OceanNLP:Speech:Data:SpeechManifest:Columns');
            testCase.verifyError(@() OceanNLP.Speech.Data.SpeechManifest.fromFSDD(string(tempname)), ...
                'OceanNLP:Speech:Data:SpeechManifest:DatasetNotFound');
        end

        function testDefaultRootIsRepoRelative(testCase)
            r = OceanNLP.Speech.Data.SpeechManifest.defaultRoot();
            testCase.verifyTrue(isfolder(r));
            expected = string(fullfile(testCase.LibraryRoot, 'Data', 'Datasets', 'Speech'));
            testCase.verifyEqual(string(java.io.File(r).getCanonicalPath()), ...
                string(java.io.File(expected).getCanonicalPath()));
        end
    end
end
