classdef RecorderTest < OceanTest.BaseTestCase
    % RECORDERTEST  OceanReplay.Record.Recorder on ToyRecordable (Master plan Sec. 5.3):
    %   invariance, laziness, multi-level point sets, static / sub-axis channels,
    %   open axes, the size cap and missing providers.

    properties
        Dir
    end

    methods (TestMethodSetup)
        function makeDir(tc)
            tc.Dir = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
        end
    end

    methods (Test)
        function testRecordingDoesNotChangeTheRun(tc)
            spec = OceanTest.OceanReplay.ToyRecordable.fullSpec();       % includes a view that draws global random numbers
            [ok, r] = OceanReplay.Record.verifyInvariance(@() OceanTest.OceanReplay.ToyRecordable(), spec, ...
                'File', fullfile(tc.Dir, "inv.h5"));
            tc.verifyTrue(ok);
            tc.verifyEqual(r.With, r.Without);
            % control: the run really depends on the global generator, so an unguarded view would show
            rng(1, 'twister'); a = OceanTest.OceanReplay.ToyRecordable().run(OceanReplay.Record.NullRecorder());
            rng(2, 'twister'); b = OceanTest.OceanReplay.ToyRecordable().run(OceanReplay.Record.NullRecorder());
            tc.verifyNotEqual(a.lastGlobal, b.lastGlobal);
            tc.verifyNotEqual(a.x, b.x);
        end

        function testOnlyWantedChannelsAreEvaluated(tc)
            toy = OceanTest.OceanReplay.ToyRecordable();
            spec = OceanReplay.Core.Spec().addGroup("numbers", 0, "energy");
            OceanReplay.Record.Recorder.run(toy, spec, fullfile(tc.Dir, "lazy.h5"));
            tc.verifyEqual(toy.calls("energy"), 6);                      % once per t, not once per (t, k)
            for n = ["vec" "mat" "img" "seq" "pts" "tok" "label"]
                tc.verifyEqual(toy.calls(n), 0, n);
            end
        end

        function testMultiLevelPointSets(tc)
            S = OceanReplay.Core.Sampling;
            spec = OceanReplay.Core.Spec() ...
                .addGroup("numbers", 0, "energy") ...
                .addGroup("images", 1, "img", struct('t', S.every(4))) ...
                .addGroup("detail", 2, "img", struct('t', S.window(6, 0))) ...
                .addGroup("detailMat", 2, "mat", struct('t', S.window(6, 0), 'k', S.list(2)));
            f = fullfile(tc.Dir, "levels.h5");
            OceanReplay.Record.Recorder.run(OceanTest.OceanReplay.ToyRecordable(), spec, f);
            r = OceanReplay.Core.Recording(f);
            tc.verifyEqual(r.stored("energy").Values.', [10 8 6 4 2 0]);
            tc.verifyEqual(r.stored("img").Values.', [8 6 4 0]);         % every 4 (8, 4, 0) + detail window (6), run order
            tc.verifyEqual(r.stored("mat").Values, [6 2]);
            tc.verifyEqual(r.info("img").Level, 1);                      % lowest level of its groups
            tc.verifyEqual(r.info("mat").Level, 2);
        end

        function testStaticAndSubAxisChannelsStoredOnce(tc)
            toy = OceanTest.OceanReplay.ToyRecordable();
            f = fullfile(tc.Dir, "static.h5");
            OceanReplay.Record.Recorder.run(toy, OceanReplay.Core.Spec().addGroup("g", 0, ["pts" "tok"]), f);
            r = OceanReplay.Core.Recording(f);
            tc.verifyEqual(r.stored("pts").P, 1);
            tc.verifyEqual(r.stored("tok").Values.', [1 2 3]);
            tc.verifyEqual(toy.calls("pts"), 1);
            tc.verifyEqual(toy.calls("tok"), 3);
        end

        function testOpenAxisIsFilledInRunOrder(tc)
            S = OceanReplay.Core.Sampling;
            f = fullfile(tc.Dir, "open.h5");
            spec = OceanReplay.Core.Spec().addGroup("g", 0, "energy", struct('t', S.union(S.every(4), S.list(7))));
            rec = OceanReplay.Record.Recorder(OceanTest.OceanReplay.ToyRecordable('OpenAxis', true), spec, f);
            tc.verifyWarning(@() rec.record(), 'OceanReplay:Recorder:notVisited');   % t = 7 never happens
            r = OceanReplay.Core.Recording(f);
            tc.verifyEqual(r.axis("t").Values, [10 8 6 4 2 0]);
            tc.verifyEqual(r.stored("energy").Values.', [8 4 0]);
            tc.verifyEqual(string(r.Meta.notVisited), "energy@t=7");
            tc.verifyEqual(rec.Report.NotVisited, "energy@t=7");
        end

        function testSizeCapRefusesLargeRuns(tc)
            spec = OceanTest.OceanReplay.ToyRecordable.fullSpec();
            spec.SizeCapBytes = 10;
            tc.verifyError(@() OceanReplay.Record.Recorder(OceanTest.OceanReplay.ToyRecordable(), spec, fullfile(tc.Dir, "cap.h5")), ...
                'OceanReplay:Recorder:tooLarge');
            tc.verifyFalse(isfile(fullfile(tc.Dir, "cap.h5")));           % refused BEFORE anything is written
        end

        function testMissingProviderIsAnError(tc)
            spec = OceanReplay.Core.Spec().addGroup("g", 0, "ghost");
            tc.verifyError(@() OceanReplay.Record.Recorder.run(OceanTest.OceanReplay.ToyRecordable(), spec, fullfile(tc.Dir, "ghost.h5")), ...
                'OceanReplay:Recorder:missingState');
        end

        function testMetaRecordsProvenance(tc)
            f = fullfile(tc.Dir, "meta.h5");
            OceanReplay.Record.Recorder.run(OceanTest.OceanReplay.ToyRecordable(), OceanTest.OceanReplay.ToyRecordable.fullSpec(), f, ...
                'Title', "toy ñ", 'RunId', "toy-1", 'Seeds', struct('data', 5), 'ViewSeed', 42);
            m = OceanReplay.Core.Recording(f).Meta;
            tc.verifyEqual(string(m.title), "toy ñ");
            tc.verifyEqual(string(m.runId), "toy-1");
            tc.verifyEqual(string(m.source.language), "matlab");
            tc.verifyEqual([m.seeds.data m.seeds.views], [5 42]);
            back = OceanReplay.Core.Spec.fromStruct(m.spec);
            tc.verifyEqual(back.toJson(), OceanTest.OceanReplay.ToyRecordable.fullSpec().toJson());
        end
    end
end
