classdef SchemaExtractTest < OceanTest.BaseTestCase
    % SCHEMAEXTRACTTEST  OceanReplay.IO.Schema catches each rule violation of
    %   Format spec Sec. 7; OceanReplay.IO.extract copies exactly the selected
    %   points, reports the missing ones, and keeps provenance (Sec. 5.2).
    %   Also the architecture guard: OceanReplay depends on no domain package.

    properties
        Dir
        File
    end

    methods (TestMethodSetup)
        function recordToy(tc)
            tc.Dir = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            tc.File = fullfile(tc.Dir, "toy.h5");
            OceanReplay.Record.Recorder.run(OceanTest.OceanReplay.ToyRecordable(), ...
                OceanTest.OceanReplay.ToyRecordable.fullSpec(), tc.File, 'RunId', "toy-archive");
        end
    end

    methods (Test)
        function testValidRecordingHasNoIssues(tc)
            tc.verifyEmpty(OceanReplay.IO.Schema.validate(tc.File));
        end

        function testEachViolationIsReported(tc)
            cases = { ...
                @(f) h5writeatt(f, '/', 'schemaVersion', "2.0"),                       'unsupported schemaVersion'; ...
                @(f) h5writeatt(f, '/channels/img', 'kind', "blob"),                   'unknown kind "blob"'; ...
                @(f) h5writeatt(f, '/channels/img', 'valueShape', int32([2 3 3])),     'does not match valueShape'; ...
                @(f) h5write(f, '/channels/mat/points', int32([99; 0]), [1 1], [2 1]), 'index outside axis "t"'; ...
                @(f) h5writeatt(f, '/views/noisy', 'method', ""),                      'needs a non-empty method'; ...
                @(f) OceanTest.OceanReplay.SchemaExtractTest.deleteAttr(f, '/views/noisy', 'inTexts'),       'inTexts = 0 or 1'; ...
                @(f) OceanTest.OceanReplay.SchemaExtractTest.deleteAttr(f, '/channels/img', 'displayRange'), 'needs displayRange'; ...
                @(f) h5writeatt(f, '/', 'meta', "{not json"),                          'not valid JSON'};
            for i = 1:size(cases, 1)
                f = fullfile(tc.Dir, sprintf("bad%d.h5", i));
                copyfile(tc.File, f);
                cases{i, 1}(char(f));
                issues = OceanReplay.IO.Schema.validate(f);
                tc.verifyTrue(any(contains(issues, cases{i, 2})), sprintf('case %d: expected "%s", got: %s', ...
                    i, cases{i, 2}, strjoin(issues, ' | ')));
            end
        end

        function testExtractCopiesSelectedPointsOnly(tc)
            S = OceanReplay.Core.Sampling;
            spec = OceanReplay.Core.Spec() ...
                .addGroup("small", 1, ["img" "energy"], struct('t', S.list([4 6]))) ...
                .addView("noisy", "", struct(), struct('t', S.list(2)));
            out = fullfile(tc.Dir, "extract.h5");
            rep = OceanReplay.IO.extract(tc.File, spec, out);
            tc.verifyEmpty(OceanReplay.IO.Schema.validate(out));
            src = OceanReplay.Core.Recording(tc.File); ex = OceanReplay.Core.Recording(out);
            tc.verifyEqual(sort(ex.names()), sort(["img" "energy" "noisy"]));
            tc.verifyEqual(ex.stored("img").Values.', [6 4]);
            for t = [6 4]
                tc.verifyEqual(ex.get("img", struct('t', t), "exact"), src.get("img", struct('t', t), "exact"));
            end
            tc.verifyEqual(ex.get("noisy", struct('t', 2), "exact"), src.get("noisy", struct('t', 2), "exact"));
            tc.verifyEqual(rep.Copied.img, 2);
            tc.verifyEmpty(rep.NotStored);
            tc.verifyEqual(string(ex.Meta.extractedFrom.archiveRunId), "toy-archive");
            tc.verifyEqual(ex.Meta.source, src.Meta.source);              % provenance unchanged
            tc.verifyEqual(ex.Meta.seeds, src.Meta.seeds);
        end

        function testExtractReportsPointsNotInTheArchive(tc)
            S = OceanReplay.Core.Sampling;
            sparse = fullfile(tc.Dir, "sparse.h5");
            OceanReplay.Record.Recorder.run(OceanTest.OceanReplay.ToyRecordable(), ...
                OceanReplay.Core.Spec().addGroup("g", 1, "img", struct('t', S.every(4))), sparse);
            rep = OceanReplay.IO.extract(sparse, OceanReplay.Core.Spec().addGroup("x", 1, "img", struct('t', S.list([4 6]))), ...
                fullfile(tc.Dir, "x.h5"));
            tc.verifyEqual(rep.NotStored, "img@t=6");                      % on the axis, never stored
            tc.verifyEqual(rep.Copied.img, 1);
        end

        function testExtractRejectsUnknownNames(tc)
            spec = OceanReplay.Core.Spec().addGroup("g", 0, "nope");
            tc.verifyError(@() OceanReplay.IO.extract(tc.File, spec, fullfile(tc.Dir, "y.h5")), 'OceanReplay:Spec:invalid');
        end

        function testOceanReplayDependsOnNoDomainPackage(tc)
            root = fullfile(tc.LibraryRoot, '+OceanReplay');
            files = dir(fullfile(root, '**', '*.m'));
            tc.assertNotEmpty(files);
            for i = 1:numel(files)
                txt = fileread(fullfile(files(i).folder, files(i).name));
                hit = regexp(txt, 'Ocean(DL|Stats|NLP|Viz|Data|Math|Algo|CS|Engine|Pipeline|Tools|Test)\.', 'match', 'once');
                tc.verifyEmpty(hit, sprintf('%s references %s', files(i).name, hit));
            end
        end
    end

    methods (Static)
        function deleteAttr(f, path, name)
            fid = H5F.open(f, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
            c = onCleanup(@() H5F.close(fid));
            gid = H5G.open(fid, path);
            H5A.delete(gid, name);
            H5G.close(gid);
        end
    end
end
