classdef FormatReplayTest < OceanTest.BaseTestCase
    % FORMATREPLAYTEST  File round trips for every kind, the MATLAB <-> h5py
    %   dimension rule, partial files, lazy reads, and nearest-point replay that
    %   never blends (Master plan Sec. 5.1, 5.4).

    properties
        Dir
        File
        Toy
    end

    methods (TestMethodSetup)
        function recordToy(tc)
            tc.Dir = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder;
            tc.File = fullfile(tc.Dir, "toy.h5");
            tc.Toy = OceanTest.OceanReplay.ToyRecordable();
            OceanReplay.Record.Recorder.run(tc.Toy, OceanTest.OceanReplay.ToyRecordable.fullSpec(), tc.File);
        end
    end

    methods (Test)
        function testEveryKindRoundTripsExactly(tc)
            r = OceanReplay.Core.Recording(tc.File);
            tc.verifyEqual(r.Status, "complete");
            for n = ["energy" "vec" "mat" "img" "seq" "pts" "tok" "label"]
                info = r.info(n);
                h = tc.Toy.History(char(n));                             % every provider call (views call img too)
                s = r.stored(n);
                tc.verifyEqual(s.P, tc.Toy.calls(n) - (n == "img") * 2, n);   % 2 extra img calls come from the view
                for p = 1:s.P
                    coords = struct(); match = true(1, numel(h));
                    for a = 1:numel(info.Axes)
                        coords.(info.Axes(a)) = s.Values(p, a);
                        match = match & arrayfun(@(e) e.Coords(info.Axes(a) == ["t" "k"]), h) == s.Values(p, a);
                    end
                    got = r.get(n, coords, "exact");
                    want = OceanReplay.Core.Kinds.normalise(info.Kind, h(find(match, 1)).Value);
                    switch info.Dtype
                        case "float32", want = single(want);
                        case "int32", want = int32(want);
                    end
                    if info.Kind == "tokens", want = reshape(want, 1, []); end
                    tc.verifyEqual(got, want, sprintf('%s point %d', n, p));
                end
            end
            tc.verifyEqual(r.info("label").Categories, ["a" "b" "c"]);
            tc.verifyEqual(r.info("mat").RowLabels, ["r1" "r2"]);
            tc.verifyEqual(r.info("img").DisplayRange, [-5 5]);
            tc.verifyEqual(r.get("tok", struct('k', 2)), ["w2" "ñ" "中"]);        % UTF-8 survives
        end

        function testViewAttributesRoundTrip(tc)
            r = OceanReplay.Core.Recording(tc.File);
            v = r.info("noisy");
            tc.verifyClass(v, 'OceanReplay.Core.ViewInfo');
            tc.verifyFalse(v.InTexts);
            tc.verifyEqual(v.Params.scale, 0.1);
            tc.verifyEqual(v.MethodId, "noisy");
            tc.verifyTrue(startsWith(v.HowMade, "img plus Gaussian noise"));
            tc.verifyEqual(r.stored("noisy").Values.', [8 2]);
        end

        function testDimensionRuleAgainstPythonFixture(tc)
            fx = fullfile(fileparts(mfilename('fullpath')), 'fixtures');
            base = single(reshape(1:24, 2, 3, 4));
            for file = ["g0_python.h5" "g0_matlab.h5"]
                R = OceanReplay.IO.Hdf5Reader(fullfile(fx, file));
                for p = 1:3
                    tc.verifyEqual(R.readPoint("/channels/x/data", [2 3 4], p), base + 100 * p, file);
                end
                tc.verifyEqual(R.text('/', 'format'), "ocean-replay", file);               % char (h5py) or string (MATLAB)
                tc.verifyEqual(R.number('/channels/x', 'valueShape'), [2 3 4], file);     % column -> row
            end
            R = OceanReplay.IO.Hdf5Reader(fullfile(fx, "g0_python.h5"));
            tc.verifyEqual(R.readPoint("/channels/tok/data", 5, 1).', ["the" "cat" "ñé" "中文" "."]);
            tc.verifyEqual(R.number('/channels/x', 'inTexts'), 1);
        end

        function testPartialFileIsReadable(tc)
            f = fullfile(tc.Dir, "partial.h5");
            w = OceanReplay.IO.Hdf5Writer(f);
            w.begin(struct('source', struct(), 'seeds', struct(), 'spec', struct()), OceanReplay.Core.Axis("t", 1:5));
            w.define(OceanReplay.Core.ChannelInfo("e", "scalar", 'Axes', "t"), "channels");
            w.append("e", 3, w.axisIndex("t", 1));
            w.append("e", 4, w.axisIndex("t", 2));
            r = OceanReplay.Core.Recording(f);                            % writer not closed
            tc.verifyTrue(r.isPartial());
            tc.verifyEqual(r.get("e", 2), single(4));
            w.close();
            tc.verifyFalse(OceanReplay.Core.Recording(f).isPartial());
        end

        function testGetReadsOnlyOnePoint(tc)
            r = OceanReplay.Core.Recording(tc.File);
            r.stored("img");                                              % point index read once (cached)
            before = r.Reader.ReadCount;
            r.get("img", struct('t', 4));
            tc.verifyEqual(r.Reader.ReadCount - before, 1);
        end

        function testNearestNeverBlends(tc)
            S = OceanReplay.Core.Sampling;
            f = fullfile(tc.Dir, "sparse.h5");
            toy = OceanTest.OceanReplay.ToyRecordable();
            OceanReplay.Record.Recorder.run(toy, OceanReplay.Core.Spec().addGroup("g", 1, ["img" "mat"], struct('t', S.every(4))), f);
            r = OceanReplay.Core.Recording(f);                            % img at t = 8, 4, 0
            [~, at] = r.get("img", struct('t', 5));  tc.verifyEqual(at.t, 4);
            [~, at] = r.get("img", struct('t', 6));  tc.verifyEqual(at.t, 8);   % tie 4 / 8 -> recorded first (8)
            [~, at] = r.get("img", struct('t', 99)); tc.verifyEqual(at.t, 8);
            stored = arrayfun(@(t) r.get("img", struct('t', t), "exact"), [8 4 0], 'UniformOutput', false);
            for q = 0:0.5:10
                v = r.get("img", struct('t', q));
                tc.verifyTrue(any(cellfun(@(s) isequal(s, v), stored)), sprintf('t = %g blended', q));
            end
            [~, at] = r.get("mat", struct('t', 6, 'k', 2.6));             % tie on t (8 / 4), then k -> 3
            tc.verifyEqual([at.t at.k], [8 3]);
            tc.verifyError(@() r.get("img", struct('t', 6), "exact"), 'OceanReplay:Recording:notStored');
            tc.verifyError(@() r.get("mat", struct('t', 6)), 'OceanReplay:Recording:coords');
        end

        function testSeriesOfAScalar(tc)
            r = OceanReplay.Core.Recording(tc.File);
            [x, y] = r.series("energy");
            h = tc.Toy.History('energy');
            tc.verifyEqual(x.', [10 8 6 4 2 0]);
            tc.verifyEqual(y, double(single([h.Value].')));
            tc.verifyError(@() r.series("img"), 'OceanReplay:Recording:series');
        end
    end
end
