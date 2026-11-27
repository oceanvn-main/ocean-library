classdef SamplingSpecTest < OceanTest.BaseTestCase
    % SAMPLINGSPECTEST  OceanReplay.Core.Sampling rules against hand-computed
    %   point sets, and Spec JSON round trips / validation (Master plan Sec. 5.2).

    methods (Test)
        function testRulesOnDescendingAxis(tc)
            S = OceanReplay.Core.Sampling; ax = 1000:-1:0;
            tc.verifyEqual(sort(ax(S.every(20).select(ax))), 0:20:1000);
            tc.verifyEqual(ax(S.all().select(ax)), ax);
            tc.verifyEqual(sort(ax(S.window(51, 1).select(ax))), [50 51 52]);
            tc.verifyEqual(sort(ax(S.list([52 50]).select(ax))), [50 52]);
            u = S.union(S.every(250), S.window(51, 1));
            tc.verifyEqual(sort(ax(u.select(ax))), [0 50 51 52 250 500 750 1000]);
        end

        function testEveryWithAnchorAndRange(tc)
            S = OceanReplay.Core.Sampling; ax = 0:30;
            tc.verifyEqual(ax(S.every(5, 1, 21).select(ax)), [1 6 11 16 21]);
            tc.verifyEqual(ax(S.every(5, 1).select(ax)), [1 6 11 16 21 26]);         % no upper bound
        end

        function testNonIntegerAxis(tc)
            S = OceanReplay.Core.Sampling; lam = [10 5 2 1 0.5 0.2 0.1];              % run order
            tc.verifyEqual(lam(S.everyIndex(3).select(lam)), [10 1 0.1]);             % indices 0, 3, 6
            tc.verifyEqual(lam(S.list(0.2).select(lam)), 0.2);
            tc.verifyEqual(lam(S.window(0.3, 0.2).select(lam)), [0.5 0.2 0.1]);
        end

        function testMissingListValues(tc)
            S = OceanReplay.Core.Sampling;
            tc.verifyEqual(S.list([50 5000]).missing(0:1000), 5000);
            tc.verifyEqual(S.union(S.all(), S.list(-1)).missing(0:10), -1);
            tc.verifyEmpty(S.every(3).missing(0:10));
        end

        function testJsonRoundTrip(tc)
            S = OceanReplay.Core.Sampling;
            spec = OceanReplay.Core.Spec() ...
                .addGroup("numbers", 0, "energy", struct('t', S.all())) ...
                .addGroup("detail", 2, ["img" "mat"], struct('t', S.union(S.every(4, 0, 8), S.list(6)), 'k', S.list(2))) ...
                .addView("noisy", "noisy", struct('scale', 0.1), struct('t', S.list(8)));
            spec.SizeCapBytes = 1e6;
            txt = spec.toJson();
            back = OceanReplay.Core.Spec.fromJson(txt);
            tc.verifyEqual(back.toJson(), txt);
            tc.verifyEqual(back.Groups(1).Channels, "energy");                       % one-element list stays a list
            tc.verifyNotEmpty(regexp(txt, '"channels":\s*\[\s*"energy"\s*\]', 'once'));    % JSON array, not a bare string
            tc.verifyEqual(back.Groups(2).At.t.select(10:-2:0), [false true true true false true]);   % t = 8 6 4 0
            f = fullfile(tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder, 'spec.json');
            spec.toFile(f);
            tc.verifyEqual(OceanReplay.Core.Spec.fromFile(f).toJson(), txt);
        end

        function testValidationAgainstCatalog(tc)
            toy = OceanTest.OceanReplay.ToyRecordable();
            [c, v] = toy.catalog(); ax = toy.axesInfo();
            S = OceanReplay.Core.Sampling;
            tc.verifyEmpty(OceanTest.OceanReplay.ToyRecordable.fullSpec().check(c, v, ax));
            bad = OceanReplay.Core.Spec() ...
                .addGroup("g", 0, ["energy" "nope"]) ...
                .addGroup("h", 0, "tok", struct('t', S.all())) ...
                .addGroup("i", 0, "img", struct('t', S.list([4 7]))) ...
                .addView("v", "unknownMethod");
            issues = bad.check(c, v, ax);
            tc.verifyNumElements(issues, 4);
            tc.verifyTrue(any(contains(issues, '"nope"')));
            tc.verifyTrue(any(contains(issues, 'does not depend on')));
            tc.verifyTrue(any(contains(issues, 'not on axis "t"')));
            tc.verifyTrue(any(contains(issues, 'unknownMethod')));
            tc.verifyError(@() bad.validate(c, v, ax), 'OceanReplay:Spec:invalid');
        end

        function testEstimateBytes(tc)
            toy = OceanTest.OceanReplay.ToyRecordable();
            [c, v] = toy.catalog(); ax = toy.axesInfo();
            S = OceanReplay.Core.Sampling;
            spec = OceanReplay.Core.Spec() ...
                .addGroup("a", 0, "img", struct('t', S.every(4))) ...                 % t = 8, 4, 0 -> 3 x 12 x 4 bytes
                .addGroup("b", 1, "mat", struct('k', S.list(2))) ...                  % 6 t x 1 k -> 6 x 6 x 4
                .addView("n", "noisy", struct('scale', 1), struct('t', S.list(2)));   % 1 x 12 x 4
            [total, per] = spec.estimateBytes(c, v, ax);
            tc.verifyEqual([per.Points], [3 6 1]);
            tc.verifyEqual(total, 3 * 48 + 6 * 24 + 48);
            spec.Precision = "float64";
            tc.verifyEqual(spec.estimateBytes(c, v, ax), 2 * total);
            tc.verifyTrue(isnan(spec.estimateBytes(c, v, OceanTest.OceanReplay.ToyRecordable('OpenAxis', true).axesInfo())));
        end
    end
end
