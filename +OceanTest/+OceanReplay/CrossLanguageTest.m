classdef CrossLanguageTest < OceanTest.BaseTestCase
    % CROSSLANGUAGETEST  Python -> MATLAB parity (Master plan Sec. 5.7). Reads
    %   the fixtures Python wrote (Python/ocean_replay/tests/make_fixtures.py):
    %   * toy_python.h5 passes MATLAB's Schema and gives exactly the values
    %     Python stored (toy_python_expected.json), UTF-8 tokens included;
    %   * a spec written by Python parses here and selects the same points.
    %   The MATLAB -> Python direction is Python's test_cross_language.py.

    methods (Test)
        function testReadPythonRecording(tc)
            f = fullfile(OceanTest.OceanReplay.CrossLanguageTest.fx(), 'toy_python.h5');
            tc.verifyEmpty(OceanReplay.IO.Schema.validate(f));
            r = OceanReplay.Core.Recording(f);
            tc.verifyEqual(string(r.Meta.source.language), "python");
            exp = jsondecode(fileread(fullfile(OceanTest.OceanReplay.CrossLanguageTest.fx(), 'toy_python_expected.json'), ...
                'Encoding', 'UTF-8')).channels;
            names = string(fieldnames(exp)).';
            tc.verifyEqual(sort(names), sort(r.names()));
            for n = names
                pts = exp.(n);
                if ~iscell(pts), pts = num2cell(pts); end
                for p = 1:numel(pts)
                    e = pts{p};
                    got = r.get(n, e.coords, "exact");
                    if isfield(e, 'tokens')
                        tc.verifyEqual(got, string(e.tokens(:)).', n);
                    else
                        want = reshape(e.data, [e.shape(:).' 1]);
                        tc.verifyEqual(double(got), want, n);
                    end
                end
            end
            tc.verifyEqual(r.get("tok", struct('k', 2)), ["w2" "ñ" "中"]);
            tc.verifyEqual(r.info("label").Categories, ["a" "b" "c"]);
            tc.verifyFalse(r.info("noisy").InTexts);
            tc.verifyEqual(r.info("noisy").Params.scale, 0.1);
        end

        function testPythonSpecSelectsTheSamePoints(tc)
            d = jsondecode(fileread(fullfile(OceanTest.OceanReplay.CrossLanguageTest.fx(), 'spec_python.json')));
            spec = OceanReplay.Core.Spec.fromStruct(d.spec);
            axis = d.axis(:).';
            tc.verifyEqual(spec.Groups(1).At.step.select(axis), logical(d.masks.a(:).'));
            tc.verifyEqual(spec.Groups(2).At.step.select(axis), logical(d.masks.b(:).'));
            tc.verifyEqual(spec.Views(1).At.step.select(axis), logical(d.masks.v(:).'));
            tc.verifyEqual(spec.Precision, "float64");
            tc.verifyEqual(spec.Views(1).Params.size, 4);
        end
    end

    methods (Test)
        function testPythonDiffusionOpensInReplayStudio(tc)
            % recorded by ocean_replay.adapters.diffusers_ddpm (Python) -> MATLAB's Diffusion layout
            f = fullfile(OceanTest.OceanReplay.CrossLanguageTest.fx(), 'ddpm_python_tiny.h5');
            tc.verifyEmpty(OceanReplay.IO.Schema.validate(f));
            s = OceanViz.App.Replay.ReplayStudio(f, 'Visible', false, 'Remember', false); c = onCleanup(@() s.close());
            tc.verifyEqual(s.Layout.name(), "Diffusion");
            tc.verifyEqual(s.Frames, 50:-1:1);
            s.setCoord("step", 30); s.stepForward();
            tc.verifyEqual(s.Coords.step, 29);
            caps = arrayfun(@(t) string(t.Caption.Text), s.Tiles);
            tc.verifyFalse(any(contains(caps, "could not draw")), strjoin(caps, " | "));
            s.showDetails(true);
            tc.verifyTrue(any(contains(string(s.DrawerText.Value), "(Eq. 2.2.5)")));
            tc.verifyTrue(any(contains(string(s.DrawerText.Value), "Recorded in: python")));
        end
    end

    methods (Static)
        function d = fx()
            d = fullfile(fileparts(mfilename('fullpath')), 'fixtures');
        end
    end
end
