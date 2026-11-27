classdef ContourFrameStrategiesTest < OceanTest.BaseTestCase
    % CONTOURFRAMESTRATEGIESTEST  SpeechContour and SpeechFrame strategies on REAL data (a
    %   held-out FSDD recording through Features.LogMel / EnergyContour), via OceanViz.Core.plot:
    %   data contract, values drawn unchanged, legend / labels, error ids. Also the
    %   BeamCandidates baseline (best candidate = longest bar, all inside the axes).

    properties
        Viz
        Wave
        St
        Grid
        Energy
        Fig
    end

    methods (TestClassSetup)
        function realData(tc)
            root = OceanNLP.Speech.Data.SpeechManifest.defaultRoot();
            tc.assumeTrue(isfolder(fullfile(root, 'FSDD')), 'FSDD not present.');
            tc.Wave = OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/3_theo_7.wav", 'Root', root);
            lm = OceanNLP.Speech.Features.LogMel(tc.Wave.SampleRate);
            tc.St = lm.intermediates(tc.Wave);
            tc.Grid = lm.Grid;
            ec = OceanNLP.Speech.Features.EnergyContour(tc.Wave.SampleRate);
            tc.Energy = ec.forward(tc.Wave);
            tc.Viz = OceanViz.Core('Theme', "Dark");
        end
    end

    methods (TestMethodSetup)
        function makeFigure(tc)
            tc.Fig = figure('Visible', 'off');
            tc.addTeardown(@() close(tc.Fig));
        end
    end

    methods (Test)
        function contourDrawsSeriesUnchanged(tc)
            ax = axes(tc.Fig);
            e = tc.Energy;
            d = struct('X', e.FrameTimes, 'Series', struct('Name', {"Energy", "Energy - 1"}, ...
                'Y', {e.Values, e.Values - 1}), 'XLabel', "Time (s)", 'YLabel', "log energy", ...
                'Intervals', [0.1 0.2; 0.3 0.35], 'IntervalName', "Speech", ...
                'RefLines', struct('Name', "Threshold", 'Y', median(e.Values)), 'Title', "Energy contour");
            tc.Viz.plot(ax, "SpeechContour", d);
            lines = findobj(ax, 'Type', 'line');
            tc.verifyNumElements(lines, 2);
            ys = sortrows(vertcat(lines.YData));
            tc.verifyEqual(ys, sortrows([e.Values; e.Values - 1]));
            tc.verifyEqual(lines(1).XData, e.FrameTimes);
            tc.verifyNumElements(findall(ax, 'Type', 'patch'), 2);
            tc.verifyNumElements(findobj(ax, 'Type', 'constantline'), 1);
            tc.verifyEqual(string(ax.XLabel.String), "Time (s)");
            tc.verifyEqual(string(ax.YLabel.String), "log energy");
            tc.verifyEqual(string(ax.Title.String), "Energy contour");
            % legend: 2 series + 1 interval entry (2nd patch hidden) + threshold
            tc.verifyEqual(sort(string(ax.Legend.String)), sort(["Speech", "Energy", "Energy - 1", "Threshold"]));
        end

        function contourErrors(tc)
            ax = axes(tc.Fig);
            e = tc.Energy;
            bad = struct('X', e.FrameTimes, 'Series', struct('Name', "x", 'Y', e.Values(1:end-1)), ...
                'XLabel', "t", 'YLabel', "y");
            tc.verifyError(@() tc.Viz.plot(ax, "SpeechContour", bad), ...
                'OceanViz:Strategies:Speech:SpeechContourStrategy:Length');
            tc.verifyError(@() tc.Viz.plot(ax, "SpeechContour", struct('X', 1)), ...
                'OceanViz:Strategies:Speech:MissingField');
        end

        function frameDrawsRawWindowAndProduct(tc)
            ax = axes(tc.Fig);
            k = 5;
            x = tc.St.Frames(:, k);
            w = tc.St.Window(:);
            g = tc.Grid;
            starts = g.frameStarts(tc.Wave.numSamples());
            d = struct('Frame', x, 'Window', w, 'SampleRate', g.SampleRate, 'FrameIndex', k, ...
                'FrameStart', starts(k), 'WindowType', g.Window);
            tc.Viz.plot(ax, "SpeechFrame", d);
            lines = findobj(ax, 'Type', 'line');
            tc.verifyNumElements(lines, 3);
            names = string({lines.DisplayName});
            raw = lines(startsWith(names, "Raw"));
            win = lines(contains(names, "window w[n] x"));
            prod = lines(startsWith(names, "Windowed"));
            tc.verifyEqual(raw.YData(:), x);
            tc.verifyEqual(prod.YData(:), w .* x, 'AbsTol', 1e-15);
            tc.verifyEqual(win.YData(:), w * max(abs(x)), 'AbsTol', 1e-15);
            tc.verifyEqual(raw.XData(:), (0:numel(x) - 1)' / g.SampleRate * 1000, 'AbsTol', 1e-12);
            tc.verifySubstring(string(ax.XLabel.String), "(ms)");
            tc.verifySubstring(string(ax.Title.String), sprintf("t = %d", k));
            tc.verifyNumElements(ax.Legend.String, 3);
        end

        function frameErrors(tc)
            ax = axes(tc.Fig);
            d = struct('Frame', tc.St.Frames(:, 1), 'Window', tc.St.Window(1:end-1), 'SampleRate', 8000);
            tc.verifyError(@() tc.Viz.plot(ax, "SpeechFrame", d), ...
                'OceanViz:Strategies:Speech:SpeechFrameStrategy:Length');
            tc.verifyError(@() tc.Viz.plot(ax, "SpeechFrame", struct('Frame', 1)), ...
                'OceanViz:Strategies:Speech:MissingField');
        end

        function beamBestIsLongestBarInsideAxes(tc)
            ax = axes(tc.Fig);
            sc = [-0.197, -3.2, -4.0, -6.5, -7.1, -7.9, -8.4, -9.9];   % all negative, best near 0
            c = struct('Text', cellstr("h" + (1:8)), 'TokenIds', {[]}, 'Score', num2cell(sc));
            tc.Viz.plot(ax, "BeamCandidates", struct('Candidates', c));
            bars = findobj(ax, 'Type', 'bar');
            base = bars(1).BaseValue;
            tc.verifyLessThan(base, min(sc));
            len = sc - base;
            tc.verifyEqual(find(len == max(len)), 1, 'Best candidate is the longest bar');
            tc.verifyGreaterThanOrEqual(min(sc), ax.XLim(1));
            tc.verifyLessThan(max(sc), ax.XLim(2));
            tc.verifyEqual(ax.YLim, [0.4, 8.6]);
        end

        function beamNonFiniteScoreStaysInsideAxes(tc)
            ax = axes(tc.Fig);
            sc = [-0.2, -0.6, -1.1, -Inf];        % -Inf: CTC-infeasible under rescoring
            c = struct('Text', cellstr("h" + (1:4)), 'TokenIds', {[]}, 'Score', num2cell(sc));
            tc.Viz.plot(ax, "BeamCandidates", struct('Candidates', c));
            tc.verifyTrue(all(isfinite(ax.XLim)));
            tc.verifyLessThan(ax.XLim(1), -1.1);
            txt = findobj(ax, 'Type', 'text');
            tc.verifyTrue(any(contains(string({txt.String}), "-Inf")));
        end
    end
end
