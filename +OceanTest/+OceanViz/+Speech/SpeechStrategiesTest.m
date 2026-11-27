classdef SpeechStrategiesTest < OceanTest.BaseTestCase
    % SPEECHSTRATEGIESTEST  Every speech strategy renders REAL data (the trained CTC model on a
    %   held-out FSDD utterance, via SpeechProcessAdapter) through OceanViz.Core.plot into an
    %   invisible figure, with axis labels carrying units, themed legends / colour bars.

    properties
        Viz
        Reader
        Data struct
        Fig
    end

    methods (TestClassSetup)
        function runRealPipeline(tc)
            file = fullfile(tc.LibraryRoot, 'Results', 'Speech', 'checkpoints', 'ctc_fsdd_best.mat');
            tc.assumeTrue(isfile(file), 'Trained checkpoint not present.');
            ck = OceanNLP.Speech.Training.SpeechCheckpoint.load(string(file));
            root = OceanNLP.Speech.Data.SpeechManifest.defaultRoot();
            w = OceanNLP.Speech.Core.Waveform.read("FSDD/recordings/3_theo_7.wav", 'Root', root);
            [tr, r] = OceanNLP.Speech.Inspect.SpeechProcessAdapter.run(ck.Model, w, 'Decoder', "beam");
            tc.Reader = r;
            steps = r.listSteps();
            get = @(n) r.getArtifact(steps([steps.Name] == n).OutputIds(1)).Value;
            d = struct();
            d.SpeechWaveform = get("Waveform");
            fr = get("Frames + window");
            d.SpeechWaveform.FrameStarts = fr.FrameStarts;
            d.SpeechWaveform.WindowLength = fr.WindowLength;
            d.Spectrogram = get("Power spectrum");
            d.MelFilterBank = get("Mel filter bank");
            d.SpeechFeatures = get("Log-mel");
            d.CTCPosterior = get("Posteriors");
            d.BeamCandidates = get("Beam candidates");
            pc = OceanNLP.Speech.Features.PitchContour(w.SampleRate, 'WindowLength', fr.WindowLength, ...
                'HopLength', fr.HopLength, 'NFFT', fr.NFFT);
            d.PitchContour = OceanViz.Strategies.Speech.PitchContourStrategy.fromFeatureSequence( ...
                pc.forward(w), 'VoicingThreshold', pc.VoicingThreshold);
            ref = string(num2cell('three'));
            hyp = string(num2cell(char(tr.Text)));
            if tr.Text == "", hyp = strings(1, 0); end
            d.ErrorAlignment = struct('Alignment', OceanNLP.Speech.Evaluation.EditAlignment.align(ref, hyp), ...
                'Reference', ref, 'Hypothesis', hyp, 'Unit', "character");
            tc.Data = d;
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
        function waveformHasTimeAxisAndLegend(tc)
            ax = tc.render("SpeechWaveform");
            tc.verifySubstring(string(ax.XLabel.String), "(s)");
            tc.verifySubstring(string(ax.YLabel.String), "Amplitude");
            line = findobj(ax, 'Type', 'line');
            tc.verifyEqual(numel(line.YData), size(tc.Data.SpeechWaveform.Samples, 1));
            tc.verifyEqual(line.YData(:), tc.Data.SpeechWaveform.Samples(:, 1));
            tc.verifyNotEmpty(ax.Legend);
            tc.verifyEqual(ax.Legend.TextColor, tc.Viz.Colors.Text);
        end

        function spectrogramIsDecibelsOfRealPower(tc)
            ax = tc.render("Spectrogram");
            img = findobj(ax, 'Type', 'image');
            P = tc.Data.Spectrogram.Values;
            tc.verifyEqual(max(img.CData(:)), 10 * log10(max(P(:))), 'AbsTol', 1e-9);
            tc.verifyEqual(min(img.CData(:)), 10 * log10(max(P(:))) - 80, 'AbsTol', 1e-9);
            tc.verifyEqual(string(ax.YLabel.String), "Frequency (Hz)");
            tc.verifySubstring(string(ax.Colorbar.Label.String), "dB");
        end

        function melBankDrawsEveryFilter(tc)
            ax = tc.render("MelFilterBank");
            W = tc.Data.MelFilterBank.Weights;
            lines = findall(ax, 'Type', 'line', 'Marker', 'none');   % includes legend-hidden filters
            tc.verifyEqual(numel(lines), size(W, 1));
            tc.verifyEqual(string(ax.XLabel.String), "Frequency (Hz)");
            tc.verifyNotEmpty(ax.Legend);
            tc.verifySubstring(string(ax.Title.String), "htk");
        end

        function featureMatrixShowsValuesUnchanged(tc)
            ax = tc.render("SpeechFeatures");
            img = findobj(ax, 'Type', 'image');
            tc.verifyEqual(img.CData, tc.Data.SpeechFeatures.Values);
            tc.verifyEqual(img.XData([1 end]), tc.Data.SpeechFeatures.FrameTimes([1 end]));
            tc.verifySubstring(string(ax.YLabel.String), "Hz");
            tc.verifySubstring(string(ax.Colorbar.Label.String), "ln(power");
        end

        function pitchShowsVoicedF0AndVoicing(tc)
            ax = tc.render("PitchContour");
            d = tc.Data.PitchContour;
            yyaxis(ax, 'left');
            tc.verifyEqual(string(ax.YLabel.String), "F0 (Hz)");
            f0 = findobj(ax, 'Type', 'line', 'Marker', 'o');
            tc.verifyEqual(f0.YData, d.F0Hz(d.F0Hz > 0));
            tc.verifyGreaterThan(numel(f0.YData), 0, 'A spoken digit has voiced frames.');
            tc.verifyNotEmpty(ax.Legend);
        end

        function posteriorHeatmapLabelsTokens(tc)
            ax = tc.render("CTCPosterior");
            img = findobj(ax, 'Type', 'image');
            tc.verifyEqual(img.CData, tc.Data.CTCPosterior.Posteriors);
            tc.verifyEqual(string(ax.YTickLabel(:))', tc.Data.CTCPosterior.Tokens);
            tc.verifySubstring(string(ax.XLabel.String), "(s)");
            tc.verifyNotEmpty(ax.Legend);
        end

        function beamCandidatesBarsAreScores(tc)
            ax = tc.render("BeamCandidates");
            bars = findobj(ax, 'Type', 'bar');
            ys = []; xs = [];
            for b = bars'
                ys = [ys, b.XData]; xs = [xs, b.YData]; %#ok<AGROW>
            end
            [~, o] = sort(ys);
            tc.verifyEqual(xs(o), [tc.Data.BeamCandidates.Candidates.Score]);
            tc.verifySubstring(string(ax.XLabel.String), "nats");
            tc.verifyNotEmpty(ax.Legend);
        end

        function errorAlignmentReportsRate(tc)
            ax = tc.render("ErrorAlignment");
            a = tc.Data.ErrorAlignment.Alignment;
            tc.verifySubstring(string(ax.Title.String), "CER");
            tc.verifySubstring(string(ax.Title.String), sprintf("%d S", a.Substitutions));
            txt = findobj(ax, 'Type', 'text');
            tc.verifyEqual(numel(txt), 2 * numel(a.Ops));
            tc.verifyNumElements(ax.Legend.String, 4);
        end

        function errorIds(tc)
            ax = axes(tc.Fig);
            bad = tc.Data.Spectrogram; bad.Values = -bad.Values;
            tc.verifyError(@() tc.Viz.plot(ax, "Spectrogram", bad), ...
                'OceanViz:Strategies:Speech:SpectrogramStrategy:NegativePower');
            tc.verifyError(@() tc.Viz.plot(ax, "CTCPosterior", struct('Tokens', "a")), ...
                'OceanViz:Strategies:Speech:MissingField');
            s = OceanViz.Strategies.Speech.WaveformStrategy();
            tc.verifyError(@() s.plot(ax, tc.Data.SpeechWaveform), 'OceanViz:Strategies:Speech:NoParent');
        end
    end

    methods
        function ax = render(tc, name)
            ax = axes(tc.Fig);
            tc.Viz.plot(ax, name, tc.Data.(name));
            tc.verifyNotEmpty(ax.Children, name + " drew nothing.");
            tc.verifyNotEmpty(string(ax.Title.String));
            tc.verifyNotEqual(string(ax.XLabel.String), "");
        end
    end
end
