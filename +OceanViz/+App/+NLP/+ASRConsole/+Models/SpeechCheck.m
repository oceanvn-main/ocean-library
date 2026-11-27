classdef SpeechCheck
    % SPEECHCHECK  Is there real speech in this audio? Gate 2 of the ASR Console: no model runs on audio that
    %   fails it (user requirement 2026-10-20: "if [the microphone] doesn't exist, then the model shouldn't run
    %   at all to save resource").
    %
    %   r = OceanViz.App.NLP.ASRConsole.Models.SpeechCheck.assess(x, fs)
    %   r.Usable (logical), r.Reason (string, empty when usable), r.SpeechSeconds, r.PeakDBFS, r.Method
    %
    %   Order of the checks (the first failure is the reason):
    %     no samples            -> "No audio."
    %     every sample is zero  -> the microphone delivered pure silence (muted, or blocked by Windows privacy)
    %     peak below -60 dBFS   -> far too quiet to contain speech
    %     speech < MinSpeech s  -> "No speech detected"
    %   Speech is measured with, in order of availability:
    %     detectspeechnn  Audio Toolbox's pretrained neural voice-activity detector (vadnet; needs Deep Learning
    %                     Toolbox) - rejects steady tones and noise (checked: a 220 Hz tone and white noise -> 0)
    %     detectSpeech    energy + spectral-spread detector: weaker, a steady loud tone (hum, whistle) passes
    %     frame energy    fraction of 20 ms frames whose RMS reaches -40 dBFS (no toolbox)

    properties (Constant)
        MinSpeechSeconds = 0.3
        QuietFloorDBFS = -60
        FallbackFrameDBFS = -40
    end

    methods (Static)
        function r = assess(x, fs)
            arguments
                x {mustBeNumeric}
                fs (1,1) double {mustBePositive}
            end
            S = OceanViz.App.NLP.ASRConsole.Models.SpeechCheck;
            x = double(x(:));
            r = struct('Usable', false, 'Reason', "", 'SpeechSeconds', 0, 'PeakDBFS', -Inf, 'Method', "");
            if isempty(x)
                r.Reason = "No audio.";
                return
            end
            peak = max(abs(x));
            r.PeakDBFS = 20 * log10(max(peak, eps));
            if peak == 0
                r.Reason = "The microphone delivered pure silence (every sample is zero): it may be muted, " + ...
                    "or blocked for desktop apps in Windows Settings > Privacy & security > Microphone.";
                return
            end
            if r.PeakDBFS < S.QuietFloorDBFS
                r.Reason = sprintf("The audio is far too quiet to contain speech (peak %.1f dBFS): " + ...
                    "check the microphone and its input level.", r.PeakDBFS);
                return
            end
            [r.SpeechSeconds, r.Method] = S.speechSeconds(x, fs);
            if r.SpeechSeconds < S.MinSpeechSeconds
                r.Reason = sprintf("No speech detected (%.2f s of speech, at least %.1f s needed; %s). " + ...
                    "Nothing was sent to the models.", r.SpeechSeconds, S.MinSpeechSeconds, r.Method);
                return
            end
            r.Usable = true;
        end

        function [seconds, method] = speechSeconds(x, fs)
            S = OceanViz.App.NLP.ASRConsole.Models.SpeechCheck;
            if exist('detectspeechnn', 'file') == 2 && license('test', 'Neural_Network_Toolbox')
                try
                    idx = detectspeechnn(x, fs);             % rows: [first last] sample of each speech region
                    seconds = sum(idx(:, 2) - idx(:, 1) + 1) / fs;
                catch                                        % too short for the network's frames: no speech
                    seconds = 0;
                end
                method = "detectspeechnn";
            elseif exist('detectSpeech', 'file') == 2 && license('test', 'Audio_System_Toolbox')
                try
                    idx = detectSpeech(x, fs);               % rows: [first last] sample of each speech region
                    seconds = sum(idx(:, 2) - idx(:, 1) + 1) / fs;
                catch                                        % shorter than its analysis window: no speech
                    seconds = 0;
                end
                method = "detectSpeech";
            else
                n = max(1, round(0.02 * fs));
                frames = floor(numel(x) / n);
                if frames == 0
                    seconds = 0;
                else
                    rms = sqrt(mean(reshape(x(1:frames * n), n, frames) .^ 2, 1));
                    seconds = nnz(20 * log10(max(rms, eps)) >= S.FallbackFrameDBFS) * n / fs;
                end
                method = "frame energy";
            end
        end
    end
end
