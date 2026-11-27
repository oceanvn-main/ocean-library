classdef SpeechCheckTest < matlab.unittest.TestCase
    % SPEECHCHECKTEST  Gate 2 of the ASR Console: real speech passes; silence, quiet noise, a pure tone, plain
    %   noise and too-short speech are refused with the right reason. Speech = Audio Toolbox's sample recording.

    methods (Static)
        function [x, fs] = speech()
            [x, fs] = audioread("FemaleSpeech-16-8-mono-3secs.wav");
            x = 0.5 * x(:, 1) / max(abs(x(:, 1)));
        end
    end

    methods (Test)
        function realSpeechIsUsable(tc)
            [x, fs] = OceanTest.OceanViz.ASRConsole.SpeechCheckTest.speech();
            r = OceanViz.App.NLP.ASRConsole.Models.SpeechCheck.assess(x, fs);
            tc.verifyTrue(r.Usable, r.Reason);
            tc.verifyGreaterThan(r.SpeechSeconds, 1, 'a 3 s reading has more than 1 s of speech');
            tc.verifyLessThanOrEqual(r.SpeechSeconds, numel(x) / fs);
            tc.verifyEqual(r.Method, "detectspeechnn");
        end

        function refusalsCarryTheRightReason(tc)
            S = "OceanViz.App.NLP.ASRConsole.Models.SpeechCheck";
            rng(0);
            [sp, fs] = OceanTest.OceanViz.ASRConsole.SpeechCheckTest.speech();
            t = (0:2 * 16000 - 1)' / 16000;
            cases = {zeros(0, 1), 16000, "No audio"; ...
                zeros(16000, 1), 16000, "pure silence"; ...
                1e-4 * randn(16000, 1), 16000, "far too quiet"; ...
                0.25 * sin(2 * pi * 220 * t), 16000, "No speech detected"; ...
                0.02 * randn(32000, 1), 16000, "No speech detected"; ...
                sp(1:round(0.2 * fs)), fs, "No speech detected"};
            for k = 1:size(cases, 1)
                r = feval(S + ".assess", cases{k, 1}, cases{k, 2});
                tc.verifyFalse(r.Usable, "case " + k);
                tc.verifySubstring(r.Reason, cases{k, 3}, "case " + k);
            end
        end

        function moreSpeechNeverTurnsAPassIntoARefusal(tc)
            % metamorphic: appending the same speech again keeps it usable and does not reduce the speech found
            [x, fs] = OceanTest.OceanViz.ASRConsole.SpeechCheckTest.speech();
            a = OceanViz.App.NLP.ASRConsole.Models.SpeechCheck.assess(x, fs);
            b = OceanViz.App.NLP.ASRConsole.Models.SpeechCheck.assess([x; x], fs);
            tc.verifyTrue(a.Usable && b.Usable);
            tc.verifyGreaterThanOrEqual(b.SpeechSeconds, a.SpeechSeconds);
        end
    end
end
