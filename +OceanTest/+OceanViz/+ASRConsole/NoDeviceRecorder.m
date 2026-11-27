classdef NoDeviceRecorder < OceanViz.App.NLP.SpeechStudio.Models.IAudioRecorder
    % NODEVICERECORDER  Test double: a computer with no audio input device (gate 1 of the ASR Console).
    %   devices() lists nothing; start() fails the way the real recorder does without a device.

    methods
        function names = devices(~)
            names = strings(1, 0);
        end

        function start(~, ~, ~)
            error('OceanTest:NoDevice', 'No audio input device.');
        end

        function [x, fs] = stop(~)
            x = zeros(0, 1);
            fs = 16000;
        end

        function dbfs = level(~)
            dbfs = -120;
        end

        function tf = isRecording(~)
            tf = false;
        end
    end
end
