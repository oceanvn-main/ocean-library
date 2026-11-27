classdef (Abstract) IAudioRecorder < handle
    % IAUDIORECORDER  Microphone capture used by the Speech Pipeline Studio.
    %
    %   The studio never calls audiorecorder directly: it talks to an IAudioRecorder so
    %   tests can inject real samples (e.g. an FSDD recording) without a sound device.
    %
    %   Implement:
    %     names = devices(obj)            string array of input device names
    %     start(obj, device, maxSeconds)  begin capturing (device = one of devices())
    %     [x, fs] = stop(obj)             end capture; x = S x 1 samples in [-1, 1], fs Hz
    %     dbfs = level(obj)               RMS level (dBFS) of the most recent ~0.1 s
    %     tf = isRecording(obj)
    %   Implementations: MicrophoneRecorder (audiorecorder), InjectedRecorder (tests).

    methods (Abstract)
        names = devices(obj)
        start(obj, device, maxSeconds)
        [x, fs] = stop(obj)
        dbfs = level(obj)
        tf = isRecording(obj)
    end

    methods (Static)
        function d = rmsDBFS(x)
            % RMSDBFS  20 log10(sqrt(mean(x.^2))) floored at -120 dBFS (full scale = 1).
            r = sqrt(mean(double(x(:)).^2));
            d = 20 * log10(max(r, 1e-6));
        end
    end
end
