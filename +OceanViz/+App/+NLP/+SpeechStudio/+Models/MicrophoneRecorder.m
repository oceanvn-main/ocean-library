classdef MicrophoneRecorder < OceanViz.App.NLP.SpeechStudio.Models.IAudioRecorder
    % MICROPHONERECORDER  IAudioRecorder over MATLAB audiorecorder (mono, 16-bit).
    %
    %   r = OceanViz.App.NLP.SpeechStudio.Models.MicrophoneRecorder('SampleRate', 16000)
    %   The captured samples go to Inference.AudioConditioner, which resamples them to
    %   the model rate; the recording rate is therefore only a capture setting.

    properties (SetAccess = private)
        SampleRate (1,1) double
        Recorder = []
        MaxSeconds (1,1) double = 10
    end

    methods
        function obj = MicrophoneRecorder(opts)
            arguments
                opts.SampleRate (1,1) double {mustBePositive} = 16000
            end
            obj.SampleRate = opts.SampleRate;
        end

        function names = devices(~)
            names = strings(1, 0);
            try
                info = audiodevinfo;
                names = string({info.input.Name});
            catch
            end
        end

        function start(obj, device, maxSeconds)
            arguments
                obj
                device (1,1) string = ""
                maxSeconds (1,1) double {mustBePositive} = 10
            end
            id = -1;
            if device ~= ""
                try
                    info = audiodevinfo;
                    k = find(string({info.input.Name}) == device, 1);
                    if ~isempty(k), id = info.input(k).ID; end
                catch
                end
            end
            obj.Recorder = audiorecorder(obj.SampleRate, 16, 1, id);
            obj.MaxSeconds = maxSeconds;
            record(obj.Recorder, maxSeconds);
        end

        function [x, fs] = stop(obj)
            if isempty(obj.Recorder)
                error('OceanViz:App:NLP:SpeechStudio:MicrophoneRecorder:NotStarted', 'Recording was not started.');
            end
            if isrecording(obj.Recorder), stop(obj.Recorder); end
            x = getaudiodata(obj.Recorder, 'double');
            fs = obj.Recorder.SampleRate;
            obj.Recorder = [];
        end

        function dbfs = level(obj)
            dbfs = -120;
            if isempty(obj.Recorder) || obj.Recorder.TotalSamples == 0, return; end
            try
                x = getaudiodata(obj.Recorder, 'double');
                n = round(0.1 * obj.SampleRate);
                dbfs = obj.rmsDBFS(x(max(1, end - n + 1):end));
            catch
            end
        end

        function tf = isRecording(obj)
            tf = ~isempty(obj.Recorder) && isrecording(obj.Recorder);
        end

        function delete(obj)
            if ~isempty(obj.Recorder) && isrecording(obj.Recorder), stop(obj.Recorder); end
        end
    end
end
