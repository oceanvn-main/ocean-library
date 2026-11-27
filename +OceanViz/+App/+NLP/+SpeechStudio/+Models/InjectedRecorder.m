classdef InjectedRecorder < OceanViz.App.NLP.SpeechStudio.Models.IAudioRecorder
    % INJECTEDRECORDER  IAudioRecorder that "captures" given real samples (no device).
    %
    %   r = OceanViz.App.NLP.SpeechStudio.Models.InjectedRecorder(x, fs, 'Name', "0_theo_0.wav")
    %   Each level() call advances the capture position by 0.1 s and returns the RMS
    %   level of that chunk, as a live meter would; stop() returns all the samples.
    %   Used by the app tests to drive the microphone path with a real recording.

    properties (SetAccess = private)
        Samples (:,1) double
        SampleRate (1,1) double
        Name (1,1) string
        Position (1,1) double = 0
        Recording (1,1) logical = false
    end

    methods
        function obj = InjectedRecorder(samples, sampleRate, opts)
            arguments
                samples {mustBeNumeric, mustBeNonempty}
                sampleRate (1,1) double {mustBePositive}
                opts.Name (1,1) string = "injected samples"
            end
            obj.Samples = double(samples(:, 1));
            obj.SampleRate = sampleRate;
            obj.Name = opts.Name;
        end

        function names = devices(obj)
            names = "Injected: " + obj.Name;
        end

        function start(obj, ~, ~)
            obj.Position = 0;
            obj.Recording = true;
        end

        function [x, fs] = stop(obj)
            obj.Recording = false;
            x = obj.Samples;
            fs = obj.SampleRate;
        end

        function dbfs = level(obj)
            n = round(0.1 * obj.SampleRate);
            S = numel(obj.Samples);
            a = min(obj.Position + 1, S);
            b = min(obj.Position + n, S);
            obj.Position = mod(b, S);
            dbfs = obj.rmsDBFS(obj.Samples(a:b));
        end

        function tf = isRecording(obj)
            tf = obj.Recording;
        end
    end
end
