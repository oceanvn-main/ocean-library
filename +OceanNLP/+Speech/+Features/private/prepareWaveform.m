function [x, w] = prepareWaveform(input, sampleRate, className)
%PREPAREWAVEFORM  Shared input policy of the spectral feature transforms.
%   Rejects a non-Waveform (':InputType') and a rate mismatch
%   (':SampleRateMismatch' -- no silent resampling), mixes multichannel audio
%   with toMono("mean") (recorded in the waveform History), and returns the
%   mono column x and the (possibly derived) mono waveform w.
arguments
    input
    sampleRate (1,1) double
    className (1,1) string
end
prefix = "OceanNLP:Speech:Features:" + className;
if ~isa(input, 'OceanNLP.Speech.Core.Waveform')
    error(prefix + ":InputType", '%s.forward expects an OceanNLP.Speech.Core.Waveform, got %s.', ...
        className, class(input));
end
if input.SampleRate ~= sampleRate
    error(prefix + ":SampleRateMismatch", ...
        '%s is configured for %g Hz but the waveform is %g Hz; resample it explicitly (OceanMath.Signal.ResampleSignal).', ...
        className, sampleRate, input.SampleRate);
end
w = input.toMono("mean");
x = w.Samples;
end
