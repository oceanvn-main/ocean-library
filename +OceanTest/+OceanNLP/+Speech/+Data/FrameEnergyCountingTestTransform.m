classdef FrameEnergyCountingTestTransform < OceanTest.OceanNLP.Speech.Data.FrameEnergyTestTransform
    % FRAMEENERGYCOUNTINGTESTTRANSFORM  FrameEnergyTestTransform + numFrames(S).
    %   numFrames(S) = S >= L ? 1 + floor((S - L) / H) : 0 -- the hook that lets
    %   AudioDataset.lengths() count frames from manifest durations alone.

    methods
        function obj = FrameEnergyCountingTestTransform(sampleRate)
            obj@OceanTest.OceanNLP.Speech.Data.FrameEnergyTestTransform(sampleRate);
        end

        function T = numFrames(obj, S)
            T = obj.countFrames(S);
        end
    end
end
