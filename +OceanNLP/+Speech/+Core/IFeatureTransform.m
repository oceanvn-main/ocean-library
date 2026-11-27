classdef (Abstract) IFeatureTransform < handle
    % IFEATURETRANSFORM  Strategy interface for speech feature transforms.
    %
    %   fs  = t.forward(input)        input: Core.Waveform (spectral transforms)
    %                                        or Core.FeatureSequence (deltas, ...)
    %                                 output: Core.FeatureSequence (never a bare matrix)
    %   cfg = t.config()              struct with every parameter (recorded in fs.Config;
    %                                 must contain a 'Type' field naming the class)
    %   d   = t.outputDim(inputDim)   feature bins produced (inputDim ignored by
    %                                 transforms whose output size is fixed)
    %
    %   A transform rejects the wrong input kind with an error id ending in
    %   ':InputType' and never resamples or reshapes silently.

    methods (Abstract)
        fs = forward(obj, input)
        cfg = config(obj)
        d = outputDim(obj, inputDim)
    end
end
