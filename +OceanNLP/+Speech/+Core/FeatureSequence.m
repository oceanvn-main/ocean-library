classdef FeatureSequence < handle
    % FEATURESEQUENCE  A feature matrix with its axes, units and provenance (immutable).
    %
    %   f = OceanNLP.Speech.Core.FeatureSequence(V, 'FrameTimes', t, 'FrequencyHz', fc, ...
    %         'AxisNames', ["mel","time"], 'Units', "log power", 'Config', cfg, 'Source', src)
    %
    %   Values is F x T: one column per analysis frame (feature bins x time),
    %   so f.toCBT() is the F x 1 x T "CBT" block OceanDL models consume.
    %   FrameTimes(t) is the centre of frame t in seconds (FrameGrid convention,
    %   Phase 0 contract sec. 3). FrequencyHz is the centre frequency of each bin
    %   when axis 1 is a frequency axis (mel / linear), [] otherwise (MFCC, deltas).
    %   Config holds the full transform configuration that produced Values.
    %   A public feature extractor never returns an anonymous matrix.

    properties (SetAccess = immutable)
        Values (:,:) double
        FrameTimes (1,:) double
        FrequencyHz (:,1) double
        AxisNames (1,2) string
        Units (1,1) string
        Config struct
        Source (1,1) string
    end

    methods
        function obj = FeatureSequence(values, opts)
            arguments
                values (:,:) {mustBeNumeric, mustBeReal}
                opts.FrameTimes (1,:) double = []
                opts.FrequencyHz (:,1) double = []
                opts.AxisNames (1,2) string = ["feature", "time"]
                opts.Units (1,1) string = ""
                opts.Config struct = struct()
                opts.Source (1,1) string = ""
            end
            [F, T] = size(values);
            if ~all(isfinite(values), 'all')
                error('OceanNLP:Speech:Core:FeatureSequence:Nonfinite', 'Feature values must be finite.');
            end
            if numel(opts.FrameTimes) ~= T
                error('OceanNLP:Speech:Core:FeatureSequence:FrameTimes', ...
                    '%d frame times for %d frames.', numel(opts.FrameTimes), T);
            end
            if ~isempty(opts.FrequencyHz) && numel(opts.FrequencyHz) ~= F
                error('OceanNLP:Speech:Core:FeatureSequence:FrequencyHz', ...
                    '%d bin frequencies for %d bins.', numel(opts.FrequencyHz), F);
            end
            obj.Values = double(values);
            obj.FrameTimes = opts.FrameTimes;
            obj.FrequencyHz = opts.FrequencyHz;
            obj.AxisNames = opts.AxisNames;
            obj.Units = opts.Units;
            obj.Config = opts.Config;
            obj.Source = opts.Source;
        end

        function T = numFrames(obj)
            T = size(obj.Values, 2);
        end

        function F = numBins(obj)
            F = size(obj.Values, 1);
        end

        function X = toCBT(obj)
            % TOCBT  F x 1 x T block (features x batch x time).
            % Values already stores F x T in column-major frame order.  Reshape
            % inserts a singleton batch axis; it does not reorder feature values.
            X = reshape(obj.Values, obj.numBins(), 1, obj.numFrames());
        end

        function f = derive(obj, values, opts)
            % DERIVE  New sequence on the same time axis (e.g. deltas, normalisation).
            arguments
                obj
                values (:,:) double
                opts.FrequencyHz (:,1) double = []
                opts.AxisNames (1,2) string = obj.AxisNames
                opts.Units (1,1) string = obj.Units
                opts.Config struct = obj.Config
            end
            % Reusing FrameTimes makes this a time-preserving transform.  The
            % constructor rejects a different number of columns.  FrequencyHz is
            % deliberately opt-in because row-changing transforms (MFCC/deltas)
            % may no longer have the original frequency coordinate.
            f = OceanNLP.Speech.Core.FeatureSequence(values, 'FrameTimes', obj.FrameTimes, ...
                'FrequencyHz', opts.FrequencyHz, 'AxisNames', opts.AxisNames, 'Units', opts.Units, ...
                'Config', opts.Config, 'Source', obj.Source);
        end
    end
end
