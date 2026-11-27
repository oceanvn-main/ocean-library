classdef SequenceClassifier < OceanDL.nn.Module
    % SEQUENCECLASSIFIER  Many-to-one recurrent classifier (Goodfellow
    %   Fig. 10.5): encoder over the sequence -> pooled summary -> linear
    %   logits.  logits = V * pool(encoder(x)) + c
    %
    %   Input: canonical "CBT" sequences, or canonical "SSCB" images when an
    %   ImageMode is set ("rows" | "columns" | "pixels", see ImageSequence).
    %   Encoder: RecurrentLayer, Bidirectional or StackedRecurrent.
    %
    %   clf = SequenceClassifier(RecurrentLayer(LSTMCell(16, 64)), 64, 10, 'ImageMode', "rows");
    %   logits = clf.forward(images);                 % 10 x N

    properties (SetAccess = protected)
        Encoder
        Pool
        Head
        ImageMode (1,1) string = "none"
    end

    methods
        function obj = SequenceClassifier(encoder, encoderOutputSize, numClasses, opts)
            arguments
                encoder OceanDL.nn.Module
                encoderOutputSize (1,1) double {mustBePositive, mustBeInteger}
                numClasses (1,1) double {mustBePositive, mustBeInteger}
                opts.Pool (1,1) string {mustBeMember(opts.Pool, ["last", "mean", "max"])} = "last"
                opts.ImageMode (1,1) string {mustBeMember(opts.ImageMode, ["none", "rows", "columns", "pixels"])} = "none"
                opts.Seed = []
            end
            obj.Encoder = encoder;
            obj.Pool = OceanDL.Recurrent.SequencePool(opts.Pool);
            obj.Head = OceanDL.nn.Linear(encoderOutputSize, numClasses, 'Seed', opts.Seed);
            obj.ImageMode = opts.ImageMode;
        end

        function [logits, summary] = forward(obj, X, mask)
            if nargin < 3
                mask = [];
            end
            if obj.ImageMode ~= "none"
                X = OceanDL.Recurrent.ImageSequence.toSequence(X, obj.ImageMode);
            end
            Y = obj.Encoder.forward(X, 'Mask', mask);
            summary = obj.Pool.forward(Y, mask);
            logits = obj.Head.forward(summary);
        end
    end
end
