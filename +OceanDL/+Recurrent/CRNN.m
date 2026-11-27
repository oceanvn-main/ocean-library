classdef CRNN < OceanDL.nn.Module
    % CRNN  Convolutional Recurrent Neural Network for image-based sequence
    %   recognition (Shi, Bai & Yao 2017), trained with CTC.
    %
    %   image (H x W x 1 x N)
    %     -> CNN backbone: [conv3x3 -> ReLU -> maxpool 2x2] x 2   feature map (H/4 x W/4 x C2 x N)
    %     -> columns as time: each of the W/4 columns is one step
    %        (ImageSequence "columns": D = (H/4) * C2, T = W/4)       "CBT"
    %     -> bidirectional GRU (every column sees the whole line)     2*Hidden x N x T
    %     -> linear per step                                          K x N x T logits
    %   K = number of symbols + 1 (class Blank = 1 for CTC).
    %
    %   The CNN turns pixels into local shape features; the BiRNN turns the
    %   column sequence into per-column class scores using left AND right
    %   context; CTC (F.ctcLoss) aligns those per-column scores with the
    %   unaligned label string; CTCDecoder.greedy reads it back.

    properties (SetAccess = protected)
        Backbone
        Encoder
        Head
        NumClasses (1,1) double
    end

    methods
        function obj = CRNN(imageHeight, numClasses, opts)
            arguments
                imageHeight (1,1) double {mustBePositive, mustBeInteger}
                numClasses (1,1) double {mustBePositive, mustBeInteger}        % including the blank
                opts.Channels (1,2) double = [32 64]
                opts.Hidden (1,1) double = 96
                opts.Bidirectional (1,1) logical = true
                opts.Seed = []
            end
            if mod(imageHeight, 4)
                error('OceanDL:Recurrent:CRNN:Height', 'Image height must be divisible by 4 (two 2x2 pools).');
            end
            import OceanDL.nn.*
            seed = @(k) OceanDL.Recurrent.CRNN.offset(opts.Seed, k);
            c1 = opts.Channels(1); c2 = opts.Channels(2);
            obj.Backbone = Sequential( ...
                Conv2d(1, c1, 3, 'Padding', 1, 'Seed', seed(1)), Activation("relu"), ...
                Legacy(OceanDL.Pooling.MaxPool2D([2 2])), ...
                Conv2d(c1, c2, 3, 'Padding', 1, 'Seed', seed(2)), Activation("relu"), ...
                Legacy(OceanDL.Pooling.MaxPool2D([2 2])));
            D = (imageHeight / 4) * c2;
            if opts.Bidirectional
                obj.Encoder = OceanDL.Recurrent.Bidirectional( ...
                    @() OceanDL.Recurrent.GRUCell(D, opts.Hidden, 'Seed', seed(3)));
                out = 2 * opts.Hidden;
            else
                obj.Encoder = OceanDL.Recurrent.RecurrentLayer( ...
                    OceanDL.Recurrent.GRUCell(D, opts.Hidden, 'Seed', seed(3)));
                out = opts.Hidden;
            end
            obj.Head = Linear(out, numClasses, 'Seed', seed(4));
            obj.NumClasses = numClasses;
        end

        function logits = forward(obj, X)
            F = obj.Backbone.forward(X);                                        % H/4 x W/4 x C2 x N
            S = OceanDL.Recurrent.ImageSequence.toSequence(F, "columns");   % D x N x T
            Y = obj.Encoder.forward(S);
            logits = obj.Head.forward(Y);                                       % K x N x T
        end
    end

    methods (Static, Access = private)
        function s = offset(seed, k)
            if isempty(seed), s = []; else, s = seed + k; end
        end
    end
end
