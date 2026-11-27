classdef ElmanCell < OceanDL.Recurrent.IRecurrentCell
    % ELMANCELL  Vanilla ("simple") RNN cell -- Goodfellow Eq. 10.8-10.9:
    %     a_t = b + W h_{t-1} + U x_t
    %     h_t = tanh(a_t)            (or ReLU: the IRNN of Le et al. 2015)
    %   Named after Elman (1990). PyTorch nn.RNN / Keras SimpleRNN / Flax SimpleCell.
    %
    %   Parameters (book names -> properties):
    %     U  input-to-hidden   (H x D)   -> Wx
    %     W  hidden-to-hidden  (H x H)   -> Wh      (the matrix whose spectrum
    %                                               decides vanishing/exploding, Sec. 10.7)
    %     b  bias              (H x 1)   -> b       (PyTorch keeps two biases b_ih + b_hh;
    %                                               they are redundant -- one here)
    %   Init U(-1/sqrt(H), 1/sqrt(H)), as PyTorch.

    properties (SetAccess = protected)
        Wx
        Wh
        b
        Nonlinearity (1,1) string = "tanh"
    end

    methods
        function obj = ElmanCell(inputSize, hiddenSize, opts)
            arguments
                inputSize (1,1) double {mustBePositive, mustBeInteger}
                hiddenSize (1,1) double {mustBePositive, mustBeInteger}
                opts.Nonlinearity (1,1) string {mustBeMember(opts.Nonlinearity, ["tanh", "relu"])} = "tanh"
                opts.Seed = []
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["double", "single"])} = "double"
            end
            obj.InputSize = inputSize;
            obj.HiddenSize = hiddenSize;
            obj.Nonlinearity = opts.Nonlinearity;
            s = obj.stream(opts.Seed);
            k = 1 / sqrt(hiddenSize);
            obj.Wx = obj.uniformParam(hiddenSize, inputSize, k, s, opts.DataType);
            obj.Wh = obj.uniformParam(hiddenSize, hiddenSize, k, s, opts.DataType);
            obj.b = obj.uniformParam(hiddenSize, 1, k, s, opts.DataType);
        end

        function [h, state] = step(obj, x, state)
            a = obj.Wx * x + obj.Wh * state.h + obj.b;
            if obj.Nonlinearity == "tanh"
                h = tanh(a);
            else
                h = relu(a);
            end
            state.h = h;
        end
    end
end
