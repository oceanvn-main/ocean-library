classdef RecurrentEncoder < OceanDL.nn.Module
    % RECURRENTENCODER  (Bi)directional stacked GRU / LSTM over padded feature batches.
    %
    %   enc = OceanNLP.Speech.Models.Encoders.RecurrentEncoder(inputDim, 'HiddenSize', 128, ...
    %           'NumLayers', 2, 'Cell', "gru", 'Bidirectional', true, 'Dropout', 0, 'Seed', 0)
    %   [H, lens] = enc.forward(X, lens)   X: F x N x T -> H: OutputDim x N x T (lens unchanged)
    %   enc.outputLength(lens) == lens;  enc.OutputDim = HiddenSize * (1 + Bidirectional)
    %
    %   Built from the existing OceanDL.Recurrent cells (GRUCell "pytorch" variant /
    %   LSTMCell), RecurrentLayer, Bidirectional ("concat") and StackedRecurrent
    %   (Goodfellow et al. 2016, Sec. 10.3, 10.5; Graves et al. 2013). For one layer
    %   and direction, at valid steps t <= T(n):
    %     GRU:  r = s(Wr x_t + Ur h + b),  z = s(Wz x_t + Uz h + b),
    %           n = tanh(Wn x_t + bn + r .* (Un h + bun)),  h_t = (1 - z) .* n + z .* h_{t-1}
    %   and at padded steps (mask m_t = 0) the state is carried and the output is 0:
    %     state_t = m_t state(x_t, state_{t-1}) + (1 - m_t) state_{t-1}.
    %   The backward direction runs t = T..1, so padded steps come first and leave
    %   h_0 = 0 untouched: utterance n's backward pass starts at its own last valid
    %   frame. Valid outputs are therefore independent of the padding (tested).
    %   Inter-layer dropout: StackedRecurrent variational dropout (global RNG; the
    %   trainer owns and checkpoints the global RNG state).
    %
    %   Initialisation: cell (layer l, direction d in {1 fwd, 2 bwd}) uses the
    %   RandStream seed Seed + 2(l-1) + d - 1.

    properties (SetAccess = protected)
        Stack
        InputDim (1,1) double
        HiddenSize (1,1) double
        NumLayers (1,1) double
        CellType (1,1) string
        IsBidirectional (1,1) logical
        DropoutP (1,1) double
        Seed (1,1) double
    end

    properties (Dependent)
        OutputDim
    end

    methods
        function obj = RecurrentEncoder(inputDim, opts)
            arguments
                inputDim (1,1) double {mustBePositive, mustBeInteger}
                opts.HiddenSize (1,1) double {mustBePositive, mustBeInteger} = 128
                opts.NumLayers (1,1) double {mustBePositive, mustBeInteger} = 2
                opts.Cell (1,1) string {mustBeMember(opts.Cell, ["gru", "lstm"])} = "gru"
                opts.Bidirectional (1,1) logical = true
                opts.Dropout (1,1) double {mustBeGreaterThanOrEqual(opts.Dropout, 0), mustBeLessThan(opts.Dropout, 1)} = 0
                opts.Seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
            end
            obj.InputDim = inputDim;
            obj.HiddenSize = opts.HiddenSize;
            obj.NumLayers = opts.NumLayers;
            obj.CellType = opts.Cell;
            obj.IsBidirectional = opts.Bidirectional;
            obj.DropoutP = opts.Dropout;
            obj.Seed = opts.Seed;
            H = opts.HiddenSize;
            layers = cell(1, opts.NumLayers);
            din = inputDim;
            for l = 1:opts.NumLayers
                base = opts.Seed + 2 * (l - 1);
                if opts.Bidirectional
                    counter = containers.Map({'next'}, {base});
                    layers{l} = OceanDL.Recurrent.Bidirectional( ...
                        @() makeCell(opts.Cell, din, H, nextSeed(counter)), 'Merge', "concat");
                    din = 2 * H;
                else
                    layers{l} = OceanDL.Recurrent.RecurrentLayer(makeCell(opts.Cell, din, H, base));
                    din = H;
                end
            end
            obj.Stack = OceanDL.Recurrent.StackedRecurrent(layers, 'Dropout', opts.Dropout, 'Variational', true);
        end

        function d = get.OutputDim(obj)
            d = obj.HiddenSize * (1 + obj.IsBidirectional);
        end

        function [H, lens] = forward(obj, X, lens)
            arguments
                obj
                X
                lens (1,:) double {mustBeNonnegative, mustBeInteger}
            end
            [F, N, T] = size(X, 1:3);
            if F ~= obj.InputDim
                error('OceanNLP:Speech:Models:RecurrentEncoder:InputDim', ...
                    'Expected %d input features (F x N x T), got %d.', obj.InputDim, F);
            end
            if numel(lens) ~= N || any(lens > T)
                error('OceanNLP:Speech:Models:RecurrentEncoder:Lengths', ...
                    'lens must hold N = %d lengths <= T = %d.', N, T);
            end
            M = OceanDL.Layout.SequenceMask.fromLengths(lens, T);
            X = OceanDL.Layout.SequenceMask.apply(OceanDL.Autograd.Tensor.wrap(X), M);
            if T == 0
                H = OceanDL.Autograd.Tensor(zeros(obj.OutputDim, N, 0, 'like', OceanDL.Autograd.Tensor.raw(X)));
                return;
            end
            H = obj.Stack.forward(X, 'Mask', M);
            H = OceanDL.Autograd.Tensor.wrap(H);
        end

        function lens = outputLength(~, lens)
            arguments
                ~
                lens double {mustBeNonnegative, mustBeInteger}
            end
        end

        function cfg = config(obj)
            cfg = struct('Type', "RecurrentEncoder", 'InputDim', obj.InputDim, 'HiddenSize', obj.HiddenSize, ...
                'NumLayers', obj.NumLayers, 'Cell', obj.CellType, 'Bidirectional', obj.IsBidirectional, ...
                'Dropout', obj.DropoutP, 'Seed', obj.Seed, 'OutputDim', obj.OutputDim);
        end
    end

    methods (Static)
        function obj = fromConfig(cfg)
            arguments
                cfg (1,1) struct
            end
            if string(cfg.Type) ~= "RecurrentEncoder"
                error('OceanNLP:Speech:Models:RecurrentEncoder:ConfigType', 'Config Type is "%s", not "RecurrentEncoder".', string(cfg.Type));
            end
            obj = OceanNLP.Speech.Models.Encoders.RecurrentEncoder(cfg.InputDim, 'HiddenSize', cfg.HiddenSize, ...
                'NumLayers', cfg.NumLayers, 'Cell', cfg.Cell, 'Bidirectional', cfg.Bidirectional, ...
                'Dropout', cfg.Dropout, 'Seed', cfg.Seed);
        end
    end
end

function c = makeCell(type, din, H, seed)
if type == "gru"
    c = OceanDL.Recurrent.GRUCell(din, H, 'Seed', seed);
else
    c = OceanDL.Recurrent.LSTMCell(din, H, 'Seed', seed);
end
end

function s = nextSeed(counter)
% Successive seeds for the two directions (containers.Map is a handle).
s = counter('next');
counter('next') = s + 1;
end
