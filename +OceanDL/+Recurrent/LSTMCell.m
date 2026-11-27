classdef LSTMCell < OceanDL.Recurrent.IRecurrentCell
    % LSTMCELL  Long Short-Term Memory (Hochreiter & Schmidhuber 1997;
    %   forget gate: Gers et al. 2000) -- Goodfellow Sec. 10.10.1.
    %
    %     z   = Wx x_t + Wh h_{t-1} + b          (4H x N), gate blocks in PyTorch order
    %     i   = sigmoid(z_i)   input gate        what to write
    %     f   = sigmoid(z_f)   forget gate       what to keep
    %     g   = tanh(z_g)      candidate         what could be written
    %     o   = sigmoid(z_o)   output gate       what to expose
    %     c_t = f .* c_{t-1} + i .* g            the cell state: a LINEAR, gated
    %                                            path through time -- when f ~ 1,
    %                                            dc_t/dc_{t-1} ~ 1, so gradient
    %                                            neither vanishes nor explodes
    %     h_t = o .* tanh(c_t)
    %
    %   Gate order i, f, g, o and the (4H x D)/(4H x H) weight shapes match
    %   PyTorch nn.LSTM (b here = b_ih + b_hh), so its weights import directly.
    %   ForgetBias (default 1, Keras' unit_forget_bias; Jozefowicz et al. 2015)
    %   is added to the forget-gate bias so the cell starts by REMEMBERING.
    %
    %   RecordGates = true stores, per step, the fraction of saturated gate
    %   units (< 0.05 or > 0.95) in GateSaturation -- a diagnostic for
    %   gates stuck open/closed.

    properties (SetAccess = protected)
        Wx
        Wh
        b
    end

    properties
        RecordGates (1,1) logical = false
        GateSaturation = zeros(0, 3)     % rows: steps; cols: [input forget output]
    end

    methods
        function obj = LSTMCell(inputSize, hiddenSize, opts)
            arguments
                inputSize (1,1) double {mustBePositive, mustBeInteger}
                hiddenSize (1,1) double {mustBePositive, mustBeInteger}
                opts.ForgetBias (1,1) double = 1
                opts.Seed = []
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["double", "single"])} = "double"
            end
            obj.InputSize = inputSize;
            obj.HiddenSize = hiddenSize;
            H = hiddenSize;
            s = obj.stream(opts.Seed);
            k = 1 / sqrt(H);
            obj.Wx = obj.uniformParam(4*H, inputSize, k, s, opts.DataType);
            obj.Wh = obj.uniformParam(4*H, H, k, s, opts.DataType);
            obj.b = obj.uniformParam(4*H, 1, k, s, opts.DataType);
            obj.b.Data(H+1:2*H) = obj.b.Data(H+1:2*H) + opts.ForgetBias;
        end

        function state = initialState(obj, N, like)
            if nargin < 3, like = 0; end
            z = zeros(obj.HiddenSize, N, 'like', OceanDL.Autograd.Tensor.raw(like));
            state = struct('h', z, 'c', z);
        end

        function [h, state] = step(obj, x, state)
            H = obj.HiddenSize;
            z = obj.Wx * x + obj.Wh * state.h + obj.b;
            i = sigmoid(z(1:H, :));
            f = sigmoid(z(H+1:2*H, :));
            g = tanh(z(2*H+1:3*H, :));
            o = sigmoid(z(3*H+1:4*H, :));
            c = f .* state.c + i .* g;
            h = o .* tanh(c);
            state.h = h;
            state.c = c;
            if obj.RecordGates
                sat = @(t) mean(t.Data < 0.05 | t.Data > 0.95, 'all');
                obj.GateSaturation(end+1, :) = [sat(i), sat(f), sat(o)];
            end
        end
    end
end
