classdef GRUCell < OceanDL.Recurrent.IRecurrentCell
    % GRUCELL  Gated Recurrent Unit (Cho et al. 2014) -- Goodfellow Sec. 10.10.2.
    %   One state vector, two gates: update z (how much of the old state to
    %   keep) and reset r (how much of it the candidate may see).
    %
    %     r   = sigmoid(Wx_r x + bx_r + Whrz_r h + bhrz_r)
    %     z   = sigmoid(Wx_z x + bx_z + Whrz_z h + bhrz_z)
    %     n   = tanh(Wx_n x + bx_n + r .* (Whn h + bhn))        Variant "pytorch"
    %         = tanh(Wx_n x + bx_n + Whn (r .* h) + bhn)        Variant "original"
    %     h_t = (1 - z) .* n + z .* h_{t-1}
    %
    %   VARIANTS (not interchangeable numerically):
    %     "original"  reset applied BEFORE the recurrent product, as in Cho et
    %                 al. 2014 and Goodfellow Eq. 10.45-10.46 (u there = z here);
    %     "pytorch"   reset applied AFTER it (PyTorch nn.GRU, cuDNN, Keras
    %                 reset_after=True) -- the default, so PyTorch weights import.
    %   Gate order r, z, n as PyTorch. Recurrent weights are stored as Whrz
    %   (2H x H, gates r and z) and Whn (H x H) because the "original" variant
    %   needs the candidate's product separately; a PyTorch weight_hh (3H x H)
    %   splits as [Whrz; Whn].

    properties (SetAccess = protected)
        Wx
        bx
        Whrz
        bhrz
        Whn
        bhn
        Variant (1,1) string = "pytorch"
    end

    methods
        function obj = GRUCell(inputSize, hiddenSize, opts)
            arguments
                inputSize (1,1) double {mustBePositive, mustBeInteger}
                hiddenSize (1,1) double {mustBePositive, mustBeInteger}
                opts.Variant (1,1) string {mustBeMember(opts.Variant, ["pytorch", "original"])} = "pytorch"
                opts.Seed = []
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["double", "single"])} = "double"
            end
            obj.InputSize = inputSize;
            obj.HiddenSize = hiddenSize;
            obj.Variant = opts.Variant;
            H = hiddenSize;
            s = obj.stream(opts.Seed);
            k = 1 / sqrt(H);
            obj.Wx = obj.uniformParam(3*H, inputSize, k, s, opts.DataType);
            obj.bx = obj.uniformParam(3*H, 1, k, s, opts.DataType);
            obj.Whrz = obj.uniformParam(2*H, H, k, s, opts.DataType);
            obj.bhrz = obj.uniformParam(2*H, 1, k, s, opts.DataType);
            obj.Whn = obj.uniformParam(H, H, k, s, opts.DataType);
            obj.bhn = obj.uniformParam(H, 1, k, s, opts.DataType);
        end

        function [h, state] = step(obj, x, state)
            H = obj.HiddenSize;
            hPrev = state.h;
            gx = obj.Wx * x + obj.bx;                       % (3H x N): r, z, n input parts
            grz = obj.Whrz * hPrev + obj.bhrz;              % (2H x N): r, z recurrent parts
            r = sigmoid(gx(1:H, :) + grz(1:H, :));
            z = sigmoid(gx(H+1:2*H, :) + grz(H+1:2*H, :));
            if obj.Variant == "pytorch"
                n = tanh(gx(2*H+1:3*H, :) + r .* (obj.Whn * hPrev + obj.bhn));
            else
                n = tanh(gx(2*H+1:3*H, :) + obj.Whn * (r .* hPrev) + obj.bhn);
            end
            h = (1 - z) .* n + z .* hPrev;
            state.h = h;
        end
    end
end
