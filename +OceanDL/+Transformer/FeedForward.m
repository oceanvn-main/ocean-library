classdef FeedForward < OceanDL.nn.Module
    % FEEDFORWARD  Position-wise MLP of a Transformer block, applied to every
    %   token independently (Vaswani et al. 2017, Eq. 2):
    %     y = W2 act(W1 x + b1) + b2,   W1: (Hidden x D), W2: (D x Hidden)
    %   Hidden is typically 4 D; act = GELU (BERT/ViT) or ReLU (original), or
    %   "swiglu": y = W2 (SiLU(W1g x) .* (W1v x)) -- Fc1 then has 2 Hidden rows.

    properties (SetAccess = protected)
        Fc1
        Act
        Drop
        Fc2
        Gated (1,1) logical = false
    end

    methods
        function obj = FeedForward(dim, hidden, opts)
            arguments
                dim (1,1) double {mustBePositive, mustBeInteger}
                hidden (1,1) double {mustBePositive, mustBeInteger}
                opts.Activation (1,1) string {mustBeMember(opts.Activation, ["gelu", "relu", "swiglu"])} = "gelu"
                opts.Dropout (1,1) double = 0
                opts.Seed = []
                opts.DataType (1,1) string = "double"
            end
            s1 = []; s2 = [];
            if ~isempty(opts.Seed), s1 = opts.Seed + 1; s2 = opts.Seed + 2; end
            gated = opts.Activation == "swiglu";
            obj.Fc1 = OceanDL.nn.Linear(dim, hidden * (1 + gated), 'Seed', s1, 'DataType', opts.DataType);
            if gated
                obj.Act = OceanDL.nn.Activation("silu"); obj.Gated = true;
            else
                obj.Act = OceanDL.nn.Activation(opts.Activation);
            end
            obj.Drop = OceanDL.nn.Dropout(opts.Dropout);
            obj.Fc2 = OceanDL.nn.Linear(hidden, dim, 'Seed', s2, 'DataType', opts.DataType);
        end

        function y = forward(obj, x)
            h = obj.Fc1.forward(x);
            if obj.Gated
                % SwiGLU (Shazeer 2020; LLaMA, SD3.5 / FLUX-era DiTs): Fc1 produces
                % [gate; value] rows, h = SiLU(gate) .* value
                H = size(h, 1) / 2;
                sz = size(h); sz(1) = H;
                g = reshape(h(1:H, :), sz); v = reshape(h(H+1:end, :), sz);
                y = obj.Fc2.forward(obj.Drop.forward(obj.Act.forward(g) .* v));
                return;
            end
            y = obj.Fc2.forward(obj.Drop.forward(obj.Act.forward(h)));
        end
    end
end
