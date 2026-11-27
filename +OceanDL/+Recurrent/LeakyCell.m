classdef LeakyCell < OceanDL.Recurrent.IRecurrentCell
    % LEAKYCELL  Leaky-integrator units with multiple time scales --
    %   Goodfellow Sec. 10.9.2 (Mozer 1992; Jaeger et al. 2007):
    %
    %     h_t = (1 - alpha) .* h_{t-1} + alpha .* tanh(Wx x_t + Wh h_{t-1} + b)
    %
    %   alpha in (0, 1] is a per-unit time constant. alpha = 1 is the Elman
    %   cell; alpha << 1 makes a unit an exponential running average that
    %   remembers for ~1/alpha steps, because dh_t/dh_{t-1} contains the term
    %   (1 - alpha) I, which is close to the identity -- a cheap, un-gated
    %   version of the LSTM's linear memory path.
    %
    %   Alphas are spread LOG-UNIFORMLY over [AlphaMin, 1] so the layer has
    %   units at many time scales at once. LearnTimeConstants = true makes
    %   them trainable through alpha = sigmoid(rho) (keeps alpha in (0,1)).

    properties (SetAccess = protected)
        Wx
        Wh
        b
        Rho                  % Parameter (learnable alphas) or [] (fixed)
        FixedAlpha           % (H x 1) array when not learnable
    end

    methods
        function obj = LeakyCell(inputSize, hiddenSize, opts)
            arguments
                inputSize (1,1) double {mustBePositive, mustBeInteger}
                hiddenSize (1,1) double {mustBePositive, mustBeInteger}
                opts.AlphaMin (1,1) double {mustBePositive, mustBeLessThanOrEqual(opts.AlphaMin, 1)} = 0.02
                opts.Alpha double = []                  % explicit (H x 1) or scalar alphas
                opts.LearnTimeConstants (1,1) logical = false
                opts.Seed = []
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["double", "single"])} = "double"
            end
            obj.InputSize = inputSize;
            obj.HiddenSize = hiddenSize;
            H = hiddenSize;
            s = obj.stream(opts.Seed);
            k = 1 / sqrt(H);
            obj.Wx = obj.uniformParam(H, inputSize, k, s, opts.DataType);
            obj.Wh = obj.uniformParam(H, H, k, s, opts.DataType);
            obj.b = obj.uniformParam(H, 1, k, s, opts.DataType);
            if isempty(opts.Alpha)
                alpha = exp(linspace(log(opts.AlphaMin), 0, H)).';   % log-uniform time scales
            else
                alpha = opts.Alpha(:) .* ones(H, 1);
            end
            alpha = cast(alpha, opts.DataType);
            if opts.LearnTimeConstants
                alpha = min(alpha, 1 - 1e-4);                        % keep logit finite
                obj.Rho = OceanDL.nn.Parameter(log(alpha ./ (1 - alpha)));
            else
                obj.FixedAlpha = alpha;
            end
        end

        function a = alpha(obj)
            % Current time constants (Tensor if learnable, array otherwise).
            if isempty(obj.Rho)
                a = obj.FixedAlpha;
            else
                a = sigmoid(obj.Rho);
            end
        end

        function [h, state] = step(obj, x, state)
            a = obj.alpha();
            candidate = tanh(obj.Wx * x + obj.Wh * state.h + obj.b);
            h = (1 - a) .* state.h + a .* candidate;
            state.h = h;
        end
    end
end
