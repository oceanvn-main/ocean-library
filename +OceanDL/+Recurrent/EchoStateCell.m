classdef EchoStateCell < OceanDL.Recurrent.IRecurrentCell
    % ECHOSTATECELL  Reservoir computing (Jaeger 2001; Maass et al. 2002) --
    %   Goodfellow Sec. 10.8.
    %
    %     h_t = (1 - a) h_{t-1} + a tanh(Win x_t + W h_{t-1} + b)
    %
    %   Win, W and b are RANDOM and FIXED: they are plain arrays, not
    %   Parameters, so parameters() is empty and no gradient is ever computed
    %   through time. Only a readout y = V h_t + c is trained -- by ordinary
    %   gradient descent on a Linear layer, or in closed form with
    %   fitRidgeReadout (the classic ESN recipe). Training is therefore a
    %   linear least-squares problem: no BPTT, no vanishing gradients.
    %
    %   The reservoir must have the ECHO STATE PROPERTY: the influence of the
    %   initial state fades, so h_t depends on the recent inputs only. The
    %   standard control is the SPECTRAL RADIUS rho(W) = max |eig(W)|:
    %   W is rescaled so rho(W) = SpectralRadius (typically ~0.9-1.2;
    %   Goodfellow notes values near 1 retain memory longest, > 1 can be
    %   chaotic). Sparse random W (Density) is the usual choice.

    properties (SetAccess = protected)
        Win                     % (H x D) fixed
        W                       % (H x H) fixed, rho(W) = SpectralRadius
        bias                    % (H x 1) fixed
        SpectralRadius (1,1) double
        LeakRate (1,1) double
    end

    methods
        function obj = EchoStateCell(inputSize, hiddenSize, opts)
            arguments
                inputSize (1,1) double {mustBePositive, mustBeInteger}
                hiddenSize (1,1) double {mustBePositive, mustBeInteger}
                opts.SpectralRadius (1,1) double {mustBePositive} = 0.95
                opts.Density (1,1) double {mustBePositive, mustBeLessThanOrEqual(opts.Density, 1)} = 0.1
                opts.InputScale (1,1) double {mustBePositive} = 1
                opts.LeakRate (1,1) double {mustBePositive, mustBeLessThanOrEqual(opts.LeakRate, 1)} = 1
                opts.Seed = []
                opts.DataType (1,1) string {mustBeMember(opts.DataType, ["double", "single"])} = "double"
            end
            obj.InputSize = inputSize;
            obj.HiddenSize = hiddenSize;
            obj.SpectralRadius = opts.SpectralRadius;
            obj.LeakRate = opts.LeakRate;
            s = obj.stream(opts.Seed);
            if isempty(s)
                s = RandStream.getGlobalStream();
            end
            H = hiddenSize;
            obj.Win = cast(opts.InputScale * (2 * rand(s, H, inputSize) - 1), opts.DataType);
            obj.bias = cast(0.1 * (2 * rand(s, H, 1) - 1), opts.DataType);
            Wr = (2 * rand(s, H, H) - 1) .* (rand(s, H, H) < opts.Density);
            rho = OceanDL.Recurrent.EchoStateCell.spectralRadius(Wr);
            if rho == 0
                error('OceanDL:Recurrent:EchoStateCell:ZeroReservoir', ...
                    'Random reservoir has spectral radius 0; increase Density or HiddenSize.');
            end
            obj.W = cast(Wr * (opts.SpectralRadius / rho), opts.DataType);
        end

        function [h, state] = step(obj, x, state)
            a = obj.LeakRate;
            candidate = tanh(obj.Win * x + obj.W * state.h + obj.bias);
            h = (1 - a) .* state.h + a .* candidate;
            state.h = h;
        end
    end

    methods (Static)
        function rho = spectralRadius(W)
            % Largest eigenvalue magnitude (dense eig: reservoirs are small).
            rho = max(abs(eig(double(full(W)))));
        end

        function [V, c] = fitRidgeReadout(Hstates, Y, lambda)
            % FITRIDGEREADOUT  Closed-form readout y = V h + c minimising
            %   ||Y - V H - c||^2 + lambda ||V||^2   (bias not penalised):
            %     [V c] = Y Ha' (Ha Ha' + lambda P)^-1,  Ha = [H; 1], P = diag(1,...,1,0)
            %   Hstates (H x M) collected reservoir states, Y (K x M) targets.
            arguments
                Hstates (:,:) double
                Y (:,:) double
                lambda (1,1) double {mustBeNonnegative} = 1e-6
            end
            Ha = [Hstates; ones(1, size(Hstates, 2))];
            P = eye(size(Ha, 1));
            P(end, end) = 0;
            Vc = (Y * Ha.') / (Ha * Ha.' + lambda * P);
            V = Vc(:, 1:end-1);
            c = Vc(:, end);
        end
    end
end
