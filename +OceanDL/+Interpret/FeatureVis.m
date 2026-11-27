classdef FeatureVis
    % FEATUREVIS  Activation maximisation: the input pattern a unit or channel
    %   responds to most (Szeliski 2nd ed. Sec. 5.4 (pp. 309-311, Figs. 5.47-5.50), Fig. 5.49 "maximum
    %   stimulus"; Olah, Mordvintsev & Schubert 2017).
    %
    %   maximise  a(x),   x = sigmoid( B (theta .* s) )   (pixels in (0, 1))
    %   by gradient ascent (Adam) on theta, with Olah et al.'s regularisers as
    %   implemented in the Lucid / Lucent libraries (tools, not in the texts):
    %     * DECORRELATED PARAMETERISATION: theta are 2-D DCT coefficients; B is the
    %       inverse DCT and s = 1 / frequency scales them -- low frequencies are
    %       cheap, high-frequency noise expensive (Lucent uses a Fourier basis).
    %     * TRANSFORMATION ROBUSTNESS: random circular shifts ('Jitter' pixels)
    %       every step, so the pattern must work at slightly different positions.
    %     * a small L2 penalty on theta ('Decay').
    %   'Decorrelate' false uses plain pixels (s = 1): no frequency prior.
    %   'Init' x0 (H x W x C in [0, 1]): START from a given image instead of grey -- e.g. the
    %   unit's best real example -- so the result shows how the unit EXAGGERATES real data
    %   ("feature accentuation", Hamblin et al. 2024; not in the texts). theta is set so that
    %   sigmoid(B (theta .* s)) = x0 clipped to [0.02, 0.98]; the L2 penalty then acts on the
    %   wave coefficients theta .* s (penalising theta itself would pull the start image apart).
    %   Objective 'neuron' = one unit (default: the centre of the layer),
    %   'channel' = the mean of the whole feature map.
    %
    %   [x, info] = FeatureVis.maximize(model, layer, channel, 'InputSize', [16 16 1])
    %       x (H x W x C) in (0, 1); info.History objective per step, info.Activation

    methods (Static)
        function [x, info] = maximize(model, layer, channel, opts)
            arguments
                model
                layer
                channel (1,1) double
                opts.InputSize (1,3) double
                opts.Objective (1,1) string {mustBeMember(opts.Objective, ["neuron", "channel"])} = "neuron"
                opts.Row double = []
                opts.Col double = []
                opts.Steps (1,1) double = 200
                opts.LR (1,1) double = 0.05
                opts.Jitter (1,1) double = 1
                opts.Decay (1,1) double = 1e-3
                opts.Seed (1,1) double = 0
                opts.Decorrelate (1,1) logical = true
                opts.Init = []
            end
            layer = char(layer);
            H = opts.InputSize(1); W = opts.InputSize(2); Cin = opts.InputSize(3);
            like = OceanDL.Interpret.ITraceable.likeOf(model);
            [B, s] = OceanDL.Interpret.FeatureVis.basis(H, W);
            if ~opts.Decorrelate, s = ones(size(s)); end
            B = cast(B, 'like', like); s = cast(s, 'like', like);
            rs = RandStream('mt19937ar', 'Seed', opts.Seed);
            theta = cast(0.01 * randn(rs, H * W, Cin), 'like', like);
            fromInit = ~isempty(opts.Init);
            if fromInit                                                  % start from the given image
                q = min(max(double(reshape(opts.Init, H * W, Cin)), 0.02), 0.98);
                theta = cast((double(B).' * log(q ./ (1 - q))) ./ double(s), 'like', like);   % B orthonormal: B^-1 = B.'
            end
            m = zeros(size(theta), 'like', like); v = m; b1 = 0.9; b2 = 0.999;
            parts = OceanDL.Interpret.Probe.partsFor(layer);
            mask = [];
            info.History = zeros(1, opts.Steps);
            x0 = OceanDL.Autograd.Tensor.raw(OceanDL.Interpret.FeatureVis.render(theta, B, s, H, W, Cin));
            info.Start = reshape(x0, H, W, Cin);
            for t = 1:opts.Steps
                T = OceanDL.Autograd.Tensor(theta, 'RequiresGrad', true);
                img = OceanDL.Interpret.FeatureVis.render(T, B, s, H, W, Cin);
                dy = randi(rs, [-opts.Jitter opts.Jitter]); dx = randi(rs, [-opts.Jitter opts.Jitter]);
                X = img(circshift(1:H, dy), circshift(1:W, dx), :, :);          % transformation robustness
                acts = model.trace(X, 'Parts', parts);
                a = acts(layer);
                if isempty(mask), mask = OceanDL.Interpret.FeatureVis.target(a, channel, opts); end
                objective = sum(a .* mask, 'all');
                if fromInit, penalty = sum((T .* s) .^ 2, 'all'); else, penalty = sum(T .^ 2, 'all'); end
                loss = -objective + opts.Decay * penalty;
                loss.backward();
                g = double(gather(T.Grad));
                OceanDL.Interpret.FeatureVis.clearParamGrads(model);
                m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g .^ 2;           % Adam (ascent on objective)
                theta = theta - opts.LR * (m / (1 - b1 ^ t)) ./ (sqrt(v / (1 - b2 ^ t)) + 1e-8);
                info.History(t) = double(OceanDL.Autograd.Tensor.raw(objective));
            end
            x = OceanDL.Autograd.Tensor.raw(OceanDL.Interpret.FeatureVis.render(theta, B, s, H, W, Cin));
            x = reshape(x, H, W, Cin);
            aFinal = OceanDL.Interpret.Probe.activation(model, reshape(x, H, W, Cin, 1), layer);
            if isempty(mask), mask = OceanDL.Interpret.FeatureVis.target(aFinal, channel, opts); end   % Steps = 0
            info.Activation = double(sum(aFinal .* OceanDL.Autograd.Tensor.raw(mask), 'all'));
            aStart = OceanDL.Interpret.Probe.activation(model, reshape(info.Start, H, W, Cin, 1), layer);
            info.StartActivation = double(sum(aStart .* OceanDL.Autograd.Tensor.raw(mask), 'all'));
        end

        function [B, s] = basis(H, W)
            % Inverse 2-D DCT (orthonormal) as a matrix on vec(image), and 1/frequency scales.
            B = kron(OceanDL.Interpret.FeatureVis.dct(W).', OceanDL.Interpret.FeatureVis.dct(H).');
            [fy, fx] = ndgrid(0:H - 1, 0:W - 1);
            s = 1 ./ max(1, sqrt((fy / H) .^ 2 + (fx / W) .^ 2) * max(H, W));
            s = s(:);
        end

        function D = dct(n)
            % Orthonormal DCT-II matrix: D * x = DCT(x); inverse = D.'.
            [k, j] = ndgrid(0:n - 1, 0:n - 1);
            D = sqrt(2 / n) * cos(pi * (2 * j + 1) .* k / (2 * n));
            D(1, :) = D(1, :) / sqrt(2);
        end
    end

    methods (Static, Access = private)
        function img = render(theta, B, s, H, W, Cin)
            pre = B * (theta .* s);                                              % (H W) x Cin
            img = reshape(sigmoid(OceanDL.Autograd.Tensor.wrap(pre)), H, W, Cin, 1);
        end

        function mask = target(a, channel, opts)
            r = OceanDL.Autograd.Tensor.raw(a);
            mask = zeros(size(r), 'like', r);
            if opts.Objective == "channel"
                mask(:, :, channel, 1) = 1 / (size(r, 1) * size(r, 2));
            else
                row = opts.Row; col = opts.Col;
                if isempty(row), row = ceil(size(r, 1) / 2); end
                if isempty(col), col = ceil(size(r, 2) / 2); end
                mask(row, col, channel, 1) = 1;
            end
        end

        function clearParamGrads(model)
            if isa(model, 'OceanDL.nn.Module')
                P = model.parameters();
                for k = 1:numel(P), P{k}.zeroGrad(); end
            end
        end
    end
end
