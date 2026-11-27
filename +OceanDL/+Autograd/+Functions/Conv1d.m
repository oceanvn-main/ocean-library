classdef Conv1d < OceanDL.Autograd.Function
    % CONV1D  1-D cross-correlation over time on CBT sequences (PyTorch
    %   torch.nn.functional.conv1d), with stride, zero padding, dilation and groups.
    %
    %   X  (Cin x N x T)            canonical "CBT" layout (OceanDL.Layout.DimFormat)
    %   Wt (Cout x Cin/G x K)       PyTorch weight order (out, in/groups, kernel)
    %   b  (Cout x 1, optional)
    %   -> Y (Cout x N x To),  To = floor((T + 2p - d(K-1) - 1)/s) + 1
    %
    % FORMULA -- a sum over the K kernel taps of time-shifted input copies:
    %   Y(:, n, t) = b + sum_{k=1}^{K} W_k * Xp(:, n, (t-1)s + (k-1)d + 1)
    %   with Xp = X zero-padded by p steps on both sides and W_k = Wt(:, :, k).
    %   For tap k the shifted window S_k = Xp(:, :, k-th strided slice) is a
    %   contiguous (Cin x N*To) reshape of the CBT array, so the whole layer is
    %   K matrix products (no per-time-step loop, no im2col buffer).
    %   GROUPS G: channels split into G blocks of Cin/G inputs and Cout/G outputs;
    %   block g only sees its own inputs:  Y_g = sum_k W_{k,g} * S_{k,g}.
    %     G = 1          one (Cout x Cin) * (Cin x N*To) product per tap
    %     G = Cin        depthwise: elementwise products (Cin/G = 1)
    %     otherwise      pagemtimes over the G blocks
    %
    % VJP (each tap differentiated separately, G = dL/dY as Cout x N*To):
    %   dL/dW_k = G * S_k.'          dL/dS_k = W_k.' * G   scattered (+=) back into
    %   the k-th strided time slice of dL/dXp, then the padding is cropped;
    %   dL/db   = sum over (n, t) of G.

    properties (SetAccess = private)
        Stride   (1,1) double = 1
        Padding  (1,1) double = 0
        Dilation (1,1) double = 1
        Groups   (1,1) double = 1
    end

    methods
        function obj = Conv1d(opts)
            arguments
                opts.Stride   (1,1) double {mustBePositive, mustBeInteger} = 1
                opts.Padding  (1,1) double {mustBeNonnegative, mustBeInteger} = 0
                opts.Dilation (1,1) double {mustBePositive, mustBeInteger} = 1
                opts.Groups   (1,1) double {mustBePositive, mustBeInteger} = 1
            end
            obj.Stride = opts.Stride;
            obj.Padding = opts.Padding;
            obj.Dilation = opts.Dilation;
            obj.Groups = opts.Groups;
        end

        function Y = forward(obj, X, Wt, b)
            [Cin, N, T] = size(X, 1:3);
            [Cout, Cig, K] = size(Wt, 1:3);
            G = obj.Groups;
            if Cin ~= Cig * G || mod(Cout, G) ~= 0
                error('OceanDL:Autograd:Conv1d:ChannelMismatch', ...
                    'Input has %d channels; weight %d x %d x %d with Groups = %d expects %d (and Cout divisible by Groups).', ...
                    Cin, Cout, Cig, K, G, Cig * G);
            end
            p = obj.Padding; s = obj.Stride; d = obj.Dilation;
            To = OceanDL.Autograd.Functions.Conv1d.outLength(T, K, s, p, d);
            if To < 1
                error('OceanDL:Autograd:Conv1d:InputTooShort', ...
                    'T = %d (padding %d) is shorter than the dilated kernel %d.', T, p, d * (K - 1) + 1);
            end
            Xp = zeros(Cin, N, T + 2*p, 'like', X);
            Xp(:, :, p+1:p+T) = X;
            M = N * To;
            Ym = zeros(Cout, M, 'like', X);
            for k = 1:K
                S = reshape(Xp(:, :, obj.taps(k, To)), Cin, M);
                Ym = Ym + obj.tapForward(Wt(:, :, k), S, Cout, Cig, M);
            end
            Y = reshape(Ym, Cout, N, To);
            hasBias = nargin > 3 && ~isempty(b);
            if hasBias
                Y = Y + reshape(b, Cout, 1);
            end
            obj.Saved = struct('Xp', Xp, 'Wt', Wt, 'In', [Cin N T], 'To', To, 'HasBias', hasBias);
        end

        function grads = backward(obj, g)
            s = obj.Saved;
            Cin = s.In(1); N = s.In(2); T = s.In(3); To = s.To;
            [Cout, Cig, K] = size(s.Wt, 1:3);
            p = obj.Padding;
            M = N * To;
            Gm = reshape(g, Cout, M);
            dW = zeros(size(s.Wt, 1:3), 'like', g);
            dXp = zeros(size(s.Xp, 1:3), 'like', g);
            for k = 1:K
                idx = obj.taps(k, To);
                S = reshape(s.Xp(:, :, idx), Cin, M);
                [dWk, dS] = obj.tapBackward(s.Wt(:, :, k), S, Gm, Cout, Cig, M);
                dW(:, :, k) = dWk;
                dXp(:, :, idx) = dXp(:, :, idx) + reshape(dS, Cin, N, To);   % idx has distinct entries
            end
            grads = {dXp(:, :, p+1:p+T), dW};
            if s.HasBias
                grads{3} = sum(Gm, 2);
            end
        end
    end

    methods (Access = private)
        function idx = taps(obj, k, To)
            % Time indices (in the padded input) read by tap k for outputs 1..To.
            first = (k - 1) * obj.Dilation + 1;
            idx = first : obj.Stride : first + obj.Stride * (To - 1);
        end

        function Yk = tapForward(obj, Wk, S, Cout, Cig, M)
            G = obj.Groups;
            if G == 1
                Yk = Wk * S;
            elseif Cig == 1                                          % depthwise (multiplier m)
                m = Cout / G;
                Yk = reshape(reshape(Wk, m, G) .* reshape(S, 1, G, M), Cout, M);
            else
                Og = Cout / G;
                Wp = permute(reshape(Wk, Og, G, Cig), [1 3 2]);      % Og x Cig x G
                Sp = permute(reshape(S, Cig, G, M), [1 3 2]);         % Cig x M x G
                Yk = reshape(permute(pagemtimes(Wp, Sp), [1 3 2]), Cout, M);
            end
        end

        function [dWk, dS] = tapBackward(obj, Wk, S, Gm, Cout, Cig, M)
            G = obj.Groups;
            if G == 1
                dWk = Gm * S.';
                dS = Wk.' * Gm;
            elseif Cig == 1
                m = Cout / G;
                G3 = reshape(Gm, m, G, M);
                dWk = reshape(sum(G3 .* reshape(S, 1, G, M), 3), Cout, 1);
                dS = reshape(sum(G3 .* reshape(Wk, m, G), 1), G, M);
            else
                Og = Cout / G;
                Wp = permute(reshape(Wk, Og, G, Cig), [1 3 2]);      % Og x Cig x G
                Sp = permute(reshape(S, Cig, G, M), [1 3 2]);         % Cig x M x G
                Gp = permute(reshape(Gm, Og, G, M), [1 3 2]);         % Og x M x G
                dWk = reshape(permute(pagemtimes(Gp, 'none', Sp, 'transpose'), [1 3 2]), Cout, Cig);
                dS = reshape(permute(pagemtimes(Wp, 'transpose', Gp, 'none'), [1 3 2]), Cig * G, M);
            end
        end
    end

    methods (Static)
        function To = outLength(T, K, s, p, d)
            % PyTorch Conv1d length formula, clamped at 0 (vectorised over T).
            To = max(0, floor((T + 2*p - d*(K - 1) - 1) / s) + 1);
        end
    end
end
