classdef Conv2d < OceanDL.Autograd.Function
    % CONV2D  2-D cross-correlation, shift-and-multiply form.
    %   X (H x W x Cin x N), Wt (KH x KW x Cin x Cout), b (Cout x 1, optional)
    %   -> Y (Ho x Wo x Cout x N), same layout and semantics as the legacy
    %   OceanDL.Conv.Conv2D (cross-correlation, zero padding).
    %
    % FORMULA -- a convolution is a SUM OVER KERNEL OFFSETS of shifted
    %   copies of the input, each multiplied by one KH x KW slice of the kernel:
    %     Y = sum_{kh,kw}  S_{kh,kw} * W_{kh,kw}   (+ b)
    %   where S_{kh,kw} (Ho*Wo*N x Cin) is the input window shifted by
    %   (kh, kw) (with stride), and W_{kh,kw} = W(kh,kw,:,:) (Cin x Cout).
    %   KH*KW matrix products on CONTIGUOUS reshapes -- no 6-D im2col buffer
    %   (measured 4x faster than the legacy im2col layer in this repo, and it
    %   keeps single precision).
    %
    % VJP (the same sum, differentiated term by term):
    %   dL/dW_{kh,kw} = S_{kh,kw}.' * G           (G = dL/dY as Ho*Wo*N x Cout)
    %   dL/dS_{kh,kw} = G * W_{kh,kw}.'          scattered back (+=) into the
    %                                            shifted window of dL/dX
    %   dL/db         = sum over positions of G

    properties (SetAccess = private)
        Stride  (1,2) double = [1 1]
        Padding (1,2) double = [0 0]
    end

    methods
        function obj = Conv2d(opts)
            arguments
                opts.Stride  (1,2) double {mustBePositive, mustBeInteger} = [1 1]
                opts.Padding (1,2) double {mustBeNonnegative, mustBeInteger} = [0 0]
            end
            obj.Stride = opts.Stride;
            obj.Padding = opts.Padding;
        end

        function Y = forward(obj, X, Wt, b)
            [H, W, Cin, N] = size(X, 1:4);
            [KH, KW, ~, Cout] = size(Wt, 1:4);
            ph = obj.Padding(1); pw = obj.Padding(2);
            sh = obj.Stride(1);  sw = obj.Stride(2);
            Ho = floor((H + 2*ph - KH) / sh) + 1;
            Wo = floor((W + 2*pw - KW) / sw) + 1;

            Xp = zeros(H + 2*ph, W + 2*pw, Cin, N, 'like', X);
            Xp(ph+1:ph+H, pw+1:pw+W, :, :) = X;
            Xp = permute(Xp, [1 2 4 3]);          % (Hp, Wp, N, Cin): windows reshape to (Ho*Wo*N x Cin)

            Ym = zeros(Ho*Wo*N, Cout, 'like', X);
            for kh = 1:KH
                rows = kh : sh : kh + sh*(Ho-1);
                for kw = 1:KW
                    cols = kw : sw : kw + sw*(Wo-1);
                    S = reshape(Xp(rows, cols, :, :), Ho*Wo*N, Cin);
                    Ym = Ym + S * reshape(Wt(kh, kw, :, :), Cin, Cout);
                end
            end
            Y = permute(reshape(Ym, Ho, Wo, N, Cout), [1 2 4 3]);
            if nargin > 3 && ~isempty(b)
                Y = Y + reshape(b, 1, 1, Cout);
            end
            obj.Saved = struct('Xp', Xp, 'Wt', Wt, 'In', [H W Cin N], 'Out', [Ho Wo], ...
                'HasBias', nargin > 3 && ~isempty(b));
        end

        function grads = backward(obj, g)
            s = obj.Saved;
            H = s.In(1); W = s.In(2); Cin = s.In(3); N = s.In(4);
            Ho = s.Out(1); Wo = s.Out(2);
            [KH, KW, ~, Cout] = size(s.Wt, 1:4);
            ph = obj.Padding(1); pw = obj.Padding(2);
            sh = obj.Stride(1);  sw = obj.Stride(2);

            G = reshape(permute(g, [1 2 4 3]), Ho*Wo*N, Cout);
            dW = zeros(size(s.Wt), 'like', g);
            dXp = zeros(size(s.Xp), 'like', g);
            for kh = 1:KH
                rows = kh : sh : kh + sh*(Ho-1);
                for kw = 1:KW
                    cols = kw : sw : kw + sw*(Wo-1);
                    S = reshape(s.Xp(rows, cols, :, :), Ho*Wo*N, Cin);
                    Wk = reshape(s.Wt(kh, kw, :, :), Cin, Cout);
                    dW(kh, kw, :, :) = reshape(S.' * G, 1, 1, Cin, Cout);
                    dXp(rows, cols, :, :) = dXp(rows, cols, :, :) + reshape(G * Wk.', Ho, Wo, N, Cin);
                end
            end
            dX = permute(dXp(ph+1:ph+H, pw+1:pw+W, :, :), [1 2 4 3]);
            grads = {dX, dW};
            if s.HasBias
                grads{3} = sum(G, 1).';
            end
        end
    end
end
