classdef WindowOps
    % WINDOWOPS  The geometry of Swin Transformer's windowed attention
    %   (Liu et al. 2021, Sec. 3.2), on feature maps X4 = (D x N x Hp x Wp)
    %   whose last two axes are the patch grid (row index fastest).
    %
    %   partition   X4 -> windows (D x (N*nW) x M^2): every non-overlapping
    %               M x M window becomes one short token sequence ("CBT"),
    %               all windows of all images batched together, so attention
    %               costs nW * M^4 instead of (Hp*Wp)^2 -- LINEAR in image area.
    %               batch index b = n + N*(w-1), window w = hw + Hw*(ww-1),
    %               local token i = r + M*(c-1) (column-major, like PatchEmbedding).
    %   reverse     the exact inverse.
    %   shift       cyclic roll of the grid by -s rows/cols (s = M/2), so the next
    %               layer's windows straddle the previous layer's window borders
    %               (cross-window connections) while the window COUNT stays the same.
    %   shiftMask   after rolling, a window can contain patches that were far apart
    %               in the image (wrapped around). They must not attend to each
    %               other: the grid is cut into 3 x 3 regions
    %               [1 : H-M], [H-M+1 : H-s], [H-s+1 : H]  (same for columns);
    %               token j is visible to token i only if both lie in the same region.
    %               Returns (M^2 x M^2 x 1 x N*nW) logical, true = may attend.
    %   relativeIndex  (M^2 x M^2) index into a (2M-1)^2-entry table: the bias for
    %               tokens i, j depends only on their offset (dr, dc) in [-(M-1), M-1]^2.
    %   All operations are reshape / permute / indexing, so they work on Tensors
    %   and are differentiable.

    methods (Static)
        function W = partition(X4, M)
            [D, N, Hp, Wp] = size(X4, 1:4);
            OceanDL.Vision.WindowOps.checkGrid(Hp, Wp, M);
            Hw = Hp / M; Ww = Wp / M;
            W = reshape(X4, D, N, M, Hw, M, Ww);
            W = permute(W, [1 2 4 6 3 5]);                     % D, N, Hw, Ww, r, c
            W = reshape(W, D, N * Hw * Ww, M * M);
        end

        function X4 = reverse(W, M, N, Hp, Wp)
            D = size(W, 1);
            Hw = Hp / M; Ww = Wp / M;
            X4 = reshape(W, D, N, Hw, Ww, M, M);
            X4 = ipermute(X4, [1 2 4 6 3 5]);                  % back to D, N, r, Hw, c, Ww
            X4 = reshape(X4, D, N, Hp, Wp);
        end

        function X4 = shift(X4, s, direction)
            % direction -1: roll by -s (before attention); +1: roll back.
            if s == 0, return; end
            [~, ~, Hp, Wp] = size(X4, 1:4);
            rows = circshift(1:Hp, direction * s);
            cols = circshift(1:Wp, direction * s);
            X4 = X4(:, :, rows, cols);
        end

        function mask = shiftMask(Hp, Wp, M, s, N)
            region = zeros(Hp, Wp);
            hs = {1:Hp-M, Hp-M+1:Hp-s, Hp-s+1:Hp};
            ws = {1:Wp-M, Wp-M+1:Wp-s, Wp-s+1:Wp};
            id = 0;
            for a = 1:3
                for b = 1:3
                    region(hs{a}, ws{b}) = id;
                    id = id + 1;
                end
            end
            win = OceanDL.Vision.WindowOps.partition(reshape(region, 1, 1, Hp, Wp), M);   % 1 x nW x M^2
            nW = size(win, 2);
            win = reshape(win, nW, M * M);
            allowed = permute(win, [2 3 1]) == permute(win, [3 2 1]);                   % M^2 x M^2 x nW
            mask = reshape(repmat(reshape(allowed, M*M, M*M, 1, 1, nW), [1 1 1 N 1]), M*M, M*M, 1, N * nW);
        end

        function idx = relativeIndex(M)
            [r, c] = ndgrid(1:M, 1:M);
            r = r(:); c = c(:);                                % token i = r + M*(c-1)
            dr = r - r.' + (M - 1);                            % 0 .. 2M-2
            dc = c - c.' + (M - 1);
            idx = dr + (2*M - 1) * dc + 1;                     % 1 .. (2M-1)^2
        end
    end

    methods (Static, Access = private)
        function checkGrid(Hp, Wp, M)
            if mod(Hp, M) || mod(Wp, M)
                error('OceanDL:Vision:WindowOps:Grid', 'Patch grid %dx%d is not divisible by window size %d.', Hp, Wp, M);
            end
        end
    end
end
