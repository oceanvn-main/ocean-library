classdef ImageSequence
    % IMAGESEQUENCE  Read an image as a SEQUENCE, so recurrent layers can
    %   process it: canonical "SSCB" (H x W x C x N) -> canonical "CBT".
    %
    %   Mode "rows"     t = image row      D = W*C,  T = H   (row-by-row scan;
    %                                                        "sequential MNIST", Le et al. 2015)
    %   Mode "columns"  t = image column   D = H*C,  T = W   (left-to-right reading --
    %                                                        CRNN text recognition,
    %                                                        Shi et al. 2017, on CNN feature maps)
    %   Mode "pixels"   t = one pixel      D = C,    T = H*W (pixel-by-pixel; the hardest,
    %                                                        longest-dependency version)
    %
    %   Implemented with permute + reshape only, so it works on arrays AND on
    %   Tensors (the gradient flows back to the image / feature map).
    %   Pixel order in "pixels" mode is column-major (down each column, then
    %   the next column), MATLAB's natural order.
    %
    %   The feature vector of a "rows" step is the row's pixels (W) for each
    %   channel: element (w, c) of row h lands at d = w + W*(c-1).

    methods (Static)
        function S = toSequence(X, mode)
            arguments
                X
                mode (1,1) string {mustBeMember(mode, ["rows", "columns", "pixels"])} = "rows"
            end
            [H, W, C, N] = size(X, 1:4);
            switch mode
                case "rows"
                    S = reshape(permute(X, [2 3 4 1]), W*C, N, H);     % (W,C,N,H) -> (W*C) x N x H
                case "columns"
                    S = reshape(permute(X, [1 3 4 2]), H*C, N, W);     % (H,C,N,W) -> (H*C) x N x W
                case "pixels"
                    S = reshape(permute(X, [3 4 1 2]), C, N, H*W);     % (C,N,H,W) -> C x N x (H*W)
            end
        end
    end
end
