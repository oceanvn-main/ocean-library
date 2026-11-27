function G = gen_tile(X, cols)
% GEN_TILE  Arrange images X (H x W x 1 x N, values in [0, 1]) into one grid
%   with `cols` columns and a 1-pixel separator; clipped to [0, 1].
X = min(1, max(0, double(gather(X))));
[H, W, ~, N] = size(X, 1:4);
rows = ceil(N / cols);
G = 0.25 * ones(rows * (H + 1) + 1, cols * (W + 1) + 1);
for k = 1:N
    r = floor((k - 1) / cols); c = mod(k - 1, cols);
    G(r * (H + 1) + 1 + (1:H), c * (W + 1) + 1 + (1:W)) = X(:, :, 1, k);
end
end
