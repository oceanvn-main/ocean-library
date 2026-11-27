classdef RoPE
    % ROPE  Rotary position embedding (Su et al. 2021) -- positions enter
    %   attention by ROTATING each query / key feature pair by a position-
    %   dependent angle, so  q(p1) . k(p2)  depends only on p1 - p2.
    %
    %   Convention ("half" / GPT-NeoX): feature i is paired with i + d/2,
    %     rot(x) = x .* cos(theta) + [-x(d/2+1:d); x(1:d/2)] .* sin(theta)
    %   with theta = [phi; phi] (d values) for the d/2 angles phi.
    %
    %   1-D: phi_j = p * base^(-j / (d/2)),  j = 0..d/2-1       (sequence position p)
    %   2-D axial (images; FLUX / Qwen-Image style): split the d/2 angles in
    %   two halves driven by the row r and the column c of the patch:
    %     phi = [r f; c f],  f_j = base^(-j / (d/4)),  j = 0..d/4-1   (d divisible by 4)
    %   Token order is column-major over the patch grid (row index fastest),
    %   matching PatchEmbedding.
    %
    %   r = RoPE.axial2d(Hp, Wp, dk)   -> struct(Cos, Sin), each (dk x T)
    %   Y = RoPE.apply(X, r)           X: (dk x T x h x N) split heads (Tensor or array)

    methods (Static)
        function r = sequence(T, d, base)
            if nargin < 3, base = 10000; end
            f = base .^ (-(0:d/2 - 1).' / (d / 2));
            phi = f * (0:T - 1);                                     % d/2 x T
            r = OceanDL.Transformer.RoPE.fromAngles(phi);
        end

        function r = axial2d(Hp, Wp, d, base)
            if nargin < 4, base = 10000; end
            if mod(d, 4)
                error('OceanDL:Transformer:RoPE:Dim', '2-D RoPE needs a head dimension divisible by 4, got %d.', d);
            end
            f = base .^ (-(0:d/4 - 1).' / (d / 4));
            [rr, cc] = ndgrid(0:Hp - 1, 0:Wp - 1);                  % column-major: row index fastest
            phi = [f * rr(:).'; f * cc(:).'];                       % d/2 x T
            r = OceanDL.Transformer.RoPE.fromAngles(phi);
        end

        function Y = apply(X, r)
            d = size(X, 1);
            T = size(X, 2);
            c = reshape(r.Cos(:, 1:T), d, T); s = reshape(r.Sin(:, 1:T), d, T);
            c = cast(c, 'like', OceanDL.Autograd.Tensor.raw(X)); s = cast(s, 'like', OceanDL.Autograd.Tensor.raw(X));
            half = d / 2;
            rotated = cat(1, -X(half + 1:d, :, :, :), X(1:half, :, :, :));
            Y = X .* c + rotated .* s;
        end

        function r = fromAngles(phi)
            theta = [phi; phi];
            r = struct('Cos', cos(theta), 'Sin', sin(theta));
        end
    end
end
