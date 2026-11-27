classdef PositionalEncoding
    % POSITIONALENCODING  Self-attention is permutation-invariant: shuffling
    %   the tokens shuffles the outputs identically. Position must therefore
    %   be ADDED to the token features. Fixed encodings (no parameters):
    %
    %   sincos1d(T, D)   Vaswani et al. 2017, Sec. 3.5, position t = 0..T-1:
    %     PE(2i,   t) = sin(t / 10000^(2i/D))
    %     PE(2i+1, t) = cos(t / 10000^(2i/D))
    %     wavelengths form a geometric progression, so relative offsets are
    %     linear functions of the encoding.
    %
    %   sincos2d(Hp, Wp, D)   2-D version for patch grids (MAE, He et al. 2022):
    %     first D/2 features encode the patch ROW, last D/2 the patch COLUMN,
    %     each with the 1-D formula. Token order is column-major over the grid,
    %     t = r + Hp*(c-1) -- the order produced by Vision.PatchEmbedding.
    %
    %   Both return (D x 1 x T), ready to broadcast-add to "CBT" tokens.
    %   The learned alternative is Transformer.LearnedPositionalEmbedding.

    methods (Static)
        function P = sincos1d(T, D, base)
            arguments
                T (1,1) double {mustBePositive, mustBeInteger}
                D (1,1) double {mustBePositive, mustBeInteger}
                base (1,1) double = 10000
            end
            if mod(D, 2)
                error('OceanDL:Transformer:PositionalEncoding:OddDim', 'D must be even, got %d.', D);
            end
            t = 0:T-1;                                        % 1 x T
            i = (0:D/2-1).';                                  % frequency index
            angle = t ./ base.^(2 * i / D);                   % (D/2) x T
            P = zeros(D, T);
            P(1:2:end, :) = sin(angle);
            P(2:2:end, :) = cos(angle);
            P = reshape(P, D, 1, T);
        end

        function P = sincos2d(Hp, Wp, D)
            arguments
                Hp (1,1) double {mustBePositive, mustBeInteger}
                Wp (1,1) double {mustBePositive, mustBeInteger}
                D (1,1) double {mustBePositive, mustBeInteger}
            end
            if mod(D, 4)
                error('OceanDL:Transformer:PositionalEncoding:Dim4', 'D must be divisible by 4, got %d.', D);
            end
            rowPE = reshape(OceanDL.Transformer.PositionalEncoding.sincos1d(Hp, D/2), D/2, Hp);
            colPE = reshape(OceanDL.Transformer.PositionalEncoding.sincos1d(Wp, D/2), D/2, Wp);
            [r, c] = ndgrid(1:Hp, 1:Wp);                      % column-major token order
            P = reshape([rowPE(:, r(:)); colPE(:, c(:))], D, 1, Hp * Wp);
        end
    end
end
