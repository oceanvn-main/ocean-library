classdef ResidualBlock
    % RESIDUALBLOCK  Utility class for residual (skip) connection arithmetic.
    %
    %   Provides static helpers that VectorizedEngine calls when BlockType='residual'.
    %   Keeps the skip logic out of the engine and makes it independently testable.
    %
    %   TWO CASES:
    %     Same dimensions (d_in == d_out):
    %       A_out = A_next + A_skip        (identity shortcut)
    %
    %     Different dimensions (d_in ~= d_out):
    %       A_out = A_next + W_skip*A_skip (projection shortcut, He et al. 2016, Option B)
    %
    %   CONVENTION (column-major, matches VectorizedEngine):
    %     A_next — (d_out x N) output of the main branch for this block
    %     A_skip — (d_in  x N) residual from the START of the block
    %     W_skip — (d_out x d_in) projection matrix (used iff d_in ~= d_out)
    %
    %   Blocks in NNTopologySpec:
    %     A 2-layer residual block spans layers [L, L+1].
    %     ResidualStart(L) is true when L starts a new 2-layer block.
    %     Blocks only form between consecutive hidden layers of EQUAL width
    %     (plain identity shortcut). Cross-width pairs use a projection shortcut.
    %
    %   Ref: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual
    %        learning for image recognition. CVPR 2016. arXiv:1512.03385
    %
    %   See also: OceanDL.Simulation.VectorizedEngine

    methods (Static)
        function A_out = applySkip(A_next, A_skip, W_skip)
            % APPLYSKIP  Add the residual shortcut to the main-path output.
            %
            %   A_out = ResidualBlock.applySkip(A_next, A_skip)
            %   A_out = ResidualBlock.applySkip(A_next, A_skip, W_skip)
            %
            %   If W_skip is provided (and non-empty), it is used to project
            %   A_skip from d_in to d_out (Option B, learned projection).
            %   Otherwise A_skip is added unchanged (identity shortcut).
            if nargin < 3 || isempty(W_skip)
                A_out = A_next + A_skip;          % identity shortcut
            else
                A_out = A_next + W_skip * A_skip; % projection shortcut
            end
        end

        function Wskip = makeProjection(d_in, d_out)
            % MAKEPROJECTION  Create a Glorot-initialised projection matrix.
            %
            %   Wskip = ResidualBlock.makeProjection(32, 64)
            %   Returns a (d_out x d_in) weight matrix using Glorot Uniform init.
            bound = sqrt(6 / (d_in + d_out));
            Wskip = (rand(d_out, d_in) * 2 - 1) * bound;
        end
    end
end
