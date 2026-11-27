classdef Latent
    % LATENT  Layout helpers shared by VAE, quantizers and priors. A latent is
    %   either a feature map "SSCB" (h x w x C x N, conv autoencoders) or a
    %   vector "CB" (C x N, MLP autoencoders) -- the repo's DimFormat labels.
    %   The layout is always passed explicitly: size() cannot tell an
    %   (h x w x C x 1) map from a matrix (MATLAB drops trailing singletons).

    methods (Static)
        function d = channelDim(layout)
            if layout == "SSCB", d = 3; else, d = 1; end
        end

        function n = batchSize(X, layout)
            if layout == "SSCB", n = size(X, 4); else, n = size(X, 2); end
        end

        function [Zc, shape] = toColumns(Z, layout)
            % One column per latent position: (C x M), M = h*w*N (column-major
            % position order, sample slowest). Works on Tensors and arrays.
            if layout == "CB"
                Zc = Z; shape = size(Z, 1:2);
                return;
            end
            shape = size(Z, 1:4);
            Zc = reshape(permute(Z, [3 1 2 4]), shape(3), shape(1) * shape(2) * shape(4));
        end

        function Z = fromColumns(Zc, shape, layout)
            % Inverse of toColumns; the channel count is taken from Zc (it may
            % differ from shape(3) after a projection).
            if layout == "CB"
                Z = Zc;
                return;
            end
            C = size(Zc, 1);
            Z = permute(reshape(Zc, C, shape(1), shape(2), shape(4)), [2 3 1 4]);
        end

        function sz = indexShape(shape, layout)
            % Shape of a per-position index/code array: (h x w x N) or (N x 1).
            if layout == "CB", sz = [shape(2) 1]; else, sz = [shape(1) shape(2) shape(4)]; end
        end

        function shape = shapeFromIndices(idx, C, layout)
            % Latent shape for a code index array (inverse of indexShape).
            if layout == "CB"
                shape = [C numel(idx)];
            else
                shape = [size(idx, 1) size(idx, 2) C size(idx, 3)];
            end
        end
    end
end
