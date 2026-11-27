classdef DifferenceOperatorAffinity < OceanMath.Operators.IDifferenceOperator
    % DIFFERENCEOPERATORAFFINITY  Luminance-guided bilateral difference operator.
    %
    %   Part of the Domain-Agnostic Regularization Architecture.
    %   Builds sparse affinity-weighted operator L = I - W, where W is the 4/8-neighbor
    %   row-normalized affinity matrix (Levin et al. 2004):
    %
    %       w_rs = exp( - (Y(r) - Y(s))^2 / (2 * sigma^2) )
    %
    %   Package: OceanMath.Operators
    %   Author: Ocean Library  |  2026-08-05

    properties
        Sigma        (1,1) double {mustBePositive} = 0.05
        Neighborhood (1,1) double {mustBeMember(Neighborhood, [4, 8])} = 8
        GrayscaleImage double
    end

    methods
        function obj = DifferenceOperatorAffinity(I_gray, options)
            % DIFFERENCEOPERATORAFFINITY Constructor.
            arguments
                I_gray double = []
                options.Sigma        (1,1) double = 0.05
                options.Neighborhood (1,1) double = 8
            end
            obj.Sigma        = options.Sigma;
            obj.Neighborhood = options.Neighborhood;
            if ~isempty(I_gray)
                I_gray = double(I_gray);
                if max(I_gray(:)) > 1.0
                    I_gray = I_gray / 255.0;
                end
                obj.GrayscaleImage = I_gray;
            end
        end

        function D = buildMatrix(obj, shape)
            % BUILDMATRIX Build sparse affinity difference operator L = I - W.
            if nargin >= 2 && ~isempty(shape)
                H = shape(1);
                W_dim = shape(2);
            else
                [H, W_dim] = size(obj.GrayscaleImage);
            end

            N = H * W_dim;
            if isempty(obj.GrayscaleImage)
                error('DifferenceOperatorAffinity:MissingImage', 'Grayscale image data is required to construct affinity matrix.');
            end

            W_affinity = obj.buildAffinityMatrix(obj.GrayscaleImage);
            D = speye(N) - W_affinity;
        end

        function d = apply(obj, f, shape)
            if nargin < 3
                shape = size(f);
            end
            D = obj.buildMatrix(shape);
            d = D * f(:);
        end

        function str = getName(obj)
            str = sprintf('Bilateral Affinity (sigma=%.2f, %d-nbr)', obj.Sigma, obj.Neighborhood);
        end
    end

    methods (Access = private)
        function W_affinity = buildAffinityMatrix(obj, Y)
            [H, W_dim] = size(Y);
            N = H * W_dim;

            if obj.Neighborhood == 8
                offsets = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
            else
                offsets = [-1 0; 0 -1; 0 1; 1 0];
            end

            num_nbrs = size(offsets, 1);
            max_entries = N * num_nbrs;

            row_idx = zeros(max_entries, 1);
            col_idx = zeros(max_entries, 1);
            vals    = zeros(max_entries, 1);
            count = 0;

            [grid_r, grid_c] = ndgrid(1:H, 1:W_dim);

            for k = 1:num_nbrs
                dr = offsets(k, 1);
                dc = offsets(k, 2);

                r_nbr = grid_r + dr;
                c_nbr = grid_c + dc;

                valid = (r_nbr >= 1 & r_nbr <= H & c_nbr >= 1 & c_nbr <= W_dim);

                idx_src = grid_r(valid) + (grid_c(valid) - 1) * H;
                idx_tgt = r_nbr(valid)  + (c_nbr(valid)  - 1) * H;

                Y_src = Y(idx_src);
                Y_tgt = Y(idx_tgt);

                w_raw = exp(-((Y_src - Y_tgt).^2) / (2 * obj.Sigma^2)) + 1e-6;

                n_pairs = numel(idx_src);
                row_idx(count+1 : count+n_pairs) = idx_src;
                col_idx(count+1 : count+n_pairs) = idx_tgt;
                vals(count+1 : count+n_pairs)    = w_raw;
                count = count + n_pairs;
            end

            row_idx = row_idx(1:count);
            col_idx = col_idx(1:count);
            vals    = vals(1:count);

            W_raw = sparse(row_idx, col_idx, vals, N, N);

            % Row-normalize so each row sums to 1
            row_sums = sum(W_raw, 2);
            row_sums(row_sums == 0) = 1.0;
            W_affinity = spdiags(1.0 ./ row_sums, 0, N, N) * W_raw;
        end
    end
end
