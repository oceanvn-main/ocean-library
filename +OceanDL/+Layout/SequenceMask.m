classdef SequenceMask
    % SEQUENCEMASK  Validity masks for right-padded sequence batches (CBT layout).
    %
    %   A batch of N sequences of lengths T_n is stored as one D x N x T array,
    %   T = max_n T_n, zero-padded on the right. The mask says which steps are real:
    %     M(n, t) = (t <= T_n)            N x T logical, true = valid
    %   apply() multiplies a CBT tensor by the mask broadcast over features:
    %     Y(:, n, t) = M(n, t) * X(:, n, t)
    %   It is an elementwise product with a CONSTANT, so it is differentiable
    %   (OceanDL.Autograd.Functions.Mul) and its VJP is the same mask:
    %     dL/dX(:, n, t) = M(n, t) * dL/dY(:, n, t)   -- padded steps get zero gradient.
    %   (Being a product, a non-finite value in a padded step would give NaN;
    %   padding in this library is zeros, so this never arises.)
    %
    % USAGE:
    %   M = OceanDL.Layout.SequenceMask.fromLengths([5 3], 5);   % 2 x 5
    %   Y = OceanDL.Layout.SequenceMask.apply(X, M);             % X: D x 2 x 5 (Tensor or array)
    %
    % See also: OceanDL.Layout.DimFormat.maskFromLengths, OceanDL.nn.Conv1d.outputLength

    methods (Static)
        function M = fromLengths(lengths, T)
            % FROMLENGTHS  lengths (vector of N valid lengths) -> N x T logical mask.
            %   T defaults to max(lengths); every length must be <= T.
            arguments
                lengths {mustBeVector(lengths, 'allow-all-empties'), mustBeNonnegative, mustBeInteger}
                T (1,1) double {mustBeNonnegative, mustBeInteger} = max([double(lengths(:)); 0])
            end
            lengths = double(lengths(:));
            if any(lengths > T)
                error('OceanDL:Layout:SequenceMask:LengthExceedsT', ...
                    'A length (max %d) exceeds the padded length T = %d.', max(lengths), T);
            end
            M = (1:T) <= lengths;
            M = reshape(M, numel(lengths), T);           % keeps N x 0 / 0 x T shapes exact
        end

        function Y = apply(X, M)
            % APPLY  Zero the padded steps of a CBT Tensor or array (differentiable).
            %   X (D x N x T), M (N x T) logical -> Y, same type as X.
            arguments
                X
                M logical
            end
            [~, N, T] = size(X, 1:3);
            if ~isequal(size(M), [N T])
                error('OceanDL:Layout:SequenceMask:SizeMismatch', ...
                    'Mask is %s, expected N x T = %d x %d for a CBT input.', mat2str(size(M)), N, T);
            end
            m = reshape(cast(M, 'like', OceanDL.Autograd.Tensor.raw(X)), 1, N, T);
            if isa(X, 'OceanDL.Autograd.Tensor')
                Y = OceanDL.Autograd.Functions.Mul().apply(X, m);
            else
                Y = X .* m;
            end
        end
    end
end
