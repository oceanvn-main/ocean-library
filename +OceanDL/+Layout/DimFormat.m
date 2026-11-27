classdef DimFormat
    % DIMFORMAT  The repo's canonical tensor layouts, and the mandatory
    %   check-and-convert step for data coming from anywhere else.
    %
    % CANONICAL LAYOUTS (column-major "features x observations", decision B):
    %   "CB"    features x batch                 dense / tabular      (D x N)
    %   "CBT"   features x batch x time          sequences            (D x N x T)
    %   "SSCB"  height x width x channels x batch images              (H x W x C x N)
    %   Letters follow MATLAB's dlarray: C = channel/feature, B = batch
    %   (observation), T = time, S = spatial. In every canonical layout ONE
    %   observation (and, for "CBT", one time step) is a CONTIGUOUS block of
    %   memory -- the property that makes mini-batch gathering and the
    %   per-step W*x_t fast in column-major MATLAB (measured 3.5x vs strided).
    %
    % WHY A CHECK IS MANDATORY AT IMPORT:
    %   The same bytes mean different things in different ecosystems.
    %   PyTorch / NumPy are row-major and usually batch-first ("BC", "BTC",
    %   "BCSS" = NCHW); CSV/table data has observations as ROWS ("BC").
    %   A wrong layout rarely crashes -- a 64 x 64 batch of 64-dim samples
    %   trains happily on transposed data -- so conversion must be explicit
    %   and validated, never guessed.
    %
    % TWO DIFFERENT PROBLEMS, TWO FUNCTIONS:
    %   1. MEMORY order (row-major buffer -> MATLAB array with the SAME index
    %      meaning):  fromRowMajor(buffer, shape)
    %      A C-order array of shape (d1,...,dk) has its LAST index fastest;
    %      MATLAB's FIRST index is fastest. Reading the buffer as
    %      (dk,...,d1) and reversing the axes restores A(i1,...,ik).
    %   2. AXIS meaning (which axis is batch / time / feature):
    %      toCanonical(X, sourceFormat)   -> permute to "CB" / "CBT" / "SSCB"
    %
    % USAGE:
    %   Xc = DimFormat.toCanonical(Xpt, "BTC");          % PyTorch batch_first seq -> D x N x T
    %   Xc = DimFormat.toCanonical(imgs, "BCSS");        % NCHW -> H x W x C x N
    %   Xc = DimFormat.fromTable(T, ["age", "hours"]);   % columnar table -> D x N
    %   DimFormat.validate(Xc, "CBT", 'FeatureDim', 32, 'Mask', m);
    %
    % See also: OceanData.Adapters.NpyReader

    properties (Constant)
        Canonical = ["CB", "CBT", "SSCB"]
    end

    methods (Static)
        function [Y, perm, target] = toCanonical(X, sourceFormat)
            % TOCANONICAL  Permute X, whose axes are labelled by sourceFormat,
            %   into the canonical layout with the same set of labels.
            arguments
                X
                sourceFormat (1,1) string
            end
            src = OceanDL.Layout.DimFormat.parse(sourceFormat, X);
            target = OceanDL.Layout.DimFormat.targetFor(src);
            perm = OceanDL.Layout.DimFormat.permutationTo(src, char(target));
            if isequal(perm, 1:numel(perm))
                Y = X;
            else
                Y = permute(X, perm);
            end
        end

        function A = fromRowMajor(buffer, shape)
            % FROMROWMAJOR  C-order (row-major) buffer of the given shape ->
            %   MATLAB array A with A(i1,...,ik) = source[i1-1, ..., ik-1].
            arguments
                buffer
                shape (1,:) double {mustBeNonnegative, mustBeInteger}
            end
            if numel(buffer) ~= prod(shape)
                error('OceanDL:Layout:DimFormat:ShapeMismatch', ...
                    'Buffer has %d elements, shape [%s] needs %d.', numel(buffer), num2str(shape), prod(shape));
            end
            k = numel(shape);
            if k <= 1
                A = reshape(buffer, [max([shape 1]) 1]);
                return;
            end
            A = permute(reshape(buffer, fliplr(shape)), k:-1:1);
        end

        function X = fromTable(T, variables)
            % FROMTABLE  Columnar table (observations = rows, one variable per
            %   column) -> canonical "CB" matrix (features x observations).
            %   Only numeric/logical variables are accepted: categorical or
            %   text columns must be encoded first (one-hot, integer ids for
            %   nn.Embedding) -- the error lists them.
            arguments
                T table
                variables (1,:) string = string(T.Properties.VariableNames)
            end
            bad = variables(~arrayfun(@(v) isnumeric(T.(v)) || islogical(T.(v)), variables));
            if ~isempty(bad)
                error('OceanDL:Layout:DimFormat:NonNumericColumns', ...
                    'Encode these non-numeric columns before conversion: %s.', strjoin(bad, ', '));
            end
            X = zeros(numel(variables), height(T));
            for k = 1:numel(variables)
                X(k, :) = double(T.(variables(k))).';
            end
        end

        function validate(X, format, opts)
            % VALIDATE  Assert X is in the canonical layout `format`.
            %   Checks rank, element type, finiteness, declared sizes, and an
            %   optional (N x T) mask. When FeatureDim does not match axis 1
            %   but matches another axis, the error names the likely source
            %   layout and the fix -- the common "forgot to transpose" bug.
            arguments
                X
                format (1,1) string {mustBeMember(format, ["CB", "CBT", "SSCB"])}
                opts.FeatureDim double = []          % expected C (for "SSCB": channels)
                opts.BatchSize double = []
                opts.Mask = []
                opts.AllowNonFinite (1,1) logical = false
            end
            fmt = char(format);
            if ~(isnumeric(X) || islogical(X)) || ~isreal(X)
                error('OceanDL:Layout:DimFormat:BadType', ...
                    'Expected a real numeric array, got %s.', class(X));
            end
            if ndims(X) > numel(fmt)
                error('OceanDL:Layout:DimFormat:BadRank', ...
                    'Layout "%s" has %d axes but X has %d.', fmt, numel(fmt), ndims(X));
            end
            sz = size(X, 1:numel(fmt));
            cAxis = find(fmt == 'C', 1);
            bAxis = find(fmt == 'B', 1);
            if ~isempty(opts.FeatureDim) && sz(cAxis) ~= opts.FeatureDim
                hint = '';
                other = find(sz == opts.FeatureDim & (1:numel(sz)) ~= cAxis, 1);
                if ~isempty(other)
                    hint = sprintf([' Axis %d has size %d -- the data is probably in a different layout ', ...
                        '(e.g. observations-first from NumPy/PyTorch/CSV). Convert explicitly with ', ...
                        'DimFormat.toCanonical(X, <source format>).'], other, opts.FeatureDim);
                end
                error('OceanDL:Layout:DimFormat:FeatureDimMismatch', ...
                    'Layout "%s": expected %d features on axis %d, found %d.%s', ...
                    fmt, opts.FeatureDim, cAxis, sz(cAxis), hint);
            end
            if ~isempty(opts.BatchSize) && sz(bAxis) ~= opts.BatchSize
                error('OceanDL:Layout:DimFormat:BatchSizeMismatch', ...
                    'Layout "%s": expected batch %d on axis %d, found %d.', fmt, opts.BatchSize, bAxis, sz(bAxis));
            end
            if ~opts.AllowNonFinite && ~all(isfinite(X(:)))
                error('OceanDL:Layout:DimFormat:NonFinite', ...
                    'X contains %d NaN/Inf values.', nnz(~isfinite(X(:))));
            end
            if ~isempty(opts.Mask)
                if ~strcmp(fmt, 'CBT')
                    error('OceanDL:Layout:DimFormat:MaskNeedsTime', 'A mask is only defined for "CBT".');
                end
                if ~isequal(size(opts.Mask), sz([2 3]))
                    error('OceanDL:Layout:DimFormat:MaskSize', ...
                        'Mask must be N x T = [%d %d], got [%s].', sz(2), sz(3), num2str(size(opts.Mask)));
                end
                m = logical(opts.Mask);
                % Right padding only: once a sequence ends it stays ended.
                if any(diff(double(m), 1, 2) > 0, 'all')
                    error('OceanDL:Layout:DimFormat:MaskNotRightPadded', ...
                        'Mask rows must be 1...1 0...0 (valid steps first, padding at the end).');
                end
            end
        end

        function m = maskFromLengths(lengths, T)
            % MASKFROMLENGTHS  lengths (N x 1) -> right-padded mask (N x T).
            arguments
                lengths (:,1) double {mustBeNonnegative, mustBeInteger}
                T (1,1) double {mustBeNonnegative, mustBeInteger} = max(lengths)
            end
            m = (1:T) <= lengths;
        end
    end

    methods (Static, Access = private)
        function src = parse(format, X)
            src = char(upper(format));
            if ~all(ismember(src, 'CBTS'))
                error('OceanDL:Layout:DimFormat:BadLabel', ...
                    'Format "%s": labels must be C, B, T or S.', src);
            end
            if ndims(X) > numel(src)
                error('OceanDL:Layout:DimFormat:BadRank', ...
                    'Format "%s" labels %d axes but X has %d.', src, numel(src), ndims(X));
            end
            if sum(src == 'C') ~= 1 || sum(src == 'B') ~= 1 || sum(src == 'T') > 1
                error('OceanDL:Layout:DimFormat:BadLabel', ...
                    'Format "%s" needs exactly one C and one B (and at most one T).', src);
            end
        end

        function target = targetFor(src)
            nS = sum(src == 'S');
            if any(src == 'T') && nS == 0 && numel(src) == 3
                target = "CBT";
            elseif nS == 2 && numel(src) == 4
                target = "SSCB";
            elseif numel(src) == 2
                target = "CB";
            else
                error('OceanDL:Layout:DimFormat:NoCanonical', ...
                    'No canonical layout for "%s" (supported: CB, CBT, SSCB).', src);
            end
        end

        function perm = permutationTo(src, tgt)
            % Stable matching: the k-th S of the source goes to the k-th S of the target.
            perm = zeros(1, numel(tgt));
            used = false(1, numel(src));
            for i = 1:numel(tgt)
                j = find(src == tgt(i) & ~used, 1);
                perm(i) = j;
                used(j) = true;
            end
        end
    end
end
