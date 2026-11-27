classdef Kinds
    % KINDS  The value kinds of the Replay format and their shape rules
    %   (Planning/System/Replay/Replay_Format_Spec.md, Sec. 2).
    %
    %   Shapes are LOGICAL (as MATLAB sees one value):
    %     scalar   [1]            vector  [n]           tokens [n] (strings)
    %     matrix   [r c]          sequence [T D]        points [N d], d = 2 or 3
    %     image    [H W C]  (dimFormat "HWC")
    %
    %   Example:
    %     msg = OceanReplay.Core.Kinds.checkValue("image", [4 4 3], rand(4, 4, 3))   % "" = valid

    properties (Constant)
        All = ["scalar" "vector" "matrix" "image" "sequence" "points" "tokens"]
    end

    methods (Static)
        function r = rank(kind)
            % Number of logical value dimensions of a kind.
            arguments
                kind (1,1) string
            end
            switch kind
                case {"scalar", "vector", "tokens"}, r = 1;
                case {"matrix", "sequence", "points"}, r = 2;
                case "image", r = 3;
                otherwise
                    error('OceanReplay:Kinds:unknown', 'Unknown kind "%s". Known kinds: %s.', ...
                        kind, strjoin(OceanReplay.Core.Kinds.All, ', '));
            end
        end

        function msg = checkShape(kind, valueShape)
            % "" when valueShape is a legal shape for kind, otherwise the reason.
            arguments
                kind (1,1) string
                valueShape (1,:) double
            end
            msg = "";
            if ~ismember(kind, OceanReplay.Core.Kinds.All)
                msg = sprintf('unknown kind "%s"', kind); return;
            end
            r = OceanReplay.Core.Kinds.rank(kind);
            if numel(valueShape) ~= r || any(valueShape < 1) || any(valueShape ~= round(valueShape))
                msg = sprintf('kind "%s" needs %d positive integer dimension(s), got %s', kind, r, mat2str(valueShape));
            elseif kind == "scalar" && valueShape ~= 1
                msg = sprintf('kind "scalar" needs valueShape [1], got %s', mat2str(valueShape));
            elseif kind == "points" && ~ismember(valueShape(2), [2 3])
                msg = sprintf('kind "points" needs N x 2 or N x 3, got %s', mat2str(valueShape));
            end
        end

        function msg = checkValue(kind, valueShape, v)
            % "" when value v has kind and exactly valueShape, otherwise the reason.
            arguments
                kind (1,1) string
                valueShape (1,:) double
                v
            end
            msg = OceanReplay.Core.Kinds.checkShape(kind, valueShape);
            if msg ~= "", return; end
            if kind == "tokens"
                if ~(isstring(v) || iscellstr(v)) || numel(v) ~= valueShape
                    msg = sprintf('kind "tokens" needs %d strings', valueShape);
                end
                return;
            end
            if ~(isnumeric(v) || islogical(v)) || ~isreal(v)
                msg = sprintf('kind "%s" needs real numeric values, got %s', kind, class(v)); return;
            end
            r = OceanReplay.Core.Kinds.rank(kind);
            if numel(v) ~= prod(valueShape) || (r >= 2 && ~isequal(size(v, 1:r), valueShape)) || (r == 1 && ~isvector(v))
                msg = sprintf('kind "%s" needs size %s, got %s', kind, mat2str(valueShape), mat2str(size(v)));
            end
        end

        function v = normalise(kind, v)
            % Canonical in-memory orientation: vectors/tokens as columns, gathered, no gpuArray.
            arguments
                kind (1,1) string
                v
            end
            if isa(v, 'gpuArray'), v = gather(v); end
            if isa(v, 'dlarray'), v = extractdata(v); end
            switch kind
                case "tokens", v = reshape(string(v), [], 1);
                case {"vector", "scalar"}, v = reshape(v, [], 1);
            end
        end

        function n = bytesPerElement(dtype)
            % Storage size of one element (strings: a rough 16 bytes, for size estimates only).
            arguments
                dtype (1,1) string
            end
            switch dtype
                case {"float32", "int32"}, n = 4;
                case "float64", n = 8;
                case "string", n = 16;
                otherwise, error('OceanReplay:Kinds:dtype', 'Unsupported dtype "%s" (float32 | float64 | int32 | string).', dtype);
            end
        end

        function c = matlabClass(dtype)
            % MATLAB class / h5create Datatype of a stored dtype.
            arguments
                dtype (1,1) string
            end
            switch dtype
                case "float32", c = 'single';
                case "float64", c = 'double';
                case "int32",   c = 'int32';
                case "string",  c = 'string';
                otherwise, error('OceanReplay:Kinds:dtype', 'Unsupported dtype "%s" (float32 | float64 | int32 | string).', dtype);
            end
        end
    end
end
