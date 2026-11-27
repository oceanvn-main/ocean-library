classdef Schema
    % SCHEMA  Validates a Replay file against the format (Format spec Sec. 7).
    %   issues = OceanReplay.IO.Schema.validate(file) returns one string per
    %   violation, each starting with its HDF5 path. Empty = valid. Both the
    %   MATLAB and the Python implementation check exactly these rules.
    %
    %   Example:
    %     issues = OceanReplay.IO.Schema.validate("run.h5");
    %     assert(isempty(issues), strjoin(issues, newline));

    properties (Constant)
        Format = "ocean-replay"
        Major = "1"
    end

    methods (Static)
        function issues = validate(file)
            arguments
                file (1,1) string {mustBeFile}
            end
            issues = strings(0, 1);
            R = OceanReplay.IO.Hdf5Reader(file);
            % 1. root
            if R.text('/', 'format') ~= OceanReplay.IO.Schema.Format
                issues(end+1) = "/: attribute format must be ""ocean-replay"""; return;
            end
            v = R.text('/', 'schemaVersion');
            if ~startsWith(v, OceanReplay.IO.Schema.Major + ".")
                issues(end+1) = sprintf('/: unsupported schemaVersion "%s" (major %s expected)', v, OceanReplay.IO.Schema.Major);
            end
            if ~ismember(R.text('/', 'status'), ["partial" "complete"])
                issues(end+1) = "/: status must be ""partial"" or ""complete""";
            end
            % 6. meta
            m = R.text('/', 'meta');
            try
                meta = jsondecode(char(m));
                for f = ["source" "seeds" "spec"]
                    if ~isfield(meta, f), issues(end+1) = sprintf('/: meta has no "%s"', f); end %#ok<AGROW>
                end
            catch
                issues(end+1) = "/: meta is missing or not valid JSON";
            end
            % 2. axes
            axLen = containers.Map('KeyType', 'char', 'ValueType', 'double');
            for a = R.datasets('/axes')
                p = "/axes/" + a; sz = R.sizeOf(p);
                vals = R.readAll(p);
                if ~isa(vals, 'double'), issues(end+1) = sprintf('%s: axis must be float64', p); end %#ok<AGROW>
                if nnz(sz > 1) > 1, issues(end+1) = sprintf('%s: axis must be 1-D', p); end %#ok<AGROW>
                axLen(char(a)) = numel(vals);
            end
            % 3-5. channels and views
            for where = ["channels" "views"]
                for n = R.children("/" + where)
                    issues = [issues; OceanReplay.IO.Schema.checkEntry(R, "/" + where + "/" + n, where == "views", axLen)]; %#ok<AGROW>
                end
            end
        end
    end

    methods (Static, Access = private)
        function issues = checkEntry(R, p, isView, axLen)
            issues = strings(0, 1);
            kind = R.text(p, 'kind');
            if ~ismember(kind, OceanReplay.Core.Kinds.All)
                issues(end+1) = sprintf('%s: unknown kind "%s"', p, kind); return;
            end
            vs = R.number(p, 'valueShape');
            msg = OceanReplay.Core.Kinds.checkShape(kind, vs);
            if msg ~= "", issues(end+1) = sprintf('%s: %s', p, msg); return; end
            axesNames = R.list(p, 'axes');
            A = numel(axesNames);
            for a = axesNames
                if ~isKey(axLen, char(a)), issues(end+1) = sprintf('%s: axis "%s" does not exist under /axes', p, a); end %#ok<AGROW>
            end
            dsz = R.sizeOf(p + "/data");
            if isempty(dsz), issues(end+1) = sprintf('%s: no data dataset', p); return; end
            if numel(dsz) ~= numel(vs) + 1 || ~isequal(dsz(1:end-1), vs)
                issues(end+1) = sprintf('%s/data: size %s does not match valueShape %s plus points', p, mat2str(dsz), mat2str(vs));
            end
            P = dsz(end);
            if A == 0
                if P ~= 1, issues(end+1) = sprintf('%s: a channel without axes must store exactly 1 point, has %d', p, P); end
                if ~isempty(R.sizeOf(p + "/points")), issues(end+1) = sprintf('%s: a channel without axes must not have points', p); end
            else
                psz = R.sizeOf(p + "/points");
                if isempty(psz)
                    issues(end+1) = sprintf('%s: no points dataset', p);
                elseif psz(1) ~= A || psz(end) ~= P
                    issues(end+1) = sprintf('%s/points: size %s, expected [%d %d] (axes x points)', p, mat2str(psz), A, P);
                elseif all(isKey(axLen, cellstr(axesNames)))
                    I = double(reshape(R.readAll(p + "/points"), A, []));
                    for a = 1:A
                        n = axLen(char(axesNames(a)));
                        if any(I(a, :) < 0 | I(a, :) >= n)
                            issues(end+1) = sprintf('%s/points: index outside axis "%s" (length %d)', p, axesNames(a), n); %#ok<AGROW>
                        end
                    end
                end
            end
            if kind == "image"
                if numel(R.number(p, 'displayRange')) ~= 2, issues(end+1) = sprintf('%s: an image needs displayRange [lo hi]', p); end
                if R.text(p, 'dimFormat') == "", issues(end+1) = sprintf('%s: an image needs dimFormat', p); end
            end
            if isView
                if R.text(p, 'method') == "", issues(end+1) = sprintf('%s: a view needs a non-empty method (HOW IT WAS MADE)', p); end
                if R.text(p, 'reference') == "", issues(end+1) = sprintf('%s: a view needs a non-empty reference', p); end
                t = R.number(p, 'inTexts');
                if ~isscalar(t) || ~ismember(t, [0 1]), issues(end+1) = sprintf('%s: a view needs inTexts = 0 or 1', p); end
            end
        end
    end
end
