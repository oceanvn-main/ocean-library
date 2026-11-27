classdef Hdf5Reader < handle
    % HDF5READER  Reads Replay files one point at a time (hyperslabs), and
    %   normalises attributes across writers (Format spec Sec. 3.4):
    %   text -> string, lists -> string row, numbers -> double row,
    %   whether MATLAB (string, shape [n 1]) or h5py (char, 1-D) wrote them.
    %   ReadCount counts dataset reads (used by the lazy-read test).
    %
    %   Example:
    %     r = OceanReplay.IO.Hdf5Reader("run.h5");
    %     x = r.readPoint("/channels/x/data", [64 64 3], 12);

    properties (SetAccess = private)
        File (1,1) string
        Info                         % h5info of the whole file (structure + attributes)
        ReadCount (1,1) double = 0
    end

    methods
        function obj = Hdf5Reader(file)
            arguments
                file (1,1) string {mustBeFile}
            end
            obj.File = file;
            obj.Info = h5info(char(file));
        end

        function tf = hasAttr(obj, path, name)
            a = obj.attrList(path);
            tf = ~isempty(a) && any(strcmp({a.Name}, name));
        end

        function v = text(obj, path, name, default)
            % A scalar text attribute as a string ("" / default when absent).
            if nargin < 4, default = ""; end
            if ~obj.hasAttr(path, name), v = string(default); return; end
            v = OceanReplay.IO.Hdf5Reader.toStrings(h5readatt(char(obj.File), char(path), name));
            if isempty(v), v = ""; else, v = v(1); end
        end

        function v = list(obj, path, name)
            % A string-list attribute as a 1 x n string row (strings(1,0) when absent or empty).
            v = strings(1, 0);
            if ~obj.hasAttr(path, name), return; end
            v = reshape(OceanReplay.IO.Hdf5Reader.toStrings(h5readatt(char(obj.File), char(path), name)), 1, []);
            v = v(strlength(v) > 0);                     % an empty list may read back as one "" (MATLAB writer)
        end

        function v = number(obj, path, name, default)
            % A numeric attribute as a double row ([] / default when absent).
            if nargin < 4, default = []; end
            if ~obj.hasAttr(path, name), v = default; return; end
            v = reshape(double(h5readatt(char(obj.File), char(path), name)), 1, []);
        end

        function names = children(obj, parent)
            % Names of the groups directly under parent ("/channels", "/views").
            names = strings(1, 0);
            g = obj.group(parent);
            if isempty(g), return; end
            for k = 1:numel(g.Groups)
                [~, n, e] = fileparts(g.Groups(k).Name);
                names(end+1) = string([n e]); %#ok<AGROW>
            end
        end

        function names = datasets(obj, parent)
            names = strings(1, 0);
            g = obj.group(parent);
            if isempty(g), return; end
            names = string({g.Datasets.Name});
        end

        function sz = sizeOf(obj, path)
            % MATLAB-order dataspace size of a dataset ([] when absent).
            [parent, n, e] = fileparts(char(path));
            parent = strrep(parent, '\', '/');
            g = obj.group(parent); sz = [];
            if isempty(g) || isempty(g.Datasets), return; end
            k = find(strcmp({g.Datasets.Name}, [n e]), 1);
            if ~isempty(k), sz = g.Datasets(k).Dataspace.Size; end
        end

        function v = readAll(obj, path)
            obj.ReadCount = obj.ReadCount + 1;
            v = h5read(char(obj.File), char(path));
        end

        function v = readPoint(obj, path, valueShape, p)
            % One point (1-based p) of a [valueShape P] dataset, in logical shape.
            obj.ReadCount = obj.ReadCount + 1;
            r = numel(valueShape);
            v = h5read(char(obj.File), char(path), [ones(1, r) p], [valueShape 1]);
            if isstring(v) || iscell(v) || ischar(v)
                v = reshape(OceanReplay.IO.Hdf5Reader.toStrings(v), [], 1);
            elseif r == 1
                v = reshape(v, [], 1);
            else
                v = reshape(v, valueShape);
            end
        end

        function g = group(obj, path)
            % h5info struct of a group ([] when absent).
            path = char(path); g = obj.Info;
            if strcmp(path, '/'), return; end
            parts = strsplit(strip(path, 'left', '/'), '/');
            cur = '';
            for i = 1:numel(parts)
                cur = [cur '/' parts{i}]; %#ok<AGROW>
                if isempty(g.Groups), g = []; return; end
                k = find(strcmp({g.Groups.Name}, cur), 1);
                if isempty(k), g = []; return; end
                g = g.Groups(k);
            end
        end
    end

    methods (Access = private)
        function a = attrList(obj, path)
            % Attributes of a group or dataset.
            g = obj.group(path);
            if ~isempty(g), a = g.Attributes; return; end
            [parent, n, e] = fileparts(char(path));
            pg = obj.group(strrep(parent, '\', '/')); a = [];
            if isempty(pg) || isempty(pg.Datasets), return; end
            k = find(strcmp({pg.Datasets.Name}, [n e]), 1);
            if ~isempty(k), a = pg.Datasets(k).Attributes; end
        end
    end

    methods (Static)
        function s = toStrings(v)
            % Any HDF5 text representation -> string column.
            if isstring(v), s = v(:);
            elseif iscell(v), s = string(v(:));
            elseif ischar(v), s = string(cellstr(v));
            elseif isempty(v), s = strings(0, 1);
            else, s = string(v(:));
            end
        end
    end
end
