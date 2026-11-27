classdef Recording < handle
    % RECORDING  Read-only view of a Replay file: what was computed, where, and
    %   how -- without the model. Values are read one point at a time.
    %
    %   get(name, coords) returns the NEAREST STORED point and its real axis
    %   values; it never interpolates or blends (Format spec Sec. 6).
    %   coords: struct with one field per axis of the channel, or a numeric row
    %   in the channel's axis order.
    %
    %   Example:
    %     r = OceanReplay.Core.Recording("run.h5");
    %     [x, at] = r.get("x", struct('step', 33));   % at.step = 40 if 40 is the nearest stored
    %     [steps, err] = r.series("x0Err");

    properties (SetAccess = private)
        File (1,1) string
        Meta (1,1) struct = struct()
        Status (1,1) string = ""
        SchemaVersion (1,1) string = ""
        Axes OceanReplay.Core.Axis = OceanReplay.Core.Axis.empty
        Channels OceanReplay.Core.ChannelInfo = OceanReplay.Core.ChannelInfo.empty
        Views OceanReplay.Core.ViewInfo = OceanReplay.Core.ViewInfo.empty
        Reader
    end

    properties (Access = private)
        PointCache      % name -> struct(Index, Values, P)
    end

    methods
        function obj = Recording(file)
            arguments
                file (1,1) string {mustBeFile}
            end
            obj.File = file;
            R = OceanReplay.IO.Hdf5Reader(file);
            obj.Reader = R;
            if R.text('/', 'format') ~= "ocean-replay"
                error('OceanReplay:Recording:format', '"%s" is not an ocean-replay file.', file);
            end
            obj.SchemaVersion = R.text('/', 'schemaVersion');
            if ~startsWith(obj.SchemaVersion, "1.")
                error('OceanReplay:Recording:version', 'Unsupported schema version "%s" (this reader: 1.x).', obj.SchemaVersion);
            end
            obj.Status = R.text('/', 'status');
            m = R.text('/', 'meta');
            if strlength(m) > 0, obj.Meta = jsondecode(char(m)); end
            for a = R.datasets('/axes')
                p = "/axes/" + a;
                obj.Axes(end+1) = OceanReplay.Core.Axis(a, reshape(double(R.readAll(p)), 1, []), ...
                    'Symbol', R.text(p, 'symbol'), 'Units', R.text(p, 'units'), 'Description', R.text(p, 'description'));
            end
            for n = R.children('/channels')
                obj.Channels(end+1) = OceanReplay.Core.ChannelInfo.fromAttrs(n, obj.attrsOf("/channels/" + n, false));
            end
            for n = R.children('/views')
                obj.Views(end+1) = OceanReplay.Core.ViewInfo.fromAttrs(n, obj.attrsOf("/views/" + n, true));
            end
            obj.PointCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end

        function n = names(obj)
            n = [OceanReplay.Core.Recording.namesOf(obj.Channels), OceanReplay.Core.Recording.namesOf(obj.Views)];
        end

        function tf = isView(obj, name)
            tf = any(OceanReplay.Core.Recording.namesOf(obj.Views) == name);
        end

        function tf = isPartial(obj)
            tf = obj.Status ~= "complete";
        end

        function info = info(obj, name)
            % ChannelInfo / ViewInfo of a name.
            arguments
                obj
                name (1,1) string
            end
            k = find(OceanReplay.Core.Recording.namesOf(obj.Channels) == name, 1);
            if ~isempty(k), info = obj.Channels(k); return; end
            k = find(OceanReplay.Core.Recording.namesOf(obj.Views) == name, 1);
            if ~isempty(k), info = obj.Views(k); return; end
            error('OceanReplay:Recording:name', 'No channel or view "%s" (available: %s).', name, strjoin(obj.names(), ', '));
        end

        function ax = axis(obj, name)
            k = find(OceanReplay.Core.Recording.namesOf(obj.Axes) == name, 1);
            if isempty(k), error('OceanReplay:Recording:axis', 'No axis "%s".', name); end
            ax = obj.Axes(k);
        end

        function s = stored(obj, name)
            % The stored points of a name: Index (P x A, 0-based) and Values (P x A axis values), in recording order.
            arguments
                obj
                name (1,1) string
            end
            key = char(name);
            if isKey(obj.PointCache, key), s = obj.PointCache(key); return; end
            info = obj.info(name); base = obj.pathOf(name);
            sz = obj.Reader.sizeOf(base + "/data");
            P = sz(end); if numel(sz) == numel(info.ValueShape), P = 1; end    % trailing singleton P dropped by h5info
            A = numel(info.Axes);
            if A == 0
                s = struct('Index', zeros(1, 0), 'Values', zeros(1, 0), 'P', P);
            else
                I = double(reshape(obj.Reader.readAll(base + "/points"), A, [])).';
                V = zeros(size(I));
                for a = 1:A
                    vals = obj.axis(info.Axes(a)).Values;
                    V(:, a) = vals(I(:, a) + 1);
                end
                s = struct('Index', I, 'Values', V, 'P', size(I, 1));
            end
            obj.PointCache(key) = s;
        end

        function [value, actual, p] = get(obj, name, coords, mode)
            % Value of `name` at the nearest stored point ("nearest") or exactly there ("exact").
            arguments
                obj
                name (1,1) string
                coords = struct()
                mode (1,1) string {mustBeMember(mode, ["nearest" "exact"])} = "nearest"
            end
            info = obj.info(name);
            s = obj.stored(name);
            A = numel(info.Axes);
            if A == 0
                p = 1; actual = struct();
            else
                c = obj.coordRow(info, coords);
                cand = (1:s.P).';
                for a = 1:A
                    d = abs(s.Values(cand, a) - c(a));
                    if mode == "exact"
                        cand = cand(d <= 1e-9 * max(1, abs(c(a))));
                    else
                        cand = cand(d == min(d));
                    end
                    if isempty(cand), break; end
                end
                if isempty(cand)
                    error('OceanReplay:Recording:notStored', '"%s" is not stored at %s.', name, mat2str(c));
                end
                p = cand(1);                                                      % ties -> recorded first
                actual = struct();
                for a = 1:A, actual.(info.Axes(a)) = s.Values(p, a); end
            end
            value = obj.Reader.readPoint(obj.pathOf(name) + "/data", info.ValueShape, p);
            if info.Kind == "tokens", value = reshape(value, 1, []); end
        end

        function [x, y] = series(obj, name)
            % All stored values of a scalar channel over its single axis (recording order).
            info = obj.info(name);
            if info.Kind ~= "scalar" || numel(info.Axes) ~= 1
                error('OceanReplay:Recording:series', 'series() needs a scalar channel over one axis; "%s" is %s over %d axes.', ...
                    name, info.Kind, numel(info.Axes));
            end
            s = obj.stored(name);
            x = s.Values;
            y = reshape(double(obj.Reader.readAll(obj.pathOf(name) + "/data")), [], 1);
        end
    end

    methods (Static)
        function n = namesOf(objs)
            % Names of an Axis / ChannelInfo / ViewInfo array as a string row (empty-safe).
            n = strings(1, 0);
            for i = 1:numel(objs), n(i) = objs(i).Name; end
        end
    end

    methods (Access = private)
        function p = pathOf(obj, name)
            if obj.isView(name), p = "/views/" + name; else, p = "/channels/" + name; end
        end

        function a = attrsOf(obj, path, isView)
            R = obj.Reader;
            a = struct('kind', R.text(path, 'kind'), 'axes', R.list(path, 'axes'), ...
                'valueShape', R.number(path, 'valueShape'), 'dtype', R.text(path, 'dtype'), ...
                'symbol', R.text(path, 'symbol'), 'units', R.text(path, 'units'), ...
                'description', R.text(path, 'description'), 'level', R.number(path, 'level', -1));
            if a.level < 0, a.level = NaN; end
            for f = ["displayRange" "dimFormat" "categories" "rowLabels" "colLabels" "labelsChannel" "symbolLatex"]
                if R.hasAttr(path, f)
                    switch f
                        case "displayRange", a.(f) = R.number(path, f);
                        case {"categories", "rowLabels", "colLabels"}, a.(f) = R.list(path, f);
                        otherwise, a.(f) = R.text(path, f);
                    end
                end
            end
            if isView
                a.method = R.text(path, 'method'); a.reference = R.text(path, 'reference');
                a.inTexts = R.number(path, 'inTexts', 0);
                a.params = R.text(path, 'params'); a.methodId = R.text(path, 'methodId');
            end
        end

        function c = coordRow(~, info, coords)
            A = numel(info.Axes);
            if isnumeric(coords)
                if numel(coords) ~= A
                    error('OceanReplay:Recording:coords', '"%s" has %d axes (%s); got %d coordinates.', ...
                        info.Name, A, strjoin(info.Axes, ', '), numel(coords));
                end
                c = double(coords(:)).'; return;
            end
            c = zeros(1, A);
            for a = 1:A
                if ~isfield(coords, info.Axes(a))
                    error('OceanReplay:Recording:coords', '"%s" needs a coordinate on axis "%s".', info.Name, info.Axes(a));
                end
                c(a) = coords.(info.Axes(a));
            end
        end
    end
end
