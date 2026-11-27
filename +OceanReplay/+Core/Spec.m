classdef Spec
    % SPEC  What to record and where (Format spec Sec. 5). The SAME JSON drives
    %   the MATLAB recorder, the Python recorder and the extractor.
    %
    %   Groups: {Name, Level, Channels, At}   At = struct, one Sampling per axis
    %                                          (an axis not named = "all")
    %   Views:  {Name, Method, Params, At}    Method = a view id in the domain's catalog
    %   A channel in several groups is stored once per point (union of their points).
    %
    %   Example:
    %     S = OceanReplay.Core.Sampling;
    %     spec = OceanReplay.Core.Spec() ...
    %        .addGroup("numbers", 0, ["alphaBar" "sigma"], struct('step', S.all())) ...
    %        .addGroup("images",  1, ["x" "eps"],          struct('step', S.every(20))) ...
    %        .addView("saliency", "saliency", struct(), struct('step', S.list([50 51 52])));
    %     spec.toFile("spec.json");

    properties
        SchemaVersion (1,1) string = "1.0"
        Precision (1,1) string {mustBeMember(Precision, ["float32" "float64"])} = "float32"
        SizeCapBytes (1,1) double = Inf
        Groups = struct('Name', {}, 'Level', {}, 'Channels', {}, 'At', {})
        Views = struct('Name', {}, 'Method', {}, 'Params', {}, 'At', {})
    end

    methods
        function obj = addGroup(obj, name, level, channels, at)
            arguments
                obj
                name (1,1) string
                level (1,1) double
                channels (1,:) string
                at (1,1) struct = struct()
            end
            obj.Groups(end+1) = struct('Name', name, 'Level', level, 'Channels', channels, 'At', at);
        end

        function obj = addView(obj, name, method, params, at)
            arguments
                obj
                name (1,1) string
                method (1,1) string
                params (1,1) struct = struct()
                at (1,1) struct = struct()
            end
            obj.Views(end+1) = struct('Name', name, 'Method', method, 'Params', params, 'At', at);
        end

        function names = channelNames(obj)
            % Every channel named by any group, in first-mention order.
            names = strings(1, 0);
            for g = obj.Groups, names = [names, g.Channels]; end %#ok<AGROW>
            names = unique(names, 'stable');
        end

        function l = level(obj, name)
            % Lowest level of the groups that record `name` (NaN if none).
            l = NaN;
            for g = obj.Groups
                if ismember(name, g.Channels), l = min(l, g.Level, 'omitnan'); end
            end
        end

        function tf = wants(obj, name, values, index)
            % True when channel `name` is wanted at the point whose axis values /
            % 0-based indices are the fields of `values` / `index`.
            tf = false;
            for g = obj.Groups
                if ismember(name, g.Channels) && OceanReplay.Core.Spec.atContains(g.At, values, index)
                    tf = true; return;
                end
            end
        end

        function tf = viewWants(obj, j, values, index)
            tf = OceanReplay.Core.Spec.atContains(obj.Views(j).At, values, index);
        end

        function issues = check(obj, channels, views, axes)
            % Problems of this spec against a catalog (ChannelInfo / ViewInfo arrays) and
            % its axes (Axis array). Empty = valid.
            issues = strings(0, 1);
            cn = arrayfun(@(c) c.Name, channels); if isempty(cn), cn = strings(1, 0); end
            vn = arrayfun(@(v) v.Name, views);    if isempty(vn), vn = strings(1, 0); end
            an = arrayfun(@(a) a.Name, axes);     if isempty(an), an = strings(1, 0); end
            for g = obj.Groups
                for c = g.Channels
                    k = find(cn == c, 1);
                    if isempty(k)
                        issues(end+1) = sprintf('group "%s": unknown channel "%s" (catalog: %s)', g.Name, c, strjoin(cn, ', ')); %#ok<AGROW>
                        continue;
                    end
                    issues = [issues; obj.checkAt(sprintf('group "%s", channel "%s"', g.Name, c), g.At, channels(k).Axes, axes, an)]; %#ok<AGROW>
                end
            end
            outNames = obj.channelNames();
            for j = 1:numel(obj.Views)
                v = obj.Views(j);
                k = find(vn == v.Method, 1);
                if isempty(k)
                    issues(end+1) = sprintf('view "%s": unknown method "%s" (catalog: %s)', v.Name, v.Method, strjoin(vn, ', ')); %#ok<AGROW>
                    continue;
                end
                if ismember(v.Name, outNames) || sum([obj.Views.Name] == v.Name) > 1
                    issues(end+1) = sprintf('view "%s": name used twice (views and channels share one namespace)', v.Name); %#ok<AGROW>
                end
                issues = [issues; obj.checkAt(sprintf('view "%s"', v.Name), v.At, views(k).Axes, axes, an)]; %#ok<AGROW>
            end
        end

        function validate(obj, channels, views, axes)
            issues = obj.check(channels, views, axes);
            if ~isempty(issues)
                error('OceanReplay:Spec:invalid', 'Invalid recording spec:\n  - %s', strjoin(issues, sprintf('\n  - ')));
            end
        end

        function [total, perName] = estimateBytes(obj, channels, views, axes)
            % Upper-bound size of a recording (closed axes, known shapes). NaN when unknown.
            perName = struct('Name', {}, 'Points', {}, 'Bytes', {});
            for c = obj.channelNames()
                info = channels(arrayfun(@(x) x.Name, channels) == c);
                n = obj.countPoints(info.Axes, axes, @(vals, idx) obj.wants(c, vals, idx));
                perName(end+1) = struct('Name', c, 'Points', n, 'Bytes', n * OceanReplay.Core.Spec.bytesPerValue(info, obj.Precision)); %#ok<AGROW>
            end
            vn = arrayfun(@(v) v.Name, views);
            for j = 1:numel(obj.Views)
                info = views(vn == obj.Views(j).Method);
                n = obj.countPoints(info.Axes, axes, @(vals, idx) obj.viewWants(j, vals, idx));
                perName(end+1) = struct('Name', obj.Views(j).Name, 'Points', n, 'Bytes', n * OceanReplay.Core.Spec.bytesPerValue(info, obj.Precision)); %#ok<AGROW>
            end
            total = sum([perName.Bytes]);
        end

        function t = toStruct(obj)
            % JSON-ready struct (Format spec Sec. 5). Arrays stay arrays even with one element.
            t = struct('schemaVersion', obj.SchemaVersion, 'precision', obj.Precision);
            if isfinite(obj.SizeCapBytes), t.sizeCapBytes = obj.SizeCapBytes; end
            t.groups = arrayfun(@(g) struct('name', g.Name, 'level', g.Level, 'channels', {cellstr(g.Channels)}, ...
                'at', OceanReplay.Core.Spec.atToStruct(g.At)), obj.Groups, 'UniformOutput', false);
            t.views = arrayfun(@(v) struct('name', v.Name, 'method', v.Method, 'params', v.Params, ...
                'at', OceanReplay.Core.Spec.atToStruct(v.At)), obj.Views, 'UniformOutput', false);
        end

        function txt = toJson(obj)
            txt = string(jsonencode(obj.toStruct(), 'PrettyPrint', true));
        end

        function toFile(obj, file)
            arguments
                obj
                file (1,1) string
            end
            fid = fopen(file, 'w', 'n', 'UTF-8');
            if fid < 0, error('OceanReplay:Spec:file', 'Cannot write "%s".', file); end
            c = onCleanup(@() fclose(fid));
            fwrite(fid, unicode2native(char(obj.toJson()), 'UTF-8'));
        end
    end

    methods (Static)
        function obj = fromJson(txt)
            arguments
                txt (1,1) string
            end
            t = jsondecode(char(txt));
            obj = OceanReplay.Core.Spec.fromStruct(t);
        end

        function obj = fromFile(file)
            arguments
                file (1,1) string
            end
            obj = OceanReplay.Core.Spec.fromJson(string(fileread(file, 'Encoding', 'UTF-8')));
        end

        function obj = fromStruct(t)
            obj = OceanReplay.Core.Spec();
            if isfield(t, 'schemaVersion'), obj.SchemaVersion = string(t.schemaVersion); end
            if isfield(t, 'precision'), obj.Precision = string(t.precision); end
            if isfield(t, 'sizeCapBytes') && ~isempty(t.sizeCapBytes), obj.SizeCapBytes = t.sizeCapBytes; end
            if isfield(t, 'groups')
                for g = OceanReplay.Core.Spec.asCell(t.groups)
                    s = g{1};
                    lvl = NaN; if isfield(s, 'level'), lvl = s.level; end
                    obj = obj.addGroup(string(s.name), lvl, string(s.channels(:).'), OceanReplay.Core.Spec.atFromStruct(s));
                end
            end
            if isfield(t, 'views')
                for v = OceanReplay.Core.Spec.asCell(t.views)
                    s = v{1};
                    m = ""; if isfield(s, 'method'), m = string(s.method); end
                    p = struct(); if isfield(s, 'params') && isstruct(s.params), p = s.params; end
                    obj = obj.addView(string(s.name), m, p, OceanReplay.Core.Spec.atFromStruct(s));
                end
            end
        end

        function tf = atContains(at, values, index)
            % All per-axis rules of `at` select the point (axes not in `at` = all).
            tf = true;
            f = fieldnames(at);
            for i = 1:numel(f)
                if ~isfield(values, f{i}), continue; end
                ix = NaN; if isfield(index, f{i}), ix = index.(f{i}); end
                if ~at.(f{i}).contains(values.(f{i}), ix), tf = false; return; end
            end
        end
    end

    methods (Static, Access = private)
        function c = asCell(x)
            if isempty(x), c = {}; elseif iscell(x), c = x(:).'; else, c = num2cell(x(:).'); end
        end

        function at = atFromStruct(s)
            at = struct();
            if ~isfield(s, 'at') || isempty(s.at), return; end
            f = fieldnames(s.at);
            for i = 1:numel(f), at.(f{i}) = OceanReplay.Core.Sampling.fromStruct(s.at.(f{i})); end
        end

        function t = atToStruct(at)
            t = struct();
            f = fieldnames(at);
            for i = 1:numel(f), t.(f{i}) = at.(f{i}).toStruct(); end
        end

        function issues = checkAt(where, at, chanAxes, axes, an)
            issues = strings(0, 1);
            f = string(fieldnames(at)).';
            for a = f
                if ~ismember(a, chanAxes)
                    issues(end+1) = sprintf('%s: rule on axis "%s", which it does not depend on (axes: %s)', ...
                        where, a, strjoin(chanAxes, ', ')); %#ok<AGROW>
                    continue;
                end
                k = find(an == a, 1);
                if isempty(k), continue; end
                if ~axes(k).isOpen()
                    bad = at.(a).missing(axes(k).Values);
                    if ~isempty(bad)
                        issues(end+1) = sprintf('%s: list values %s are not on axis "%s"', where, mat2str(bad), a); %#ok<AGROW>
                    end
                end
            end
            for a = chanAxes
                if ~ismember(a, an)
                    issues(end+1) = sprintf('%s: depends on axis "%s", which the domain does not declare', where, a); %#ok<AGROW>
                end
            end
        end

        function n = countPoints(chanAxes, axes, wantsFcn)
            % Number of points of the Cartesian grid of the channel's axes that are wanted (NaN if an axis is open).
            if isempty(chanAxes), n = 1; return; end
            an = arrayfun(@(a) a.Name, axes);
            vals = cell(1, numel(chanAxes));
            for i = 1:numel(chanAxes)
                ax = axes(an == chanAxes(i));
                if ax.isOpen(), n = NaN; return; end
                vals{i} = ax.Values;
            end
            sz = cellfun(@numel, vals);
            n = 0;
            for lin = 1:prod(sz)
                sub = cell(1, numel(sz)); [sub{:}] = ind2sub([sz 1], lin);
                v = struct(); ix = struct();
                for i = 1:numel(chanAxes)
                    v.(chanAxes(i)) = vals{i}(sub{i}); ix.(chanAxes(i)) = sub{i} - 1;
                end
                n = n + wantsFcn(v, ix);
            end
        end

        function b = bytesPerValue(info, precision)
            if isempty(info.ValueShape), b = NaN; return; end
            d = info.Dtype; if d == "", d = precision; end
            b = prod(info.ValueShape) * OceanReplay.Core.Kinds.bytesPerElement(d);
        end
    end
end
