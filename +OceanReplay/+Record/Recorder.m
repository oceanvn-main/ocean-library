classdef Recorder < handle
    % RECORDER  Records a domain's run into a Replay file, following a Spec.
    %   At every point the domain reports (point(coords, state)) it:
    %     1. resolves which channels and views the spec wants here (on axis VALUES);
    %     2. evaluates ONLY those channels' lazy providers and appends them;
    %     3. computes the wanted views inside RngGuard, with their own stream;
    %     4. returns to the loop. Nothing wanted -> one lookup per channel.
    %   A channel over fewer axes than the loop (e.g. per-epoch inside an
    %   epoch x step loop) is stored once per distinct point of its own axes;
    %   a channel with NO axes is stored the first time the state provides it
    %   (e.g. a final result reported with recorder.point(struct(), ...)).
    %   Before running, the spec is validated against the domain's catalog and
    %   the size is estimated; above Spec.SizeCapBytes the run is refused.
    %
    %   Example:
    %     rec = OceanReplay.Record.Recorder(recordable, spec, "run.h5", ...
    %           'Title', "k-means on X", 'Seeds', struct('data', 3));
    %     out = rec.record();
    %     r = OceanReplay.Core.Recording("run.h5");

    properties (SetAccess = private)
        Spec
        Recordable
        File (1,1) string
        Writer
        Channels            % catalog ChannelInfo array
        ViewCatalog         % catalog ViewInfo array
        ViewInfos           % one ViewInfo per spec view (Name, MethodId, Params set)
        AxesInfo
        ViewStream          % RandStream for all views (seed ViewSeed)
        EstimatedBytes (1,1) double = NaN
        Report = struct()
    end

    properties (Access = private)
        Opts
        Done                % name -> containers.Map of stored point keys
        ChannelNames (1,:) string = strings(1, 0)
    end

    methods
        function obj = Recorder(recordable, spec, file, opts)
            arguments
                recordable (1,1) OceanReplay.Record.IRecordable
                spec (1,1) OceanReplay.Core.Spec
                file (1,1) string
                opts.Overwrite (1,1) logical = false
                opts.ViewSeed (1,1) double = 1007
                opts.Title (1,1) string = ""
                opts.RunId (1,1) string = ""
                opts.Seeds (1,1) struct = struct()
            end
            obj.Recordable = recordable; obj.Spec = spec; obj.File = file; obj.Opts = opts;
            obj.AxesInfo = recordable.axesInfo();
            [obj.Channels, obj.ViewCatalog] = recordable.catalog();
            if isempty(obj.Channels), obj.Channels = OceanReplay.Core.ChannelInfo.empty; end
            if isempty(obj.ViewCatalog), obj.ViewCatalog = OceanReplay.Core.ViewInfo.empty; end
            spec.validate(obj.Channels, obj.ViewCatalog, obj.AxesInfo);
            [obj.EstimatedBytes, perName] = spec.estimateBytes(obj.Channels, obj.ViewCatalog, obj.AxesInfo);
            obj.Report.Estimate = perName;
            if isfinite(spec.SizeCapBytes)
                if isnan(obj.EstimatedBytes)
                    warning('OceanReplay:Recorder:unknownSize', ...
                        'Size cannot be estimated in advance (open axis or run-time shape); the %g-byte cap is not enforced.', spec.SizeCapBytes);
                elseif obj.EstimatedBytes > spec.SizeCapBytes
                    error('OceanReplay:Recorder:tooLarge', 'Estimated size %.3g bytes exceeds the spec''s cap of %.3g bytes.', ...
                        obj.EstimatedBytes, spec.SizeCapBytes);
                end
            end
        end

        function out = record(obj)
            % Open the file, run the domain with this recorder, close. Returns the run's output.
            meta = obj.meta();
            obj.Writer = OceanReplay.IO.Hdf5Writer(obj.File, 'Overwrite', obj.Opts.Overwrite, 'Precision', obj.Spec.Precision);
            obj.Writer.begin(meta, obj.AxesInfo);
            cn = arrayfun(@(c) c.Name, obj.Channels);
            obj.ChannelNames = obj.Spec.channelNames();
            obj.Done = containers.Map('KeyType', 'char', 'ValueType', 'any');
            for n = obj.ChannelNames
                info = obj.Channels(cn == n);
                info.Level = obj.Spec.level(n);
                obj.Writer.define(info, "channels");
                obj.Done(char(n)) = containers.Map('KeyType', 'char', 'ValueType', 'logical');
            end
            vn = arrayfun(@(v) v.Name, obj.ViewCatalog);
            obj.ViewInfos = OceanReplay.Core.ViewInfo.empty;
            for j = 1:numel(obj.Spec.Views)
                sv = obj.Spec.Views(j);
                v = obj.ViewCatalog(vn == sv.Method);
                v.Name = sv.Name; v.MethodId = sv.Method; v.Params = sv.Params;
                obj.ViewInfos(j) = v;
                obj.Writer.define(v, "views");
                obj.Done(char(sv.Name)) = containers.Map('KeyType', 'char', 'ValueType', 'logical');
            end
            obj.ViewStream = RandStream('mt19937ar', 'Seed', obj.Opts.ViewSeed);
            out = obj.Recordable.run(obj);
            obj.finish(meta);
        end

        function point(obj, coords, state)
            % Called by the domain at each point of its loop (see IRecordable).
            arguments
                obj
                coords (1,1) struct
                state (1,1) struct
            end
            idx = struct();
            for f = string(fieldnames(coords)).'
                idx.(f) = obj.Writer.axisIndex(f, coords.(f));
            end
            for n = obj.ChannelNames
                info = obj.Writer.infoOf(n);
                [ok, key, ix] = obj.pointOf(info, coords, idx, n);
                if ~ok || ~obj.Spec.wants(n, coords, idx), continue; end
                if ~isfield(state, n)
                    if isempty(info.Axes), continue; end                % no-axis channel: stored when first provided
                    error('OceanReplay:Recorder:missingState', ...
                        'The spec wants "%s" at %s, but the domain''s state has no provider for it.', n, jsonencode(coords));
                end
                obj.Writer.append(n, OceanReplay.Record.Recorder.evaluate(state.(n)), ix);
                obj.markDone(n, key);
            end
            for j = 1:numel(obj.ViewInfos)
                v = obj.ViewInfos(j); n = v.Name;
                [ok, key, ix] = obj.pointOf(v, coords, idx, n);
                if ~ok || ~obj.Spec.viewWants(j, coords, idx), continue; end
                compute = v.Compute; params = v.Params; stream = obj.ViewStream;
                value = OceanReplay.Record.RngGuard.run(@() compute(state, params, stream));
                obj.Writer.append(n, value, ix);
                obj.markDone(n, key);
            end
        end
    end

    methods (Access = private)
        function markDone(obj, name, key)
            done = obj.Done(char(name));      % containers.Map is a handle: this updates the stored map
            done(key) = true;                 %#ok<NASGU>
        end

        function [ok, key, ix] = pointOf(obj, info, coords, idx, n)
            % The point of this channel's own axes; ok = false if a coordinate is missing or already stored.
            A = numel(info.Axes); ix = zeros(1, A); ok = true; key = '';
            for a = 1:A
                if ~isfield(coords, info.Axes(a)), ok = false; return; end
                ix(a) = idx.(info.Axes(a));
            end
            key = sprintf('%d,', ix);
            if isKey(obj.Done(char(n)), key), ok = false; end
        end

        function m = meta(obj)
            d = obj.Recordable.describe();
            created = datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z''');
            runId = obj.Opts.RunId;
            if runId == "", stamp = created; stamp.Format = 'yyyyMMdd-HHmmss'; runId = "run-" + string(stamp); end
            title = obj.Opts.Title; if title == "", title = string(class(obj.Recordable)); end
            seeds = obj.Opts.Seeds; seeds.views = obj.Opts.ViewSeed;
            m = struct('title', title, 'created', string(created), 'runId', runId, ...
                'source', struct('language', "matlab", 'library', "OceanReplay (MATLAB)", 'matlab', string(version), ...
                                 'oceanCommit', OceanReplay.Record.Recorder.commit()), ...
                'model', d.model, 'seeds', seeds, 'references', {d.references}, 'notation', d.notation, ...
                'spec', obj.Spec.toStruct());
            if isfield(d, 'layout'), m.layout = string(d.layout); end    % optional replay-layout hint
        end

        function finish(obj, meta)
            % Report requested "list" points that the run never visited (open axes), then close.
            notVisited = strings(0, 1);
            for n = obj.ChannelNames
                info = obj.Writer.infoOf(n);
                for g = obj.Spec.Groups
                    if ~ismember(n, g.Channels), continue; end
                    for a = string(fieldnames(g.At)).'
                        k = find(info.Axes == a, 1);
                        if isempty(k), continue; end
                        visited = obj.visitedValues(a);
                        for v = g.At.(a).missing(visited)
                            notVisited(end+1, 1) = sprintf('%s@%s=%g', n, a, v); %#ok<AGROW>
                        end
                    end
                end
            end
            obj.Report.NotVisited = unique(notVisited);
            for n = obj.ChannelNames
                if isempty(obj.Writer.infoOf(n).Axes) && obj.Writer.count(n) == 0
                    warning('OceanReplay:Recorder:neverProvided', 'The spec wants "%s" (no axes), but the run never provided it.', n);
                end
            end
            if ~isempty(notVisited)
                meta.notVisited = cellstr(obj.Report.NotVisited);
                obj.Writer.setMeta(meta);
                warning('OceanReplay:Recorder:notVisited', 'Requested points never visited by the run: %s', ...
                    strjoin(obj.Report.NotVisited, ', '));
            end
            obj.Writer.close();
        end

        function v = visitedValues(obj, axisName)
            v = reshape(h5read(char(obj.File), char("/axes/" + axisName)), 1, []);
        end
    end

    methods (Static)
        function out = run(recordable, spec, file, varargin)
            % One-call form: out = Recorder.run(recordable, spec, file, name-value options...).
            rec = OceanReplay.Record.Recorder(recordable, spec, file, varargin{:});
            out = rec.record();
        end
    end

    methods (Static, Access = private)
        function v = evaluate(p)
            if isa(p, 'function_handle'), v = p(); else, v = p; end
        end

        function c = commit()
            % Git commit of the library, or "uncommitted" (no commits yet / git unavailable).
            c = "uncommitted";
            try
                root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
                [st, out] = system(sprintf('git -C "%s" rev-parse HEAD', root));
                if st == 0, c = string(strtrim(out)); end
            catch
            end
        end
    end
end
