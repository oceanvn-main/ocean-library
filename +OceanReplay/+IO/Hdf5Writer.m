classdef Hdf5Writer < handle
    % HDF5WRITER  Streams a recording to an HDF5 file (Format spec Sec. 3).
    %   Chunks of about 8 KiB along the points axis; points are appended as the run produces them, so
    %   an interrupted run leaves a readable file with status "partial".
    %   MATLAB stores a value of logical shape S at point p as data(:,..,:,p),
    %   i.e. size [S P]; h5py then sees (P, reversed S) -- the G0-verified rule.
    %
    %   Example:
    %     w = OceanReplay.IO.Hdf5Writer("run.h5", 'Overwrite', true);
    %     w.begin(meta, axes);  w.define(info, "channels");
    %     w.append("x", x, w.axisIndex("step", 1000));  w.close();

    properties (SetAccess = private)
        File (1,1) string
        Closed (1,1) logical = false
    end

    properties (Access = private)
        Infos       % name -> ChannelInfo/ViewInfo (ValueShape/Dtype resolved at first write)
        Paths       % name -> group path ("/channels/x" | "/views/x")
        Counts      % name -> number of points written
        Created     % name -> datasets exist
        AxisValues  % axis -> values written so far (run order)
        AxisOpen    % axis -> open (appendable) axis
        Precision (1,1) string = "float32"
    end

    methods
        function obj = Hdf5Writer(file, opts)
            arguments
                file (1,1) string
                opts.Overwrite (1,1) logical = false
                opts.Precision (1,1) string {mustBeMember(opts.Precision, ["float32" "float64"])} = "float32"
            end
            if isfile(file)
                if ~opts.Overwrite
                    error('OceanReplay:Hdf5Writer:exists', '"%s" exists (use ''Overwrite'', true).', file);
                end
                delete(file);
            end
            obj.File = file; obj.Precision = opts.Precision;
            fid = H5F.create(char(file), 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');   % empty file for root attrs
            H5F.close(fid);
            M = @() containers.Map('KeyType', 'char', 'ValueType', 'any');
            obj.Infos = M(); obj.Paths = M(); obj.Counts = M(); obj.Created = M();
            obj.AxisValues = M(); obj.AxisOpen = M();
        end

        function begin(obj, meta, axes)
            % Root attributes (status "partial") and the axis datasets.
            arguments
                obj
                meta (1,1) struct
                axes OceanReplay.Core.Axis
            end
            f = char(obj.File);
            for a = axes(:).'
                p = ['/axes/' char(a.Name)];
                if a.isOpen()
                    h5create(f, p, Inf, 'ChunkSize', 64, 'Datatype', 'double');
                else
                    h5create(f, p, numel(a.Values), 'Datatype', 'double');
                    h5write(f, p, a.Values(:));
                end
                h5writeatt(f, p, 'symbol', a.Symbol);
                h5writeatt(f, p, 'units', a.Units);
                h5writeatt(f, p, 'description', a.Description);
                obj.AxisValues(char(a.Name)) = a.Values;
                obj.AxisOpen(char(a.Name)) = a.isOpen();
            end
            h5writeatt(f, '/', 'format', "ocean-replay");
            h5writeatt(f, '/', 'schemaVersion', "1.0");
            h5writeatt(f, '/', 'status', "partial");
            obj.setMeta(meta);
        end

        function setMeta(obj, meta)
            h5writeatt(char(obj.File), '/', 'meta', string(jsonencode(meta)));
        end

        function define(obj, info, where)
            % Register a channel ("channels") or view ("views"); datasets are created at the first append.
            arguments
                obj
                info (1,1) OceanReplay.Core.ChannelInfo
                where (1,1) string {mustBeMember(where, ["channels" "views"])}
            end
            key = char(info.Name);
            if isKey(obj.Infos, key), error('OceanReplay:Hdf5Writer:duplicate', 'Name "%s" defined twice.', key); end
            if info.Dtype == "", info.Dtype = obj.Precision; end
            if info.Kind == "image" && numel(info.DisplayRange) ~= 2
                error('OceanReplay:Hdf5Writer:displayRange', 'Image "%s" needs DisplayRange [lo hi] (how to map values to grey levels).', key);
            end
            for a = info.Axes
                if ~isKey(obj.AxisValues, char(a))
                    error('OceanReplay:Hdf5Writer:axis', '"%s" depends on axis "%s", which was not declared.', key, a);
                end
            end
            obj.Infos(key) = info; obj.Paths(key) = "/" + where + "/" + info.Name;
            obj.Counts(key) = 0; obj.Created(key) = false;
        end

        function idx = axisIndex(obj, axisName, value)
            % 0-based index of `value` on an axis; an open axis appends new values in run order.
            key = char(axisName);
            if ~isKey(obj.AxisValues, key), error('OceanReplay:Hdf5Writer:axis', 'Unknown axis "%s".', key); end
            vals = obj.AxisValues(key);
            k = find(vals == value, 1);
            if isempty(k)
                if ~obj.AxisOpen(key)
                    error('OceanReplay:Hdf5Writer:notOnAxis', 'Value %g is not on the closed axis "%s".', value, key);
                end
                vals(end+1) = value; k = numel(vals);
                h5write(char(obj.File), ['/axes/' key], value, k, 1);
                obj.AxisValues(key) = vals;
            end
            idx = k - 1;
        end

        function append(obj, name, value, idx)
            % Append one point: value (logical shape) at 0-based axis indices idx (channel axis order).
            key = char(name);
            info = obj.Infos(key);
            if numel(idx) ~= numel(info.Axes)
                error('OceanReplay:Hdf5Writer:index', '"%s" needs %d axis indices, got %d.', key, numel(info.Axes), numel(idx));
            end
            value = OceanReplay.Core.Kinds.normalise(info.Kind, value);
            msg = info.checkValue(value);
            if msg ~= "", error('OceanReplay:Hdf5Writer:value', '"%s": %s.', key, msg); end
            if ~obj.Created(key)
                if isempty(info.ValueShape)
                    info.ValueShape = OceanReplay.Core.ChannelInfo.shapeOf(info.Kind, value);
                    obj.Infos(key) = info;
                end
                obj.create(key, info);
            end
            f = char(obj.File); p = char(obj.Paths(key));
            vs = info.ValueShape; P = obj.Counts(key) + 1;
            if isempty(info.Axes) && P > 1
                error('OceanReplay:Hdf5Writer:static', '"%s" has no axes and was already recorded once.', key);
            end
            cls = OceanReplay.Core.Kinds.matlabClass(info.Dtype);
            if strcmp(cls, 'string'), v = string(value); else, v = cast(value, cls); end
            h5write(f, [p '/data'], reshape(v, [vs 1]), [ones(1, numel(vs)) P], [vs 1]);
            if ~isempty(info.Axes)
                h5write(f, [p '/points'], int32(idx(:)), [1 P], [numel(idx) 1]);
            end
            obj.Counts(key) = P;
        end

        function n = count(obj, name)
            n = obj.Counts(char(name));
        end

        function info = infoOf(obj, name)
            info = obj.Infos(char(name));
        end

        function close(obj)
            if obj.Closed, return; end
            h5writeatt(char(obj.File), '/', 'status', "complete");
            obj.Closed = true;
        end
    end

    methods (Access = private)
        function create(obj, key, info)
            f = char(obj.File); p = char(obj.Paths(key));
            vs = info.ValueShape;
            % Chunks of about 8 KiB (at most 64 points) along the points axis: one point per chunk for big values,
            % many for small ones (thousands of 4-byte scalar chunks would dominate the file).
            chunkP = max(1, min(64, floor(8192 / (prod(vs) * OceanReplay.Core.Kinds.bytesPerElement(info.Dtype)))));
            h5create(f, [p '/data'], [vs Inf], 'ChunkSize', [vs chunkP], ...
                'Datatype', OceanReplay.Core.Kinds.matlabClass(info.Dtype));
            A = numel(info.Axes);
            if A > 0
                h5create(f, [p '/points'], [A Inf], 'ChunkSize', [A 64], 'Datatype', 'int32');
            end
            s = info.attrs();
            fn = fieldnames(s);
            for i = 1:numel(fn), h5writeatt(f, p, fn{i}, s.(fn{i})); end
            obj.Created(key) = true;
        end
    end
end
