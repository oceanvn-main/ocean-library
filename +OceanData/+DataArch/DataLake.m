classdef DataLake < OceanData.DataArch.IDataStore
    % DATALAKE  Schema-on-read, raw partitioned data storage.
    %
    %   The Data Lake is the first permanent storage layer after the ODS
    %   (Operational Data Store / landing zone). It stores everything in
    %   its native form — no schema is enforced on write. The caller
    %   interprets the schema when reading.
    %
    %   WHEN TO USE:
    %     - Store all raw incoming data cheaply and durably
    %     - Retain full history (no expiry policy)
    %     - Later pipelines can re-read and interpret with new logic
    %
    %   WHEN NOT TO USE:
    %     - When you need fast aggregated queries → use DataWarehouse
    %     - When you need ML-ready feature matrices → use FeatureStore
    %     - When you need entity deduplication    → use MDMStore
    %
    %   STORAGE LAYOUT (disk):
    %     <rootDir>/
    %       <partition_key_1>.mat   → struct(data, meta, timestamp, key)
    %       raw/
    %         demographics.mat
    %
    %   Keys use '/' as a path separator → OS subdirectories.
    %   E.g. key='raw/demographics' → <rootDir>/raw/demographics.mat
    %
    %   USAGE:
    %     lake = OceanData.DataArch.DataLake('rootDir', './lake/');
    %
    %     lake.put('raw/demographics', X, struct('source','census'));
    %     entry = lake.get('raw/demographics');
    %     entry.data       % the matrix X
    %     entry.meta       % struct('source','census')
    %     entry.timestamp  % datetime of put()
    %
    %     lake.list()      % {'raw/demographics', ...}
    %     lake.exists('raw/demographics')  % true
    %     lake.delete('raw/demographics')
    %
    %   See also: OceanData.DataArch.IDataStore
    %             OceanData.DataArch.DataWarehouse
    %             OceanData.DataArch.FeatureStore

    % ------------------------------------------------------------------ %
    properties (SetAccess = private)
        RootDir (1,1) string   % Absolute path to the lake root directory
    end

    % ================================================================== %
    methods

        function obj = DataLake(opts)
            % DATALAKE  Construct a DataLake backed by a directory on disk.
            %
            %   lake = DataLake('rootDir', './lake/')
            %
            % Name-Value:
            %   rootDir (string): path to root directory (created if missing)
            arguments
                opts.rootDir (1,1) string = "./lake/"
            end
            obj.RootDir = fullfile(opts.rootDir);
            if ~isfolder(obj.RootDir)
                mkdir(obj.RootDir);
            end
        end

        % -------------------------------------------------------------- %
        function put(obj, key, data, meta, opts)
            % PUT  Write a record to the lake (no schema enforced).
            %
            %   lake.put('raw/demo', X, struct('source', 'csv'))
            %
            %   By default (overwrite=false) the write is SKIPPED if the key
            %   already exists — preventing redundant disk I/O on every re-run.
            %   To explicitly overwrite: lake.put(key, data, meta, overwrite=true)
            arguments
                obj
                key  (1,1) string
                data
                meta (1,1) struct  = struct()
                opts.overwrite (1,1) logical = false
                opts.asParquet (1,1) logical = false
            end

            % Write-once guard: skip if key present and overwrite not requested
            if ~opts.overwrite && obj.exists(key)
                fprintf('[DataLake] SKIP write (already exists): %s\n', key);
                return;
            end

            path_ = obj.keyToPath_(key);
            obj.ensureDir_(fileparts(path_));

            entry.data      = data;      %#ok<STRNU>
            entry.meta      = meta;
            entry.timestamp = datetime('now');
            entry.key       = key;

            save(path_, 'entry', '-v7.3');
            
            % Save a Parquet version if requested
            if opts.asParquet
                if istable(data)
                    pPath = obj.keyToParquetPath_(key);
                    obj.ensureDir_(fileparts(pPath));
                    parquetwrite(pPath, data);
                else
                    warning('OceanData:DataArch:DataLake:NotATable', ...
                        'Cannot write Parquet: Data is a %s, but parquetwrite requires a table.', class(data));
                end
            end
        end

        % -------------------------------------------------------------- %
        function entry = get(obj, key)
            % GET  Retrieve a record by key.
            %
            %   entry = lake.get('raw/demographics')
            %   entry.data, entry.meta, entry.timestamp, entry.key
            %
            %   Throws OceanData:DataArch:DataLake:KeyNotFound if missing.
            arguments
                obj
                key (1,1) string
            end

            path_ = obj.keyToPath_(key);
            if ~isfile(path_)
                error('OceanData:DataArch:DataLake:KeyNotFound', ...
                    'Key not found in DataLake: "%s"\nLooked at: %s', ...
                    key, path_);
            end
            loaded = load(path_, 'entry');
            entry  = loaded.entry;
        end

        % -------------------------------------------------------------- %
        function keys = list(obj)
            % LIST  Return all partition keys in this lake.
            %
            %   keys = lake.list()   % (1 x N) cell array of strings
            files = dir(fullfile(obj.RootDir, '**', '*.mat'));
            keys  = cell(1, numel(files));
            % Normalise root: char, no trailing separator
            root = char(obj.RootDir);
            if ~isempty(root) && (root(end) == filesep || root(end) == '/')
                root = root(1:end-1);
            end
            prefixLen = numel(root) + 1;   % +1 for separator after root
            for i = 1:numel(files)
                fullp    = fullfile(files(i).folder, files(i).name);
                % Slice off root prefix (length-based, avoids string/char mismatch)
                relative = fullp(prefixLen+1 : end);
                % Normalise OS separators to forward-slash
                relative = strrep(relative, '\', '/');
                % Strip .mat extension
                keys{i}  = relative(1:end-4);
            end
        end

        % -------------------------------------------------------------- %
        function tf = exists(obj, key)
            % EXISTS  True if the key has a record in this lake.
            arguments
                obj
                key (1,1) string
            end
            tf = isfile(obj.keyToPath_(key));
        end

        % -------------------------------------------------------------- %
        function delete(obj, key)
            % DELETE  Remove a record by key (silently ignores missing).
            arguments
                obj
                key (1,1) string
            end
            path_ = obj.keyToPath_(key);
            if isfile(path_)
                delete(path_);
            end
        end

    end

    % ================================================================== %
    methods (Access = private)

        function path = keyToPath_(obj, key)
            % KEYTOPATH_  Convert a forward-slash key to an OS file path.
            %   'raw/demographics' → '<rootDir>/raw/demographics.mat'
            parts = split(key, '/');
            path  = fullfile(obj.RootDir, parts{:});
            path  = path + ".mat";
        end

        function path = keyToParquetPath_(obj, key)
            % KEYTOPARQUETPATH_  Convert key to a Parquet file path.
            parts = split(key, '/');
            path  = fullfile(obj.RootDir, parts{:});
            path  = path + ".parquet";
        end

        function ensureDir_(~, dirPath)
            % ENSUREDIR_  Create directory if it does not exist.
            if ~isempty(dirPath) && ~isfolder(dirPath)
                mkdir(dirPath);
            end
        end

    end

end
