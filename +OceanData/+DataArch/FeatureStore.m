classdef FeatureStore < handle
    % FEATURESTORE  Versioned, named, disk-backed ML feature matrix store.
    %
    %   Sits at the output of the feature-engineering stage. Caches
    %   transformed feature matrices (e.g., Fourier maps, PCA projections)
    %   so they can be reused across experiments without recomputing.
    %
    %   DESIGN:
    %     Implements IDataStore. Each entry is stored as a .mat file:
    %       <rootDir>/<key>.mat  →  struct(data, meta, timestamp, key)
    %
    %     Key convention:  '<transform>/<dataset>/<version>'
    %       e.g.  'fourier/spam/v1', 'pca/boston/v3'
    %
    %     Keys use '/' separator → stored as OS sub-directories.
    %
    %   METADATA FIELDS (recommended, not enforced):
    %     meta.transform   — name of the feature extractor used
    %     meta.params      — struct of hyperparams (e.g., sigma, nComponents)
    %     meta.nSamples    — N
    %     meta.nFeatures   — D (output dimension)
    %     meta.source      — key of the DataLake/DataWarehouse entry it came from
    %
    %   USAGE:
    %     fs = OceanData.DataArch.FeatureStore('rootDir', './features/');
    %
    %     % Save Fourier-mapped features
    %     fs.put('fourier/spam/v1', Phi, struct('sigma', 3.5, 'nFeatures', 512));
    %
    %     % Load in another experiment
    %     entry = fs.get('fourier/spam/v1');
    %     Phi   = entry.data;   sigma = entry.meta.sigma;
    %
    %     % Compute-or-load pattern (avoids recompute)
    %     entry = fs.getOrCompute('fourier/spam/v1', @() ffm.transform(X), meta);
    %
    %   See also: OceanData.DataArch.DataLake
    %             OceanData.DataArch.IDataStore

    % ================================================================== %
    properties (SetAccess = private)
        rootDir  (1,1) string
    end

    % ================================================================== %
    methods

        function obj = FeatureStore(opts)
            arguments
                opts.rootDir (1,1) string = "./features/"
            end
            obj.rootDir = opts.rootDir;
            if ~isfolder(obj.rootDir)
                mkdir(obj.rootDir);
            end
        end

        % -------------------------------------------------------------- %
        function put(obj, key, data, meta)
            % PUT  Save a feature matrix under a versioned key.
            %
            %   fs.put('fourier/spam/v1', Phi, struct('sigma',3.5))
            %
            %   key  — forward-slash delimited version string
            %   data — the feature matrix (any numeric array)
            %   meta — optional struct with extraction parameters
            arguments
                obj
                key  (1,1) string
                data
                meta struct = struct()
            end
            entry.data      = data;
            entry.meta      = meta;
            entry.timestamp = datetime('now');
            entry.key       = char(key);

            p = obj.keyToPath_(key);
            obj.ensureDir_(fileparts(p));
            save(p, 'entry');
        end

        % -------------------------------------------------------------- %
        function entry = get(obj, key)
            % GET  Load the feature entry for a given key.
            %   Returns struct with fields: data, meta, timestamp, key.
            %   Throws KeyNotFound if key has never been stored.
            p = obj.keyToPath_(key);
            if ~isfile(p)
                error('OceanData:DataArch:FeatureStore:KeyNotFound', ...
                    'No feature entry found for key: "%s"', key);
            end
            loaded = load(p, 'entry');
            entry  = loaded.entry;
        end

        % -------------------------------------------------------------- %
        function entry = getOrCompute(obj, key, computeFn, meta)
            % GETORCOMPUTE  Return cached features, or compute and cache.
            %
            %   entry = fs.getOrCompute('fourier/spam/v1', @() ffm.transform(X), meta)
            %
            %   If 'key' already exists, loads and returns it immediately.
            %   Otherwise, calls computeFn(), stores result, then returns.
            arguments
                obj
                key       (1,1) string
                computeFn function_handle
                meta      struct = struct()
            end
            if obj.exists(key)
                entry = obj.get(key);
            else
                data = computeFn();
                obj.put(key, data, meta);
                entry = obj.get(key);
            end
        end

        % -------------------------------------------------------------- %
        function keys = list(obj)
            % LIST  Return all stored feature keys.
            files = dir(fullfile(obj.rootDir, '**', '*.mat'));
            keys  = cell(1, numel(files));
            for i = 1:numel(files)
                % Convert OS path back to forward-slash key
                rel  = strrep(files(i).folder, char(obj.rootDir), '');
                rel  = strrep(rel, '\', '/');
                rel  = strtrim(strrep(rel, '/', '', 1));  % strip leading /
                [~, name, ~] = fileparts(files(i).name);
                if isempty(rel)
                    keys{i} = name;
                else
                    keys{i} = sprintf('%s/%s', rel, name);
                end
            end
        end

        % -------------------------------------------------------------- %
        function tf = exists(obj, key)
            % EXISTS  True if a feature entry exists for this key.
            tf = isfile(obj.keyToPath_(key));
        end

        % -------------------------------------------------------------- %
        function delete(obj, key)
            % DELETE  Remove the feature entry for a key (idempotent).
            p = obj.keyToPath_(key);
            if isfile(p)
                delete(p);
            end
        end

        % -------------------------------------------------------------- %
        function describe(obj)
            % DESCRIBE  Print a summary of stored features.
            keys = obj.list();
            fprintf('FeatureStore @ %s  (%d entries)\n', obj.rootDir, numel(keys));
            for i = 1:numel(keys)
                try
                    e = obj.get(keys{i});
                    sz = size(e.data);
                    fprintf('  %-35s  [%s]  %s\n', keys{i}, ...
                        strjoin(string(sz), '×'), ...
                        char(e.timestamp, 'yyyy-MM-dd HH:mm'));
                catch
                    fprintf('  %-35s  (unreadable)\n', keys{i});
                end
            end
        end

    end

    % ================================================================== %
    methods (Access = private)

        function p = keyToPath_(obj, key)
            % Convert 'fourier/spam/v1' → '<rootDir>/fourier/spam/v1.mat'
            safe = strrep(char(key), '/', filesep);
            safe = strrep(safe, '\', filesep);
            p    = fullfile(obj.rootDir, [safe '.mat']);
        end

        function ensureDir_(~, d)
            if ~isempty(d) && ~isfolder(d)
                mkdir(d);
            end
        end

    end
end
