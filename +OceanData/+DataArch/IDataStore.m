classdef (Abstract) IDataStore < handle
    % IDATASTORE  Abstract base contract for all storage layers.
    %
    %   Defines the minimum interface that every storage layer must
    %   implement, regardless of whether it is a raw DataLake,
    %   a schema-enforced DataWarehouse, a FeatureStore, or an MDMStore.
    %
    %   DESIGN PRINCIPLE:
    %     "Key-based store" pattern — every record is addressed by a string
    %     key. Concrete implementations decide how keys map to physical
    %     storage (disk path, memory slot, entity ID, etc.).
    %
    %   STORAGE SEMANTICS:
    %     - put(key, data, meta) — write a record (overwrite if exists)
    %     - get(key)             — retrieve a record
    %     - list()               — enumerate all keys
    %     - exists(key)          — check without reading
    %     - delete(key)          — remove a record
    %
    %   USAGE (via concrete subclass):
    %     lake = OceanData.DataArch.DataLake('rootDir', './lake/');
    %     lake.put('raw/demo', X, struct('source', 'csv'));
    %     entry = lake.get('raw/demo');   % entry.data, entry.meta, entry.timestamp
    %
    %   See also: OceanData.DataArch.DataLake
    %             OceanData.DataArch.DataWarehouse
    %             OceanData.DataArch.FeatureStore

    % ================================================================== %
    methods (Abstract)

        % PUT  Write a record under the given key.
        %
        %   store.put(key, data, meta)
        %
        %   key  - char or string, unique record identifier
        %   data - any MATLAB value (matrix, struct, cell, ...)
        %   meta - struct of metadata (may be empty struct)
        put(obj, key, data, meta)

        % GET  Retrieve a record by key.
        %
        %   entry = store.get(key)
        %
        %   Returns a struct with fields:
        %     .data      - the stored value
        %     .meta      - the metadata struct passed to put()
        %     .timestamp - datetime of the last put()
        %     .key       - the key string
        %
        %   Throws OceanData:DataArch:<Class>:KeyNotFound if missing.
        entry = get(obj, key)

        % LIST  Return all keys currently stored.
        %
        %   keys = store.list()    % returns (1 x N) cell array of char
        keys = list(obj)

        % EXISTS  Return true if key is present, false otherwise.
        %
        %   tf = store.exists(key)
        tf = exists(obj, key)

        % DELETE  Remove a record by key.
        %
        %   store.delete(key)
        %
        %   Silently ignores missing keys (idempotent).
        delete(obj, key)

    end

    % ================================================================== %
    methods

        % DESCRIBE  Print a one-line summary of this store's contents.
        function describe(obj)
            keys = obj.list();
            fprintf('%s  [%d records]\n', class(obj), numel(keys));
            for i = 1:numel(keys)
                fprintf('  %-40s\n', keys{i});
            end
        end

    end

end
