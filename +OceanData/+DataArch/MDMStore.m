classdef MDMStore < handle
    % MDMSTORE  Master Data Management store for entity golden records.
    %   Microscopic version of data warehouse
    %   Maintains one authoritative ("golden") record per entity ID.
    %   When conflicting records arrive from multiple source systems,
    %   MDMStore resolves them into a single source of truth.
    %
    %   DESIGN:
    %     Each entity has a unique string ID.
    %     Records are structs (any fields).
    %     Three merge strategies control conflict resolution:
    %       'latest'  — overwrite all fields with the incoming record
    %       'union'   — keep existing fields; only add new fields from incoming
    %       'max'     — for numeric fields, keep whichever value is larger
    %
    %   STORAGE FORMAT:
    %     <rootDir>/<entityId>.mat  — one .mat file per entity
    %
    %   USAGE:
    %     mdm = OceanData.DataArch.MDMStore('rootDir', './mdm/');
    %
    %     % Merge records from different sources
    %     mdm.merge('user_001', struct('name','Alice','age',30), 'latest');
    %     mdm.merge('user_001', struct('name','Alice','age',32,'email','a@x.com'), 'union');
    %
    %     % Retrieve the golden record
    %     rec = mdm.resolve('user_001');
    %     rec.age    % 30 (union kept original)
    %     rec.email  % 'a@x.com' (union added missing field)
    %
    %     mdm.list()    % all known entity IDs
    %
    %   See also: OceanData.DataArch.DataWarehouse
    %             OceanData.DataArch.IDataStore

    % ================================================================== %
    properties (SetAccess = private)
        rootDir  (1,1) string
    end

    % ================================================================== %
    methods

        function obj = MDMStore(opts)
            arguments
                opts.rootDir (1,1) string = "./mdm/"
            end
            obj.rootDir = opts.rootDir;
            if ~isfolder(obj.rootDir)
                mkdir(obj.rootDir);
            end
        end

        % -------------------------------------------------------------- %
        function merge(obj, entityId, record, strategy)
            % MERGE  Upsert a record for the given entity ID.
            %
            %   mdm.merge(id, record, strategy)
            %
            %   id       — char/string entity identifier
            %   record   — struct (any fields)
            %   strategy — 'latest' | 'union' | 'max'
            %
            %   'latest': incoming record completely replaces existing.
            %   'union':  existing fields are preserved; only NEW fields
            %             from the incoming record are added.
            %   'max':    for each numeric field, keep max(existing, incoming).
            %             For non-numeric, behave like 'union'.
            arguments
                obj
                entityId (1,1) string
                record   struct
                strategy (1,1) string {mustBeMember(strategy, {'latest','union','max'})} = 'latest'
            end
            existing = obj.loadRecord_(entityId);
            if isempty(fieldnames(existing))
                % No existing record — store directly
                golden = record;
            else
                golden = obj.applyStrategy_(existing, record, strategy);
            end
            golden.mdm_lastMerged_  = datetime('now');
            golden.mdm_strategy_    = char(strategy);
            obj.saveRecord_(entityId, golden);
        end

        % -------------------------------------------------------------- %
        function record = resolve(obj, entityId)
            % RESOLVE  Retrieve the golden record for an entity.
            %
            %   record = mdm.resolve('user_001')
            %
            %   Throws EntityNotFound if ID has never been merged.
            rec = obj.loadRecord_(entityId);
            if isempty(fieldnames(rec))
                error('OceanData:DataArch:MDMStore:EntityNotFound', ...
                    'No golden record found for entity "%s".', entityId);
            end
            record = rec;
        end

        % -------------------------------------------------------------- %
        function tf = exists(obj, entityId)
            % EXISTS  True if a golden record exists for this entity ID.
            tf = isfile(obj.entityPath_(entityId));
        end

        % -------------------------------------------------------------- %
        function remove(obj, entityId)
            % REMOVE  Delete the golden record for an entity.
            p = obj.entityPath_(entityId);
            if isfile(p), delete(p); end
        end

        % -------------------------------------------------------------- %
        function ids = list(obj)
            % LIST  Return all known entity IDs.
            files = dir(fullfile(obj.rootDir, '*.mat'));
            ids   = cell(1, numel(files));
            for i = 1:numel(files)
                [~, name, ~] = fileparts(files(i).name);
                ids{i} = name;
            end
        end

        % -------------------------------------------------------------- %
        function n = count(obj)
            % COUNT  Return number of known entities.
            n = numel(obj.list());
        end

        % -------------------------------------------------------------- %
        function describe(obj)
            % DESCRIBE  Print a summary of the MDM store.
            ids = obj.list();
            fprintf('MDMStore @ %s  (%d entities)\n', obj.rootDir, numel(ids));
            for i = 1:min(numel(ids), 10)
                fprintf('  %s\n', ids{i});
            end
            if numel(ids) > 10
                fprintf('  ... and %d more.\n', numel(ids)-10);
            end
        end

    end

    % ================================================================== %
    methods (Access = private)

        function golden = applyStrategy_(~, existing, incoming, strategy)
            switch strategy
                case 'latest'
                    % Replace entirely
                    golden = incoming;

                case 'union'
                    % Start with existing, add any fields not already present
                    golden     = existing;
                    newFields  = fieldnames(incoming);
                    for i = 1:numel(newFields)
                        f = newFields{i};
                        if ~isfield(golden, f)
                            golden.(f) = incoming.(f);
                        end
                    end

                case 'max'
                    % For numeric: keep max. For non-numeric: keep existing (union).
                    golden     = existing;
                    newFields  = fieldnames(incoming);
                    for i = 1:numel(newFields)
                        f = newFields{i};
                        if ~isfield(golden, f)
                            golden.(f) = incoming.(f);
                        elseif isnumeric(incoming.(f)) && isnumeric(golden.(f))
                            golden.(f) = max(golden.(f), incoming.(f));
                        end
                        % non-numeric: keep existing (union behavior)
                    end
            end
        end

        function rec = loadRecord_(obj, entityId)
            p = obj.entityPath_(entityId);
            if ~isfile(p)
                rec = struct();
                return;
            end
            loaded = load(p, 'record');
            rec    = loaded.record;
        end

        function saveRecord_(obj, entityId, record)
            p = obj.entityPath_(entityId); %#ok<NASGU>
            save(p, 'record');
        end

        function p = entityPath_(obj, entityId)
            % Sanitize entity ID for use as a filename (replace / with _)
            safe = strrep(char(entityId), '/', '_');
            safe = strrep(safe, '\', '_');
            p    = fullfile(obj.rootDir, [safe '.mat']);
        end

    end

end
