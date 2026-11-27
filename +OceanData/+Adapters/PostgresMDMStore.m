classdef PostgresMDMStore < handle
    % POSTGRESMDMSTORE  MDM golden record store backed by PostgreSQL JSONB.
    %
    %   Identical API to MDMStore.m but uses a single Postgres table:
    %
    %     CREATE TABLE mdm_entities (
    %       entity_id  TEXT PRIMARY KEY,
    %       record     JSONB,
    %       strategy   TEXT,
    %       updated_at TIMESTAMPTZ DEFAULT now()
    %     );
    %
    %   Struct ↔ JSON via jsonencode() / jsondecode().
    %
    %   CO-EXISTENCE:
    %     mdm = OceanData.DataArch.MDMStore(...)           % file-based
    %     mdm = OceanData.Adapters.PostgresMDMStore(cfg)   % Postgres
    %
    %   USAGE:
    %     cfg = OceanData.Adapters.DbConfig();
    %     mdm = OceanData.Adapters.PostgresMDMStore(cfg);
    %     mdm.connect();
    %     mdm.merge('user_001', struct('name','Alice','age',30), 'latest');
    %     rec = mdm.resolve('user_001');
    %     mdm.disconnect();
    %
    %   See also: OceanData.DataArch.MDMStore (file-based equivalent)

    properties (SetAccess = private)
        Adapter_  OceanData.Adapters.PostgresAdapter
        TableName_(1,1) string = "mdm_entities"
    end

    methods
        function obj = PostgresMDMStore(config, opts)
            arguments
                config OceanData.Adapters.DbConfig
                opts.TableName (1,1) string = "mdm_entities"
            end
            obj.Adapter_   = OceanData.Adapters.PostgresAdapter(config);
            obj.TableName_ = opts.TableName;
        end

        function connect(obj)
            obj.Adapter_.connect();
            obj.ensureTable_();
        end

        function disconnect(obj)
            obj.Adapter_.disconnect();
        end

        % -------------------------------------------------------------- %
        function merge(obj, entityId, record, strategy)
            % MERGE  Upsert an entity golden record with conflict resolution.
            %
            %   strategy — 'latest' | 'union' | 'max'
            arguments
                obj
                entityId (1,1) string
                record   struct
                strategy (1,1) string {mustBeMember(strategy, ...
                    {'latest','union','max'})} = 'latest'
            end
            existing = obj.loadRecord_(entityId);

            if isempty(fieldnames(existing))
                golden = record;
            else
                golden = obj.applyStrategy_(existing, record, strategy);
            end

            jsonStr  = strrep(jsonencode(golden), '''', '''''');
            safeId   = strrep(char(entityId), '''', '''''');
            safeStrat = strrep(char(strategy), '''', '''''');

            sql = sprintf([ ...
                'INSERT INTO %s (entity_id, record, strategy) VALUES ' ...
                '(E''%s'', E''%s''::jsonb, E''%s'') ' ...
                'ON CONFLICT (entity_id) DO UPDATE SET ' ...
                'record = EXCLUDED.record, strategy = EXCLUDED.strategy, ' ...
                'updated_at = now();'], ...
                obj.TableName_, safeId, jsonStr, safeStrat);
            obj.Adapter_.execute(sql);
        end

        % -------------------------------------------------------------- %
        function record = resolve(obj, entityId)
            % RESOLVE  Retrieve the golden record for an entity.
            rec = obj.loadRecord_(entityId);
            if isempty(fieldnames(rec))
                error('OceanData:DataArch:MDMStore:EntityNotFound', ...
                    'No golden record found for entity "%s".', entityId);
            end
            record = rec;
        end

        % -------------------------------------------------------------- %
        function tf = exists(obj, entityId)
            safeId = strrep(char(entityId), '''', '''''');
            sql    = sprintf( ...
                "SELECT COUNT(*) AS n FROM %s WHERE entity_id = E'%s';", ...
                obj.TableName_, safeId);
            T  = obj.Adapter_.fetch(sql);
            tf = T.n(1) > 0;
        end

        % -------------------------------------------------------------- %
        function remove(obj, entityId)
            safeId = strrep(char(entityId), '''', '''''');
            sql    = sprintf("DELETE FROM %s WHERE entity_id = E'%s';", ...
                obj.TableName_, safeId);
            obj.Adapter_.execute(sql);
        end

        % -------------------------------------------------------------- %
        function ids = list(obj)
            sql = sprintf('SELECT entity_id FROM %s ORDER BY updated_at DESC;', ...
                obj.TableName_);
            rows = obj.Adapter_.fetch(sql);
            if isempty(rows)
                ids = {};
            else
                ids = {rows.entity_id};
            end
        end

        % -------------------------------------------------------------- %
        function n = count(obj)
            sql  = sprintf('SELECT COUNT(*) AS n FROM %s;', obj.TableName_);
            rows = obj.Adapter_.fetch(sql);
            n    = rows(1).n;
        end

    end

    % ================================================================== %
    methods (Access = private)

        function ensureTable_(obj)
            sql = sprintf([ ...
                'CREATE TABLE IF NOT EXISTS %s (' ...
                'entity_id  TEXT PRIMARY KEY, ' ...
                'record     JSONB NOT NULL, ' ...
                'strategy   TEXT, ' ...
                'updated_at TIMESTAMPTZ DEFAULT now()' ...
                ');'], obj.TableName_);
            obj.Adapter_.execute(sql);
        end

        function rec = loadRecord_(obj, entityId)
            safeId = strrep(char(entityId), '''', '''''');
            sql    = sprintf( ...
                "SELECT record FROM %s WHERE entity_id = E'%s';", ...
                obj.TableName_, safeId);
            rows = obj.Adapter_.fetch(sql);
            if isempty(rows)
                rec = struct();
                return;
            end
            % rows(1).record is a plain char JSON string (JSONB → toString)
            rec = jsondecode(rows(1).record);
        end

        function golden = applyStrategy_(~, existing, incoming, strategy)
            switch strategy
                case 'latest'
                    golden = incoming;
                case 'union'
                    golden    = existing;
                    newFields = fieldnames(incoming);
                    for i = 1:numel(newFields)
                        f = newFields{i};
                        if ~isfield(golden, f)
                            golden.(f) = incoming.(f);
                        end
                    end
                case 'max'
                    golden    = existing;
                    newFields = fieldnames(incoming);
                    for i = 1:numel(newFields)
                        f = newFields{i};
                        if ~isfield(golden, f)
                            golden.(f) = incoming.(f);
                        elseif isnumeric(incoming.(f)) && isnumeric(golden.(f))
                            golden.(f) = max(golden.(f), incoming.(f));
                        end
                    end
            end
        end

    end
end
