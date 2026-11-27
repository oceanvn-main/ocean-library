classdef PostgresAdapter < OceanData.Adapters.IDbAdapter
    % POSTGRESADAPTER  PostgreSQL connection via pure Java JDBC.
    %
    %   Does NOT require the MATLAB Database Toolbox.
    %   Uses the PostgreSQL JDBC driver (postgresql-42.7.3.jar) directly
    %   through MATLAB's built-in Java integration.
    %
    %   SETUP (one-time, already done if jar is in +Adapters/ folder):
    %     The constructor automatically adds the jar to the Java classpath.
    %
    %   USAGE:
    %     cfg = OceanData.Adapters.DbConfig();
    %     db  = OceanData.Adapters.PostgresAdapter(cfg);
    %     db.connect();
    %     db.execute('CREATE TABLE IF NOT EXISTS t (x INT)');
    %     T = db.fetch('SELECT * FROM t');
    %     db.disconnect();
    %
    %   See also: OceanData.Adapters.DbConfig
    %             OceanData.Adapters.IDbAdapter

    properties (SetAccess = private)
        Config    OceanData.Adapters.DbConfig
        Conn_     % java.sql.Connection object
    end

    % Path to the JDBC jar alongside this file
    properties (Constant, Access = private)
        JAR_NAME = 'postgresql-42.7.3.jar'
    end

    methods
        function obj = PostgresAdapter(config)
            arguments
                config OceanData.Adapters.DbConfig
            end
            obj.Config = config;
            obj.ensureJar_();   % add JDBC jar to Java classpath if needed
        end

        % -------------------------------------------------------------- %
        function connect(obj)
            % CONNECT  Open PostgreSQL connection via JDBC.
            cfg = obj.Config;
            url  = sprintf('jdbc:postgresql://%s:%d/%s', ...
                char(cfg.Host), cfg.Port, char(cfg.Database));
            props = java.util.Properties();
            props.setProperty('user',     char(cfg.User));
            props.setProperty('password', char(cfg.Password));
            try
                driver     = org.postgresql.Driver();
                obj.Conn_  = driver.connect(url, props);
                if isempty(obj.Conn_) || obj.Conn_ == 0
                    error('Driver returned null — check URL and credentials.');
                end
                obj.Conn_.setAutoCommit(true);
            catch ME
                error('OceanData:Adapters:PostgresAdapter:ConnectFailed', ...
                    'Connection failed: %s', ME.message);
            end
            fprintf('[PostgresAdapter] Connected: %s\n', obj.Config.describe());
        end

        % -------------------------------------------------------------- %
        function disconnect(obj)
            if ~isempty(obj.Conn_)
                obj.Conn_.close();
                obj.Conn_ = [];
                fprintf('[PostgresAdapter] Disconnected.\n');
            end
        end

        % -------------------------------------------------------------- %
        function execute(obj, sql)
            % EXECUTE  Run DDL or DML — CREATE/DROP/INSERT/DELETE/UPDATE.
            obj.assertConnected_();
            stmt = obj.Conn_.createStatement();
            try
                stmt.execute(char(sql));
            catch ME
                stmt.close();
                error('OceanData:Adapters:PostgresAdapter:ExecuteFailed', ...
                    'SQL failed: %s\nSQL: %s', ME.message, sql);
            end
            stmt.close();
        end

        % -------------------------------------------------------------- %
        function rows = fetch(obj, sql)
            % FETCH  Run SELECT → struct array (one element per row).
            %   Empty struct([]) if the query returns no rows.
            obj.assertConnected_();
            stmt = obj.Conn_.createStatement();
            try
                rs = stmt.executeQuery(char(sql));
            catch ME
                stmt.close();
                error('OceanData:Adapters:PostgresAdapter:FetchFailed', ...
                    'SQL failed: %s\nSQL: %s', ME.message, sql);
            end
            rows = obj.resultSetToStructs_(rs);
            rs.close();
            stmt.close();
        end

        % -------------------------------------------------------------- %
        function tf = isConnected(obj)
            try
                tf = ~isempty(obj.Conn_) && ~obj.Conn_.isClosed();
            catch
                tf = false;
            end
        end

    end

    % ================================================================== %
    methods (Access = private)

        function ensureJar_(obj)
            % Add the PostgreSQL JDBC jar to the Java classpath if not already there.
            thisDir = fileparts(mfilename('fullpath'));
            jarPath = fullfile(thisDir, obj.JAR_NAME);
            if ~isfile(jarPath)
                error('OceanData:Adapters:PostgresAdapter:JarMissing', ...
                    ['PostgreSQL JDBC jar not found at:\n  %s\n' ...
                     'Download it with PowerShell:\n' ...
                     '  Invoke-WebRequest -Uri "https://jdbc.postgresql.org/download/%s" -OutFile "%s"'], ...
                    jarPath, obj.JAR_NAME, jarPath);
            end
            % Only add if not already on classpath (avoid duplicate warning)
            cp = javaclasspath('-dynamic');
            if ~any(strcmp(cp, jarPath))
                javaaddpath(jarPath);
            end
        end

        function assertConnected_(obj)
            if ~obj.isConnected()
                error('OceanData:Adapters:PostgresAdapter:NotConnected', ...
                    'Not connected. Call connect() first.');
            end
        end

        function rows = resultSetToStructs_(~, rs)
            % Convert java.sql.ResultSet → MATLAB struct array.
            % All values are cast to native MATLAB types at read time.
            meta     = rs.getMetaData();
            nCols    = meta.getColumnCount();
            colNames = cell(1, nCols);
            colTypes = cell(1, nCols);
            for c = 1:nCols
                colNames{c} = char(meta.getColumnName(c));
                colTypes{c} = upper(char(meta.getColumnTypeName(c)));
            end

            rows = struct([]);  % empty struct array
            while rs.next()
                s = struct();
                for c = 1:nCols
                    switch colTypes{c}
                        case {'INT4','INT8','INT2','INTEGER','BIGINT','SMALLINT'}
                            s.(colNames{c}) = double(rs.getInt(c));
                        case {'FLOAT4','FLOAT8','DOUBLE PRECISION','NUMERIC','DECIMAL'}
                            s.(colNames{c}) = double(rs.getDouble(c));
                        case {'BOOL','BOOLEAN'}
                            s.(colNames{c}) = logical(rs.getBoolean(c));
                        otherwise  % TEXT, VARCHAR, JSONB, etc.
                            val = rs.getString(c);
                            if rs.wasNull()
                                s.(colNames{c}) = '';
                            else
                                s.(colNames{c}) = char(val);
                            end
                    end
                end
                if isempty(rows)
                    rows = s;
                else
                    rows(end+1) = s; %#ok<AGROW>
                end
            end
        end

    end
end
