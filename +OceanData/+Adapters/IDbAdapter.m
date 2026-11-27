classdef (Abstract) IDbAdapter < handle
    % IDBADAPTER  Abstract interface for database connection adapters.
    %
    %   Defines the minimal contract any relational DB adapter must satisfy
    %   so that higher-level components (PostgresWarehouse, etc.) can work
    %   with any backend without modification.
    %
    %   PATTERN:
    %     Adapter pattern — wraps a vendor-specific MATLAB database
    %     connection behind a uniform Ocean Library interface.
    %
    %   See also: OceanData.Adapters.PostgresAdapter
    %             OceanData.Adapters.DbConfig

    methods (Abstract)
        % CONNECT  Open the database connection.
        connect(obj)

        % DISCONNECT  Close and release the connection.
        disconnect(obj)

        % EXECUTE  Run a DDL or DML statement with no result set.
        %   sql — char or string SQL statement
        execute(obj, sql)

        % FETCH  Run a SELECT and return results as a MATLAB table.
        %   sql — char or string SQL SELECT statement
        %   T   — MATLAB table with column names from the result set
        T = fetch(obj, sql)

        % ISCONNECTED  Return true if the connection is currently open.
        tf = isConnected(obj)
    end
end
