classdef StreamingReader < OceanPipeline.Ingestion.IDataReader
    % STREAMINGREADER  STUB — Streaming data reader (not yet implemented).
    %
    %   Contract-only class. Calling read() throws NotImplemented.
    %   Exists to reserve the interface slot and allow downstream code
    %   to reference OceanPipeline.Ingestion.StreamingReader as a type
    %   without runtime failures in code paths that don't call read().
    %
    %   Future implementation should support:
    %     - Kafka / message queue consumption
    %     - TCP socket byte streams
    %     - WebSocket feeds
    %
    %   See also: OceanPipeline.Ingestion.IDataReader

    properties (SetAccess = private)
        Endpoint  (1,1) string = ""
    end

    methods
        function obj = StreamingReader(endpoint)
            arguments
                endpoint (1,1) string = ""
            end
            obj.Endpoint = endpoint;
        end

        % -------------------------------------------------------------- %
        function result = read(~)
            error('OceanPipeline:Ingestion:StreamingReader:NotImplemented', ...
                ['StreamingReader is a stub. ' ...
                 'Implement Kafka/TCP/WebSocket support in a future DA phase.']);
        end

        % -------------------------------------------------------------- %
        function src = getSource(obj)
            src = obj.Endpoint;
        end

        % -------------------------------------------------------------- %
        function tf = isReady(~)
            tf = false;   % always not ready — stub
        end
    end
end
