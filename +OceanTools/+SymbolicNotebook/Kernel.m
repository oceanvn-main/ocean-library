classdef Kernel < handle
    %KERNEL Guarded command kernel for the browser symbolic notebook.
    %   The kernel is stateful, but its public boundary is plain structs so a
    %   loopback transport can JSON-encode requests and responses. Imported
    %   source is scanned as text and is never evaluated.

    properties (Constant)
        ProtocolVersion = "1.0.0"
        EngineVersion = "1.0.0"
    end

    properties (SetAccess = private)
        SessionId (1,1) string
        Registry (1,1) OceanMath.Symbolic.Registry
    end

    properties (Access = private)
        RecordCache containers.Map
        Compositions containers.Map
        CancelledJobs containers.Map
        CompositionCounter (1,1) double = 0
    end

    methods
        function obj = Kernel(sessionId)
            arguments
                sessionId (1,1) string = ""
            end
            if sessionId == ""
                [~, token] = fileparts(tempname);
                sessionId = "osn-" + erase(string(token), "_");
            end
            obj.SessionId = sessionId;
            obj.Registry = OceanMath.Symbolic.Registry();
            obj.RecordCache = containers.Map('KeyType','char','ValueType','any');
            obj.Compositions = containers.Map('KeyType','char','ValueType','any');
            obj.CancelledJobs = containers.Map('KeyType','char','ValueType','logical');
        end

        function response = execute(obj, request)
            arguments
                obj (1,1) OceanTools.SymbolicNotebook.Kernel
                request (1,1) struct
            end
            jobId = obj.textField(request, 'jobId', "");
            revision = obj.numericField(request, 'notebookRevision', 0);
            response = obj.baseResponse(jobId, revision);
            try
                obj.validateEnvelope(request);
                command = string(request.command);
                payload = request.payload;
                if command ~= "cancelJob" && isKey(obj.CancelledJobs, char(jobId))
                    response.verdict = "unsupported";
                    response.errorCode = "OceanTools:SymbolicNotebook:Cancelled";
                    response.notices = "The queued job was cancelled before execution.";
                    return;
                end
                switch command
                    case "scanText"
                        response = obj.scanText(payload, response);
                    case "acceptRecord"
                        response = obj.acceptRecord(payload, response);
                    case "compose"
                        response = obj.compose(payload, response);
                    case "transform"
                        response = obj.transform(payload, response);
                    case "inspect"
                        response = obj.inspect(payload, response);
                    case "expand"
                        response = obj.expand(payload, response);
                    case "cancelJob"
                        response = obj.cancelJob(payload, response);
                    otherwise
                        response.verdict = "unsupported";
                        response.errorCode = "OceanTools:SymbolicNotebook:UnsupportedCommand";
                        response.notices = "The kernel does not implement command " + command + ".";
                end
            catch exception
                response.verdict = obj.verdictForException(exception);
                response.errorCode = string(exception.identifier);
                response.notices = string(exception.message);
                response.output = struct();
            end
        end
    end

    methods (Access = private)
        function validateEnvelope(obj, request)
            required = {'protocolVersion','sessionId','jobId','notebookRevision','command','payload','sourceRefs'};
            if ~all(isfield(request, required))
                error('OceanTools:SymbolicNotebook:Envelope', ...
                    'Request requires protocolVersion, sessionId, jobId, notebookRevision, command, payload, and sourceRefs.');
            end
            if string(request.protocolVersion) ~= obj.ProtocolVersion
                error('OceanTools:SymbolicNotebook:ProtocolVersion', ...
                    'Unsupported protocol version %s.', string(request.protocolVersion));
            end
            if string(request.sessionId) ~= obj.SessionId
                error('OceanTools:SymbolicNotebook:Session', 'The request does not belong to this kernel session.');
            end
            if ~(isstruct(request.payload) && isscalar(request.payload)) || ...
                    ~(isstruct(request.sourceRefs) || isempty(request.sourceRefs))
                error('OceanTools:SymbolicNotebook:Envelope', 'payload and sourceRefs must be JSON objects or arrays.');
            end
            if strlength(string(request.jobId)) == 0 || strlength(string(request.jobId)) > 128
                error('OceanTools:SymbolicNotebook:Envelope', 'jobId must contain 1 through 128 characters.');
            end
        end

        function response = scanText(obj, payload, response)
            sourceText = obj.requiredText(payload, 'sourceText', 2e6);
            language = lower(obj.requiredText(payload, 'language', 16));
            filePath = obj.requiredText(payload, 'filePath', 4096);
            result = OceanMath.Symbolic.SourceExtractor.scanText(sourceText, language, filePath);
            fingerprint = obj.sha256(sourceText);
            records = result.functions;
            for k = 1:numel(records)
                recordKey = fingerprint + ":" + string(k);
                obj.RecordCache(char(recordKey)) = records(k);
                records(k).kernelRecordKey = recordKey;
                records(k).fingerprint = fingerprint;
                records(k).extractorVersion = "OceanMath.Symbolic.SourceExtractor/1.0";
                records(k).matchingStatus = "unresolved";
            end
            result.functions = records;
            result.fingerprint = fingerprint;
            result.extractorVersion = "OceanMath.Symbolic.SourceExtractor/1.0";
            response.output = result;
            response.verdict = "valid";
        end

        function response = acceptRecord(obj, payload, response)
            recordKey = obj.requiredText(payload, 'recordKey', 160);
            if ~isKey(obj.RecordCache, char(recordKey))
                error('OceanTools:SymbolicNotebook:UnknownRecord', ...
                    'acceptRecord requires a record key issued by scanText in this kernel session.');
            end
            alias = obj.textField(payload, 'alias', "");
            record = obj.RecordCache(char(recordKey));
            acceptedAlias = obj.Registry.accept(record, alias);
            response.output = struct('recordKey',recordKey,'recordId',string(record.id), ...
                'alias',acceptedAlias,'classification',string(record.classification), ...
                'provenance',obj.provenance(record));
            response.notices = obj.Registry.Notices;
            response.verdict = "valid";
        end

        function response = compose(obj, payload, response)
            if isfield(payload,'operator') && isfield(payload,'leftCompositionId') && isfield(payload,'rightCompositionId')
                left = obj.composition(payload.leftCompositionId);
                right = obj.composition(payload.rightCompositionId);
                composition = obj.Registry.combine(left, string(payload.operator), right);
            else
                alias = obj.requiredText(payload, 'alias', 40);
                if ~isfield(payload,'arguments') || ~isstruct(payload.arguments)
                    error('OceanTools:SymbolicNotebook:Payload', 'compose requires an arguments struct array.');
                end
                argumentsList = payload.arguments;
                required = {'expression','domain','shape'};
                if ~isempty(argumentsList) && ~all(isfield(argumentsList,required))
                    error('OceanTools:SymbolicNotebook:Payload', 'Each argument requires expression, domain, and shape.');
                end
                for k = 1:numel(argumentsList)
                    if isfield(argumentsList(k),'compositionId') && strlength(string(argumentsList(k).compositionId)) > 0
                        nested = obj.composition(argumentsList(k).compositionId);
                        argumentsList(k).expression = nested.expression;
                        argumentsList(k).nestedAst = nested.ast;
                    end
                end
                composition = obj.Registry.compose(alias, argumentsList);
            end
            compositionId = obj.saveComposition(composition);
            response.output = struct('compositionId',compositionId,'composition',composition);
            response.checks = composition.checks;
            response.verdict = obj.verdictFromChecks(composition.checks);
        end

        function response = transform(obj, payload, response)
            composition = obj.composition(obj.requiredText(payload,'compositionId',128));
            operation = obj.requiredText(payload,'operation',32);
            variable = obj.textField(payload,'variable',"x");
            if ~any(operation == ["simplify","differentiate","inspect"])
                response.verdict = "unsupported";
                response.errorCode = "OceanTools:SymbolicNotebook:UnsupportedTransform";
                response.notices = "Registry transformations support simplify, differentiate, and inspect.";
                return;
            end
            report = obj.Registry.manipulate(composition, operation, variable);
            response.output = report;
            response.checks = report.checks;
            if report.status == "unsupported"
                response.verdict = "unsupported";
            elseif report.status == "conditional"
                response.verdict = "unknown";
            else
                response.verdict = obj.verdictFromChecks(report.checks);
            end
        end

        function response = inspect(obj, payload, response)
            expression = obj.requiredText(payload,'expression',2048);
            workspace = obj.workspaceFromPayload(payload);
            report = workspace.inspect(expression);
            response.output = obj.publicReport(report);
            response.checks = report.checks;
            response.verdict = obj.verdictFromChecks(report.checks);
        end

        function response = expand(obj, payload, response)
            if isfield(payload,'compositionId')
                composition = obj.composition(payload.compositionId);
                text = obj.expandAst(composition.ast);
            else
                alias = obj.requiredText(payload,'alias',40);
                if ~isfield(payload,'arguments')
                    error('OceanTools:SymbolicNotebook:Payload', 'expand requires arguments.');
                end
                text = obj.Registry.expand(alias, string(payload.arguments));
            end
            response.output = struct('expanded',text);
            response.verdict = "valid";
        end

        function response = cancelJob(obj, payload, response)
            target = obj.requiredText(payload,'targetJobId',128);
            obj.CancelledJobs(char(target)) = true;
            response.output = struct('targetJobId',target,'cancelled',true);
            response.verdict = "valid";
            response.notices = "Cancellation is cooperative: queued work is skipped; an active MATLAB operation runs to its guarded limit.";
        end

        function workspace = workspaceFromPayload(obj, payload) %#ok<INUSL>
            workspace = OceanMath.Symbolic.Workspace();
            if ~isfield(payload,'symbols') || isempty(payload.symbols), return; end
            symbols = payload.symbols;
            if ~isstruct(symbols) || ~all(isfield(symbols,{'name','domain'}))
                error('OceanTools:SymbolicNotebook:Payload', 'symbols must contain name and domain fields.');
            end
            for k = 1:numel(symbols)
                workspace.addSymbol(string(symbols(k).name), string(symbols(k).domain));
            end
        end

        function text = expandAst(obj, ast)
            if string(ast.kind) == "binary"
                text = "(" + obj.expandAst(ast.left) + ") " + string(ast.operator) + " (" + obj.expandAst(ast.right) + ")";
                return;
            end
            if string(ast.kind) ~= "call"
                error('OceanTools:SymbolicNotebook:AST', 'Unknown composition node kind.');
            end
            args = strings(1,numel(ast.arguments));
            for k = 1:numel(ast.arguments)
                if isfield(ast.arguments(k),'nestedAst')
                    args(k) = obj.expandAst(ast.arguments(k).nestedAst);
                else
                    args(k) = string(ast.arguments(k).expression);
                end
            end
            text = obj.Registry.expand(string(ast.operator), args);
        end

        function composition = composition(obj, compositionId)
            key = char(string(compositionId));
            if ~isKey(obj.Compositions,key)
                error('OceanTools:SymbolicNotebook:UnknownComposition', 'Unknown composition ID %s.', string(compositionId));
            end
            composition = obj.Compositions(key);
        end

        function id = saveComposition(obj, composition)
            obj.CompositionCounter = obj.CompositionCounter + 1;
            id = "composition-" + string(obj.CompositionCounter);
            obj.Compositions(char(id)) = composition;
        end

        function response = baseResponse(obj, jobId, revision)
            response = struct('protocolVersion',obj.ProtocolVersion,'sessionId',obj.SessionId, ...
                'jobId',jobId,'notebookRevision',revision,'verdict',"unknown", ...
                'checks',struct('label',{},'status',{},'detail',{}),'notices',strings(0,1), ...
                'output',struct(),'engineVersion',obj.EngineVersion,'errorCode',"");
        end
    end

    methods (Static, Access = private)
        function value = requiredText(data, field, maximum)
            if ~isfield(data,field)
                error('OceanTools:SymbolicNotebook:Payload', 'Missing payload field %s.', field);
            end
            value = string(data.(field));
            if ~isscalar(value) || strlength(value) == 0 || strlength(value) > maximum
                error('OceanTools:SymbolicNotebook:Payload', '%s must contain 1 through %d characters.', field, maximum);
            end
        end

        function value = textField(data, field, fallback)
            if isfield(data,field), value = string(data.(field)); else, value = fallback; end
            if ~isscalar(value), value = fallback; end
        end

        function value = numericField(data, field, fallback)
            if isfield(data,field) && isnumeric(data.(field)) && isscalar(data.(field)) && isfinite(data.(field))
                value = double(data.(field));
            else
                value = fallback;
            end
        end

        function digest = sha256(text)
            engine = javaMethod('getInstance','java.security.MessageDigest','SHA-256');
            engine.update(unicode2native(char(text),'UTF-8'));
            bytes = typecast(engine.digest(),'uint8');
            digest = lower(string(reshape(dec2hex(bytes,2).',1,[])));
        end

        function provenance = provenance(record)
            provenance = struct('file',string(record.file),'sourceLines',record.sourceLines, ...
                'expressionSourceLine',record.expressionSourceLine,'expressionOrigin',string(record.expressionOrigin));
        end

        function verdict = verdictFromChecks(checks)
            if isempty(checks), verdict = "valid"; return; end
            statuses = string({checks.status});
            if any(statuses == "invalid"), verdict = "invalid";
            elseif any(statuses == "unknown"), verdict = "unknown";
            else, verdict = "valid";
            end
        end

        function verdict = verdictForException(exception)
            id = string(exception.identifier);
            if contains(id,["OpaqueExpansion","UnsupportedExpansion","Unsupported"])
                verdict = "unsupported";
            elseif contains(id,["ContractMismatch","ShapeMismatch","Domain","Arity","Payload","Envelope","Session","Protocol","Unknown"])
                verdict = "invalid";
            else
                verdict = "invalid";
            end
        end

        function report = publicReport(report)
            removable = intersect(fieldnames(report), {'symbolicExpanded','symbolicResult'});
            if ~isempty(removable), report = rmfield(report,removable); end
        end
    end
end
