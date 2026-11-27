function worker(host, port, sessionId)
%WORKER Run a newline-delimited JSON kernel over a loopback TCP connection.
%   This function is launched by the local HTTP companion. The transport is
%   private to localhost; imported source is still handled only by the text
%   scanner behind SymbolicNotebook.Kernel.
arguments
    host (1,1) string
    port (1,1) double {mustBeInteger,mustBePositive,mustBeLessThanOrEqual(port,65535)}
    sessionId (1,1) string
end
if ~any(host == ["127.0.0.1","localhost"])
    error('OceanTools:SymbolicNotebook:LoopbackOnly', 'The symbolic notebook worker only connects to loopback.');
end
client = tcpclient(host, port, 'Timeout', 30);
configureTerminator(client, "LF");
kernel = OceanTools.SymbolicNotebook.Kernel(sessionId);
cleanup = onCleanup(@() clearClient(client)); %#ok<NASGU>
while true
    try
        requestJson = string(readline(client));
    catch exception
        if any(strcmp(exception.identifier, {'MATLAB:networklib:tcpclient:ReadFailed', ...
                'instrument:interface:tcpip:readwrite:invalidConnection'}))
            return;
        end
        rethrow(exception);
    end
    if requestJson == "__shutdown__", return; end
    try
        responseJson = OceanTools.SymbolicNotebook.dispatchJson(kernel, requestJson);
    catch exception
        response = struct('protocolVersion',kernel.ProtocolVersion,'sessionId',kernel.SessionId, ...
            'jobId',"",'notebookRevision',0,'verdict',"invalid", ...
            'checks',struct('label',{},'status',{},'detail',{}), ...
            'notices',string(exception.message),'output',struct(), ...
            'engineVersion',kernel.EngineVersion,'errorCode',string(exception.identifier));
        responseJson = string(jsonencode(response));
    end
    writeline(client, responseJson);
end
end

function clearClient(client)
try
    flush(client);
catch
end
end
