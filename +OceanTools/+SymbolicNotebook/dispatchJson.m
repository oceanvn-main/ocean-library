function responseJson = dispatchJson(kernel, requestJson)
%DISPATCHJSON Decode one fixed-protocol request and encode its response.
arguments
    kernel (1,1) OceanTools.SymbolicNotebook.Kernel
    requestJson (1,1) string
end
if strlength(requestJson) > 4e6
    error('OceanTools:SymbolicNotebook:RequestLimit', ...
        'A symbolic notebook request is limited to 4,000,000 characters.');
end
request = jsondecode(requestJson);
responseJson = string(jsonencode(kernel.execute(request)));
end
