classdef SymbolicNotebookKernelTest < OceanTest.BaseTestCase
    methods (Test)
        function scansMatlabAndPythonWithoutExecutingSource(testCase)
            kernel = OceanTools.SymbolicNotebook.Kernel("test-session");
            matlabText = join(["function y = entropyTerm(p)", ...
                "error('this imported source must never run')", "y = -p * log(p);", "end"], newline);
            pythonText = join(["def affine(x, w, b):", ...
                "    raise RuntimeError('this imported source must never run')", "    return x * w + b"], newline);

            matlabResponse = kernel.execute(testCase.request(kernel,"scan-m","scanText", ...
                struct('sourceText',matlabText,'language',"matlab",'filePath',"entropyTerm.m")));
            pythonResponse = kernel.execute(testCase.request(kernel,"scan-p","scanText", ...
                struct('sourceText',pythonText,'language',"python",'filePath',"affine.py")));

            testCase.verifyEqual(matlabResponse.verdict,"valid");
            testCase.verifyEqual(string(matlabResponse.output.functions(1).classification),"transparent");
            testCase.verifyEqual(string(pythonResponse.output.functions(1).classification),"transparent");
            testCase.verifyEqual(strlength(string(matlabResponse.output.fingerprint)),64);
            testCase.verifyNotEmpty(matlabResponse.output.functions(1).kernelRecordKey);
        end

        function acceptsKernelIssuedRecordsAndRenamesCollisions(testCase)
            kernel = OceanTools.SymbolicNotebook.Kernel("test-session");
            first = testCase.scan(kernel,"function y = score(x)"+newline+"y = x + 1;"+newline+"end", ...
                "matlab","first.m","one");
            second = testCase.scan(kernel,"def score(x):"+newline+"    return x * 2", ...
                "python","second.py","two");
            accepted1 = kernel.execute(testCase.request(kernel,"accept-1","acceptRecord", ...
                struct('recordKey',first.output.functions(1).kernelRecordKey,'alias',"score")));
            accepted2 = kernel.execute(testCase.request(kernel,"accept-2","acceptRecord", ...
                struct('recordKey',second.output.functions(1).kernelRecordKey,'alias',"score")));

            testCase.verifyEqual(accepted1.output.alias,"score");
            testCase.verifyEqual(accepted2.output.alias,"score_2");
            testCase.verifyTrue(any(contains(accepted2.notices,"score_2")));
            rejected = kernel.execute(testCase.request(kernel,"accept-x","acceptRecord", ...
                struct('recordKey',"client-invented",'alias',"bad")));
            testCase.verifyEqual(rejected.verdict,"invalid");
        end

        function composesAcrossFilesAndExpandsTransparentOperators(testCase)
            kernel = OceanTools.SymbolicNotebook.Kernel("test-session");
            first = testCase.scan(kernel,"function y = offset(x)"+newline+"y = x + 1;"+newline+"end", ...
                "matlab","offset.m","one");
            second = testCase.scan(kernel,"def scale(x):"+newline+"    return x * 2", ...
                "python","scale.py","two");
            testCase.accept(kernel,first,"offset","accept-offset");
            testCase.accept(kernel,second,"scale","accept-scale");
            argument = struct('expression',"x",'domain',"real",'shape',"scalar");
            inner = kernel.execute(testCase.request(kernel,"compose-1","compose", ...
                struct('alias',"offset",'arguments',argument)));
            nestedArgument = struct('expression',"",'domain',"real",'shape',"scalar", ...
                'compositionId',inner.output.compositionId);
            outer = kernel.execute(testCase.request(kernel,"compose-2","compose", ...
                struct('alias',"scale",'arguments',nestedArgument)));
            expanded = kernel.execute(testCase.request(kernel,"expand","expand", ...
                struct('compositionId',outer.output.compositionId)));

            testCase.verifyEqual(inner.verdict,"unknown");
            testCase.verifyEqual(outer.verdict,"unknown");
            testCase.verifyEqual(expanded.verdict,"valid");
            testCase.verifyTrue(contains(expanded.output.expanded,"x"));
            testCase.verifyTrue(contains(expanded.output.expanded,"+ 1"));
            testCase.verifyTrue(contains(expanded.output.expanded,"* 2"));
        end

        function rejectsContradictoryContractAndDistinguishesUnknown(testCase)
            kernel = OceanTools.SymbolicNotebook.Kernel("test-session");
            scanned = testCase.scan(kernel,"function y = rootish(x)"+newline+"y = sqrt(x);"+newline+"end", ...
                "matlab","rootish.m","one");
            accepted = testCase.accept(kernel,scanned,"rootish","accept");
            kernel.Registry.updateContracts(accepted.output.alias, ...
                struct('name',"x",'kind',"parameter",'domain',"positive",'shape',"scalar"), ...
                struct('name',"y",'kind',"return",'domain',"real",'shape',"scalar"));

            unknownArg = struct('expression',"u",'domain',"unknown",'shape',"scalar");
            unknown = kernel.execute(testCase.request(kernel,"unknown","compose", ...
                struct('alias',"rootish",'arguments',unknownArg)));
            invalidArg = struct('expression',"z",'domain',"complex",'shape',"scalar");
            invalid = kernel.execute(testCase.request(kernel,"invalid","compose", ...
                struct('alias',"rootish",'arguments',invalidArg)));

            testCase.verifyEqual(unknown.verdict,"unknown");
            testCase.verifyEqual(invalid.verdict,"invalid");
            testCase.verifyEqual(invalid.errorCode,"OceanMath:Symbolic:ContractMismatch");
        end

        function inspectsThroughSafeWorkspaceAndRoundTripsJson(testCase)
            kernel = OceanTools.SymbolicNotebook.Kernel("test-session");
            payload = struct('expression',"x^2 + 1", ...
                'symbols',struct('name',"x",'domain',"real"));
            response = kernel.execute(testCase.request(kernel,"inspect","inspect",payload));
            testCase.verifyEqual(response.verdict,"valid");
            testCase.verifyTrue(contains(response.output.result,"x"));
            encoded = OceanTools.SymbolicNotebook.dispatchJson(kernel, ...
                string(jsonencode(testCase.request(kernel,"json","inspect",payload))));
            decoded = jsondecode(encoded);
            testCase.verifyEqual(string(decoded.jobId),"json");
            testCase.verifyFalse(isfield(decoded.output,'symbolicResult'));
        end

        function opaqueOperatorsStayAtomic(testCase)
            kernel = OceanTools.SymbolicNotebook.Kernel("test-session");
            source = join(["function y = choose(x)","if x > 0","y = x;","else","y = -x;","end","end"],newline);
            scanned = testCase.scan(kernel,source,"matlab","choose.m","one");
            testCase.verifyEqual(string(scanned.output.functions(1).classification),"opaque");
            testCase.accept(kernel,scanned,"choose","accept");
            argument = struct('expression',"x",'domain',"real",'shape',"scalar");
            composition = kernel.execute(testCase.request(kernel,"compose","compose", ...
                struct('alias',"choose",'arguments',argument)));
            transformed = kernel.execute(testCase.request(kernel,"differentiate","transform", ...
                struct('compositionId',composition.output.compositionId,'operation',"differentiate",'variable',"x")));
            expanded = kernel.execute(testCase.request(kernel,"expand","expand", ...
                struct('compositionId',composition.output.compositionId)));
            testCase.verifyEqual(transformed.verdict,"unsupported");
            testCase.verifyEqual(expanded.verdict,"unsupported");
        end
    end

    methods (Access = private)
        function response = scan(testCase,kernel,text,language,path,job)
            response = kernel.execute(testCase.request(kernel,job,"scanText", ...
                struct('sourceText',text,'language',language,'filePath',path)));
            testCase.assertSuccess(response);
        end

        function response = accept(testCase,kernel,scanResponse,alias,job)
            response = kernel.execute(testCase.request(kernel,job,"acceptRecord", ...
                struct('recordKey',scanResponse.output.functions(1).kernelRecordKey,'alias',alias)));
            testCase.assertSuccess(response);
        end

        function request = request(~,kernel,job,command,payload)
            request = struct('protocolVersion',"1.0.0",'sessionId',kernel.SessionId, ...
                'jobId',job,'notebookRevision',1,'command',command,'payload',payload, ...
                'sourceRefs',struct([]));
        end

        function assertSuccess(testCase,response)
            testCase.verifyTrue(any(response.verdict == ["valid","unknown"]), ...
                "Kernel response failed: " + join(response.notices,"; "));
        end
    end
end
