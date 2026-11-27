classdef SymbolicRegistryTest < OceanTest.BaseTestCase
    methods (Test)
        function composesOpaqueOperatorWithContracts(testCase)
            scan=OceanMath.Symbolic.SourceExtractor.scanText("function y = f(loss, theta)"+newline+"y=runSolver(loss,theta);"+newline+"end","matlab","f.m");
            record=scan.functions; record.classification="opaque";
            registry=OceanMath.Symbolic.Registry(); alias=registry.accept(record,"f");
            args(1)=struct('expression',"L",'domain',"unknown",'shape',"unknown");
            args(2)=struct('expression',"theta0",'domain',"real",'shape',"vector[p]");
            result=registry.compose(alias,args);
            testCase.verifyEqual(result.expression,"f(L, theta0)");
            testCase.verifyEqual(result.ast.kind,"call");
        end

        function renamesAliasCollision(testCase)
            scan=OceanMath.Symbolic.SourceExtractor.scanText("function y = f(x)"+newline+"y=x+1;"+newline+"end","matlab","a.m");
            second=scan.functions; second.id="other";
            registry=OceanMath.Symbolic.Registry(); registry.accept(scan.functions,"f");
            alias=registry.accept(second,"f");
            testCase.verifyEqual(alias,"f_2");
            testCase.verifyNotEmpty(registry.Notices);
        end

        function expandsTransparentFunction(testCase)
            scan=OceanMath.Symbolic.SourceExtractor.scanText("function y = affine(x,a)"+newline+"y=a*x+1;"+newline+"end","matlab","affine.m");
            registry=OceanMath.Symbolic.Registry(); registry.accept(scan.functions,"g");
            expanded=registry.expand("g",["z","c"]);
            testCase.verifyEqual(expanded,"(c)*(z)+1");
        end

        function displayOnlyFormulaCannotExpand(testCase)
            scan=OceanMath.Symbolic.SourceExtractor.scanText("function y = f(x)"+newline+"y=x+1;"+newline+"end","matlab","f.m");
            record=scan.functions;
            record.computationalSupport="display-only";
            registry=OceanMath.Symbolic.Registry();
            registry.accept(record,"f");
            testCase.verifyError(@() registry.expand("f","z"), ...
                'OceanMath:Symbolic:UnsupportedExpansion');
        end

        function derivativeAcrossOpaqueIsExplicitlyUnsupported(testCase)
            scan=OceanMath.Symbolic.SourceExtractor.scanText("function y = f(x)"+newline+"if x>0"+newline+"y=x;"+newline+"end","matlab","f.m");
            registry=OceanMath.Symbolic.Registry(); registry.accept(scan.functions,"f");
            arg=struct('expression',"x",'domain',"real",'shape',"scalar");
            composition=registry.compose("f",arg);
            report=registry.manipulate(composition,"differentiate","x");
            testCase.verifyEqual(report.status,"unsupported");
            testCase.verifySubstring(report.detail,"opaque operator");
        end

        function editedContractsUseDomainLattice(testCase)
            scan=OceanMath.Symbolic.SourceExtractor.scanText("function y = f(x)"+newline+"y=x+1;"+newline+"end","matlab","f.m");
            registry=OceanMath.Symbolic.Registry(); registry.accept(scan.functions,"f");
            in=struct('name',"x",'kind',"scalar",'domain',"real",'shape',"scalar");
            out=struct('name',"y",'kind',"scalar",'domain',"real",'shape',"scalar");
            registry.updateContracts("f",in,out);
            arg=struct('expression',"z",'domain',"positive",'shape',"scalar");
            composition=registry.compose("f",arg);
            testCase.verifyEqual(composition.checks.status,"passed");
            testCase.verifyEqual(composition.domain,"real");
        end

        function rejectsInvalidContractVocabulary(testCase)
            scan=OceanMath.Symbolic.SourceExtractor.scanText("function y = f(x)"+newline+"y=x;"+newline+"end","matlab","f.m");
            registry=OceanMath.Symbolic.Registry(); registry.accept(scan.functions,"f");
            in=struct('name',"x",'kind',"scalar",'domain',"banana",'shape',"scalar");
            out=struct('name',"y",'kind',"scalar",'domain',"real",'shape',"scalar");
            testCase.verifyError(@() registry.updateContracts("f",in,out),'OceanMath:Symbolic:ContractDomain');
        end

        function validatesMatrixMultiplicationShapes(testCase)
            left=struct('expression',"A",'domain',"real",'shape',"matrix[m,n]", ...
                'checks',struct([]),'operatorId',"",'ast',struct('kind',"atom"));
            right=struct('expression',"x",'domain',"real",'shape',"vector[n]", ...
                'checks',struct([]),'operatorId',"",'ast',struct('kind',"atom"));
            registry=OceanMath.Symbolic.Registry(); result=registry.combine(left,"*",right);
            testCase.verifyEqual(result.shape,"vector[m]");
            testCase.verifyEqual(result.checks.status,"passed");
        end
    end
end
