classdef SymbolicSourceExtractorTest < OceanTest.BaseTestCase
    methods (Test)
        function extractsStraightLineMatlab(testCase)
            source="function y = affine(a, W, b)"+newline+"u = a * W;"+newline+"y = u + b;"+newline+"end";
            scan=OceanMath.Symbolic.SourceExtractor.scanText(source,"matlab","affine.m");
            testCase.verifyEqual(scan.functions.classification,"transparent");
            testCase.verifyEqual(scan.functions.expression,"(a * W) + b");
            testCase.verifyEqual(scan.functions.computationalSupport,"conditional-symbolic");
        end

        function extractsPythonToSameForm(testCase)
            source="def affine(a, W, b):"+newline+"    u = a @ W"+newline+"    return u + b";
            scan=OceanMath.Symbolic.SourceExtractor.scanText(source,"python","affine.py");
            testCase.verifyEqual(scan.functions.classification,"transparent");
            testCase.verifyEqual(scan.functions.expression,"(a * W) + b");
            testCase.verifyEqual(scan.functions.computationalSupport,"abstract-only");
        end

        function preservesComplexFunctionAsOpaque(testCase)
            source="function thetaHat = optimize(loss, theta0)"+newline+"for k=1:100"+newline+"thetaHat=step(loss,theta0);"+newline+"end";
            scan=OceanMath.Symbolic.SourceExtractor.scanText(source,"matlab","optimize.m");
            testCase.verifyEqual(scan.functions.classification,"opaque");
            testCase.verifyEqual(scan.functions.inputs,["loss","theta0"]);
            testCase.verifyEqual(strlength(scan.functions.expression),0);
        end

        function capturesDocumentedEntropyFormula(testCase)
            source="function H = Entropy(X)"+newline+"% H(X) = -sum(p(x) * log2(p(x)))"+newline+"H = helper(X);"+newline+"end";
            scan=OceanMath.Symbolic.SourceExtractor.scanText(source,"matlab","Entropy.m");
            candidate=scan.functions.formulaCandidates;
            testCase.verifyEqual(candidate.origin,"documentation-derived");
            testCase.verifyEqual(candidate.computationalSupport,"display-only");
            testCase.verifySubstring(candidate.expression,"H(X) = -sum");
        end

        function rejectsUnsupportedLanguage(testCase)
            testCase.verifyError(@() OceanMath.Symbolic.SourceExtractor.scanText("x", "ruby"), ...
                'OceanMath:Symbolic:Language');
        end

        function pythonNoReturnSideEffectIsNotExtractable(testCase)
            source="@logged"+newline+"async def save_result(path):"+newline+"    print(path)";
            scan=OceanMath.Symbolic.SourceExtractor.scanText(source,"python","writer.py");
            testCase.verifyEqual(scan.functions.classification,"not-extractable");
            testCase.verifyEmpty(scan.functions.outputs);
        end

        function scansMultilinePythonSignature(testCase)
            source="def affine("+newline+"    x,"+newline+"    weight"+newline+"):"+newline+"    return x * weight";
            scan=OceanMath.Symbolic.SourceExtractor.scanText(source,"python","affine.py");
            testCase.verifyEqual(scan.functions.inputs,["x","weight"]);
            testCase.verifyEqual(scan.functions.expression,"x * weight");
        end

        function normalizedExpressionPreservesLiteralEvidence(testCase)
            source="function y = scale(x,a)"+newline+"y = x .* a;"+newline+"end";
            scan=OceanMath.Symbolic.SourceExtractor.scanText(source,"matlab","scale.m");
            testCase.verifyEqual(scan.functions.expression,"x * a");
            testCase.verifyEqual(scan.functions.literalExpression,"y = x .* a;");
            testCase.verifyEqual(scan.functions.expressionOrigin,"normalized traced expression");
        end

        function documentedFormulaRequiresVariableMapping(testCase)
            source="function H = entropy(X)"+newline+"% H(X) = -sum(p * log(p))"+newline+"if isempty(X)"+newline+"H=0;"+newline+"end";
            scan=OceanMath.Symbolic.SourceExtractor.scanText(source,"matlab","entropy.m");
            validation=OceanMath.Symbolic.SourceExtractor.validateFormulaCandidate(scan.functions,1);
            testCase.verifyFalse(validation.validated);
            testCase.verifyTrue(any(validation.unresolvedVariables=="p"));
        end

        function sourceCommandsAreNeverAcceptedAsExpressions(testCase)
            source="function y = dangerous(x)"+newline+"y = system('do-not-run');"+newline+"end";
            scan=OceanMath.Symbolic.SourceExtractor.scanText(source,"matlab","dangerous.m");
            testCase.verifyEqual(scan.functions.classification,"opaque");
            testCase.verifyEqual(strlength(scan.functions.expression),0);
        end

        function scansRepositoryEntropyAsOpaqueWithFormulas(testCase)
            file=fullfile(testCase.LibraryRoot,"+OceanMath","+Information","Entropy.m");
            scan=OceanMath.Symbolic.SourceExtractor.scanFile(file);
            testCase.verifyEqual(scan.functions(1).displayName,"Entropy");
            testCase.verifyEqual(scan.functions(1).classification,"opaque");
            testCase.verifyGreaterThanOrEqual(numel(scan.functions(1).formulaCandidates),2);
        end

        function zeroArityMetadataIsOpaqueAndComplete(testCase)
            file=fullfile(testCase.LibraryRoot,"+OceanMath","+Information","ScoreFunction.m");
            scan=OceanMath.Symbolic.SourceExtractor.scanFile(file);
            names=string({scan.functions.displayName}); metadata=scan.functions(names=="getMetadata");
            testCase.verifyEqual(metadata.signature,"getMetadata()");
            testCase.verifyEqual(metadata.classification,"opaque");
            testCase.verifyEqual(strlength(metadata.expression),0);
            testCase.verifyFalse(any(ismissing([metadata.id,metadata.qualifiedName, ...
                metadata.displayName,metadata.signature,metadata.classification])));
        end
    end
end
