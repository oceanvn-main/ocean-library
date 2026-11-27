classdef SymbolicWorkspaceTest < OceanTest.BaseTestCase
    % SYMBOLICWORKSPACETEST Exercises real Symbolic Math Toolbox operations.
    properties (TestParameter)
        rejectedExpression = struct( ...
            'systemCall', "system('whoami')", ...
            'statementSequence', "x;delete('*')", ...
            'functionHandle', "@sin", ...
            'arrayLiteral', "[1,2]", ...
            'objectDispatch', "java.lang.Runtime", ...
            'stringEvaluation', "eval('1')")
    end
    methods (Test)
        function gaussianAliasAndDerivative(testCase)
            workspace = OceanMath.Symbolic.Workspace();
            workspace.addSymbol("x", "real");
            workspace.addSymbol("mu", "real");
            workspace.addSymbol("sigma", "positive");
            workspace.define("G", ["t", "m", "s"], ...
                "exp(-(t-m)^2/(2*s^2))/(s*sqrt(2*pi))", ...
                ["real", "real", "positive"], "positive");
            report = workspace.inspect("G(x,mu,sigma)");
            testCase.verifyEqual(report.compact, "G(x,mu,sigma)");
            testCase.verifyTrue(contains(report.compactLatex, "G"));
            testCase.verifyTrue(contains(report.compactLatex, "\mathrm{G}"));
            testCase.verifyFalse(contains(report.compactLatex, "\operatorname"));
            testCase.verifyFalse(contains(report.expanded, "ows"));
            x = workspace.inspect("x");
            mu = workspace.inspect("mu");
            sigma = workspace.inspect("sigma");
            expected = exp(-(x.symbolicResult-mu.symbolicResult)^2/(2*sigma.symbolicResult^2)) / ...
                (sigma.symbolicResult*sqrt(2*sym(pi)));
            testCase.verifyTrue(isAlways(simplify(report.symbolicExpanded-expected) == 0));
            differentiated = workspace.calculate("G(x,mu,sigma)", "differentiate", "x");
            testCase.verifyTrue(isAlways(simplify(differentiated.symbolicResult-diff(expected,x.symbolicResult)) == 0));
        end

        function definiteAndUnknownDomains(testCase)
            workspace = OceanMath.Symbolic.Workspace();
            workspace.addSymbol("x", "real");
            workspace.define("PositiveIdentity", "t", "t", "positive", "positive");
            unknown = workspace.inspect("PositiveIdentity(x)");
            testCase.verifyTrue(any(string({unknown.checks.status}) == "unknown"));
            testCase.verifyError(@() workspace.inspect("PositiveIdentity(0)"), 'OceanMath:Symbolic:Domain');
            testCase.verifyError(@() workspace.inspect("PositiveIdentity(-2)"), 'OceanMath:Symbolic:Domain');
            passed = workspace.inspect("PositiveIdentity(2)");
            testCase.verifyTrue(all(string({passed.checks.status}) == "passed"));
            testCase.verifyError(@() workspace.define("Bad", "t", "-1", "real", "positive"), 'OceanMath:Symbolic:Domain');
        end

        function collisionPreservesExistingBlocks(testCase)
            workspace = OceanMath.Symbolic.Workspace();
            first = workspace.define("Square", "x", "x^2");
            second = workspace.define("Square", "x", "x^3");
            reserved = workspace.addSymbol("sin", "real");
            testCase.verifyEqual(first, "Square");
            testCase.verifyEqual(second, "Square_2");
            testCase.verifyEqual(reserved, "sin_2");
            original = workspace.inspect("Square(3)");
            added = workspace.inspect("Square_2(3)");
            testCase.verifyEqual(double(original.symbolicResult), 9);
            testCase.verifyEqual(double(added.symbolicResult), 27);
            testCase.verifyNotEmpty(workspace.Notices);
        end

        function renameUpdatesCallsAndPreservesParameterScope(testCase)
            workspace = OceanMath.Symbolic.Workspace();
            workspace.addSymbol("x", "real");
            workspace.define("Square", "t", "t^2");
            workspace.define("Shift", "x", "Square(x)+1");
            workspace.define("Global", "t", "t+x");
            workspace.rename("Square", "Power");
            testCase.verifyEqual(workspace.Definitions(2).expression, "Power(x)+1");
            workspace.rename("x", "globalX");
            testCase.verifyEqual(workspace.Definitions(2).expression, "Power(x)+1");
            testCase.verifyEqual(workspace.Definitions(3).expression, "t+globalX");
            report = workspace.inspect("Shift(2)");
            testCase.verifyEqual(double(report.symbolicResult), 5);
        end

        function nestedBlocksDoNotCaptureGlobalSymbols(testCase)
            workspace = OceanMath.Symbolic.Workspace();
            workspace.addSymbol("x", "real");
            workspace.define("Inner", "t", "t+x");
            workspace.define("Outer", "x", "Inner(x)");
            report = workspace.inspect("Outer(2)");
            globalValue = workspace.inspect("x");
            testCase.verifyTrue(isAlways(simplify(report.symbolicResult-globalValue.symbolicResult-2) == 0));
        end

        function scalarOperationsAndSimultaneousSubstitution(testCase)
            workspace = OceanMath.Symbolic.Workspace();
            workspace.addSymbol("x", "real");
            workspace.addSymbol("y", "real");
            simplified = workspace.calculate("sin(x)^2+cos(x)^2", "simplify");
            testCase.verifyEqual(double(simplified.symbolicResult), 1);
            expanded = workspace.calculate("(x+1)^3", "expand");
            expected = workspace.inspect("x^3+3*x^2+3*x+1");
            testCase.verifyTrue(isAlways(simplify(expanded.symbolicResult-expected.symbolicResult) == 0));
            integrated = workspace.calculate("x^2", "integrate", "x");
            variable = workspace.inspect("x");
            testCase.verifyTrue(isAlways(simplify(diff(integrated.symbolicResult,variable.symbolicResult)-variable.symbolicResult^2) == 0));
            substituted = workspace.calculate("x-y", "substitute", "x", struct('x', "y", 'y', "x"));
            expectedSwap = workspace.inspect("y-x");
            testCase.verifyTrue(isAlways(simplify(substituted.symbolicResult-expectedSwap.symbolicResult) == 0));
            solved = workspace.calculate("x^2=4", "solve", "x");
            testCase.verifyEqual(sort(double(solved.symbolicResult)), [-2;2]);
            factored = workspace.calculate("x^2-1", "factor");
            testCase.verifyTrue(isAlways(simplify(prod(factored.symbolicResult)-(variable.symbolicResult^2-1)) == 0));
        end

        function bundleRoundTripAndCollisionRemapping(testCase)
            original = OceanMath.Symbolic.Workspace();
            original.addSymbol("a", "real");
            original.define("F", "x", "x+a");
            original.define("H", "a", "F(a)");
            decoded = jsondecode(jsonencode(original.exportBundle()));
            restored = OceanMath.Symbolic.Workspace();
            restored.importBundle(decoded);
            testCase.verifyEqual(restored.Definitions(2).expression, "F(a)");
            target = OceanMath.Symbolic.Workspace();
            target.addSymbol("a", "positive");
            target.define("F", "x", "x^2");
            names = target.importBundle(decoded);
            testCase.verifyEqual(names(1).newName, "a_2");
            testCase.verifyEqual(names(2).newName, "F_2");
            testCase.verifyEqual(target.Definitions(2).expression, "x+a_2");
            testCase.verifyEqual(target.Definitions(3).expression, "F_2(a)");
            preserved = target.inspect("F(3)");
            testCase.verifyEqual(double(preserved.symbolicResult), 9);
            empty = OceanMath.Symbolic.Workspace();
            emptySource = OceanMath.Symbolic.Workspace();
            empty.importBundle(jsondecode(jsonencode(emptySource.exportBundle())));
            testCase.verifyEmpty(empty.Symbols);
        end

        function rejectedImportIsTransactional(testCase)
            workspace = OceanMath.Symbolic.Workspace();
            workspace.addSymbol("existing", "real");
            before = jsonencode(workspace.exportBundle());
            source = OceanMath.Symbolic.Workspace();
            source.addSymbol("incoming", "positive");
            source.define("Safe", "x", "x+incoming");
            bundle = source.exportBundle();
            bundle.definitions(1).expression = "system(1)";
            testCase.verifyError(@() workspace.importBundle(bundle), 'OceanMath:Symbolic:UnknownName');
            testCase.verifyEqual(jsonencode(workspace.exportBundle()), before);
            bundle.definitions(1).expression = "Safe(x)";
            testCase.verifyError(@() workspace.importBundle(bundle), 'OceanMath:Symbolic:UnknownName');
            testCase.verifyEqual(jsonencode(workspace.exportBundle()), before);
        end

        function grammarRejectsCode(testCase, rejectedExpression)
            workspace = OceanMath.Symbolic.Workspace();
            workspace.addSymbol("x", "real");
            testCase.verifyError(@() workspace.inspect(rejectedExpression), 'OceanMath:Symbolic:Syntax');
        end

        function grammarRejectsInvalidContracts(testCase)
            workspace = OceanMath.Symbolic.Workspace();
            workspace.addSymbol("x", "real");
            testCase.verifyError(@() workspace.inspect("unknown+1"), 'OceanMath:Symbolic:UnknownName');
            testCase.verifyError(@() workspace.inspect("x(1)=2"), 'OceanMath:Symbolic:UnknownName');
            testCase.verifyError(@() workspace.define("Bad", ["x","x"], "x"), 'OceanMath:Symbolic:Parameters');
            testCase.verifyError(@() workspace.inspect("x^101"), 'OceanMath:Symbolic:Limit');
            testCase.verifyError(@() workspace.inspect("1/0"), 'OceanMath:Symbolic:Nonfinite');
            negative = workspace.inspect("-2^2");
            testCase.verifyEqual(double(negative.symbolicResult), -4);
        end

        function substitutionRetainsContractsAfterCancellation(testCase)
            workspace = OceanMath.Symbolic.Workspace();
            workspace.addSymbol("x", "real");
            workspace.define("PositiveConstant", "s", "1", "positive", "positive");
            report = workspace.inspect("PositiveConstant(x)");
            testCase.verifyEqual(double(report.symbolicResult), 1);
            testCase.verifyError(@() workspace.calculate("PositiveConstant(x)", "substitute", "x", struct('x',"0")), ...
                'OceanMath:Symbolic:Domain');
            testCase.verifyError(@() workspace.calculate("x/x", "substitute", "x", struct('x',"0")), ...
                'OceanMath:Symbolic:Nonfinite');
            testCase.verifyError(@() workspace.calculate("log(x)-log(x)", "substitute", "x", struct('x',"0")), ...
                'OceanMath:Symbolic:Nonfinite');
            substituted = workspace.calculate("PositiveConstant(x)", "substitute", "x", struct('x',"2"));
            testCase.verifyTrue(all(string({substituted.checks.status}) == "passed"));
        end

        function renamingDoesNotRewriteScientificNotation(testCase)
            workspace = OceanMath.Symbolic.Workspace();
            workspace.addSymbol("e3", "real");
            workspace.define("Scientific", "x", "x+1e3");
            workspace.rename("e3", "scale");
            testCase.verifyEqual(workspace.Definitions(1).expression, "x+1e3");
            report = workspace.inspect("Scientific(2)");
            testCase.verifyEqual(double(report.symbolicResult), 1002);
        end

        function renamedGlobalCannotBeCapturedByAFormalParameter(testCase)
            workspace = OceanMath.Symbolic.Workspace();
            workspace.addSymbol("x", "real");
            workspace.define("F", "t", "t+x");
            actualName = workspace.rename("x", "t");
            testCase.verifyEqual(actualName, "t_2");
            testCase.verifyEqual(workspace.Definitions(1).expression, "t+t_2");
            report = workspace.inspect("F(2)");
            globalValue = workspace.inspect(actualName);
            testCase.verifyTrue(isAlways(simplify(report.symbolicResult-globalValue.symbolicResult-2) == 0));
        end

        function importedGlobalSuffixAvoidsExistingAndIncomingFormals(testCase)
            target = OceanMath.Symbolic.Workspace();
            target.addSymbol("x", "real");
            target.define("Existing", "x_2", "x_2+1");
            source = OceanMath.Symbolic.Workspace();
            source.addSymbol("x", "real");
            source.define("Incoming", "x_3", "x_3+x");
            renames = target.importBundle(source.exportBundle());
            testCase.verifyEqual(renames(1).newName, "x_4");
            testCase.verifyEqual(target.Definitions(2).expression, "x_3+x_4");
            report = target.inspect("Incoming(2)");
            importedGlobal = target.inspect("x_4");
            testCase.verifyTrue(isAlways(simplify(report.symbolicResult-importedGlobal.symbolicResult-2) == 0));
            existing = target.inspect("Existing(2)");
            testCase.verifyEqual(double(existing.symbolicResult), 3);
        end
    end
end
