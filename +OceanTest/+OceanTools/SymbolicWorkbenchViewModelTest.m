classdef SymbolicWorkbenchViewModelTest < OceanTest.BaseTestCase
    % Public presentation workflows; no external AI service required.
    methods (Test)
        function bundleImportClearsPriorSource(testCase)
            folder=testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(OceanMath.Symbolic.Workspace());
            vm.importFile(string(fullfile(testCase.LibraryRoot,'+OceanMath','+Information','ScoreFunction.m')));
            vm.saveBundle(string(fullfile(folder.Folder,'bundle.json')));
            vm.importFile(string(fullfile(folder.Folder,'bundle.json')));
            testCase.verifyEqual(vm.SourceText,"");
            testCase.verifyEqual(vm.SourcePath,"");
            testCase.verifyEmpty(vm.Candidates);
            testCase.verifyTrue(contains(vm.AIDraftText,"No AI draft"));
        end
        function candidatePromotionRetainsEvidence(testCase)
            folder=testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            path=fullfile(folder.Folder,'square.py');
            fid=fopen(path,'w'); fprintf(fid,'y = x**2\n'); fclose(fid);
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(OceanMath.Symbolic.Workspace());
            vm.importFile(string(path)); vm.useCandidate(1);
            vm.defineBlock("Square","x","real",vm.Expression,"nonnegative");
            vm.addSymbol("x","real"); vm.calculate();
            definition=vm.getBlock(1);
            testCase.verifyEqual(string(definition.parameterDomains),"real");
            testCase.verifyEqual(definition.provenance.sourceLines,1);
            testCase.verifyEqual(string(definition.provenance.sourcePath),string(path));
            testCase.verifyTrue(contains(vm.ReportText,"source-candidate"));
            testCase.verifyTrue(contains(vm.AIDraftText,"No AI draft"));
        end
        function calculateAndClearStaleResult(testCase)
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(OceanMath.Symbolic.Workspace());
            vm.addSymbol("x","real"); vm.Expression="x^2";
            vm.Operation="differentiate"; vm.calculate();
            testCase.verifyTrue(contains(vm.ReportText,"2*x"));
            vm.Expression="system('invalid')"; vm.calculate();
            testCase.verifyEqual(vm.ResultLatex,"");
            testCase.verifyTrue(startsWith(vm.Status,"Cannot calculate:"));
        end
        function gaussianAndCollisionNotices(testCase)
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(OceanMath.Symbolic.Workspace());
            vm.gaussianExample();
            testCase.verifyEqual(vm.Expression,"G(x,mu,sigma)");
            testCase.verifyFalse(startsWith(vm.Status,"Cannot"));
            vm.gaussianExample();
            testCase.verifyEqual(vm.Expression,"G_2(x_2,mu_2,sigma_2)");
            testCase.verifyTrue(contains(vm.Notices,"G_2"));
        end
        function substitutionsRejectMalformedInput(testCase)
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(OceanMath.Symbolic.Workspace());
            vm.addSymbol("x","real"); vm.Expression="x+1";
            vm.Operation="substitute"; vm.Substitutions="x=2"; vm.calculate();
            testCase.verifyTrue(contains(vm.ReportText,"Result: 3"));
            vm.Substitutions="not a substitution"; vm.calculate();
            testCase.verifyEqual(vm.ResultLatex,"");
        end
        function sourceImportIsReadOnly(testCase)
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(OceanMath.Symbolic.Workspace());
            path=fullfile(testCase.LibraryRoot,'+OceanMath','+Information','ScoreFunction.m');
            vm.importFile(string(path));
            testCase.verifyNotEmpty(vm.SourceText);
            testCase.verifyEmpty(vm.Workspace.Definitions);
            testCase.verifyEmpty(vm.Workspace.Symbols);
            testCase.verifyTrue(contains(vm.Status,"not a complete symbolic translation"));
        end
        function renameUpdatesCurrentCall(testCase)
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(OceanMath.Symbolic.Workspace());
            vm.gaussianExample(); vm.renameBlock("G","density");
            testCase.verifyEqual(vm.Expression,"density(x,mu,sigma)");
            vm.calculate();
            testCase.verifyFalse(startsWith(vm.Status,"Cannot"));
        end
    end
end
