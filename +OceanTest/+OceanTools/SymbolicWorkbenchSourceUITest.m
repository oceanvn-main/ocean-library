classdef SymbolicWorkbenchSourceUITest < OceanTest.BaseTestCase
    % Source-symbol review workflow without executing imported source.

    methods (Test)
        function matlabFunctionCanBeAcceptedAsFormula(testCase)
            folder=testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            path=fullfile(folder.Folder,'affine.m');
            testCase.writeText(path,["function y = affine(x,w,b)"; "y = w*x+b;"; "end"]);
            workspace=OceanMath.Symbolic.Workspace();
            registry=OceanMath.Symbolic.Registry();
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(workspace,registry);

            vm.importFile(string(path));
            testCase.verifyEqual(size(vm.FunctionRows,1),1);
            testCase.verifyEqual(string(vm.FunctionRows{1,3}),"transparent");
            vm.selectSourceFunction(1);
            testCase.verifyTrue(contains(vm.SelectedFunctionDetail,"ABSTRACT SYMBOLIC INTERFACE"));
            testCase.verifyEqual(size(vm.FormulaRows,1),1);
            vm.acceptSelectedFormula(1,"A");

            testCase.verifyEqual(string(registry.Records(1).alias),"A");
            testCase.verifyEqual(string(registry.Records(1).record.classification),"transparent");
            testCase.verifyTrue(contains(vm.Status,"Confirm"));
            testCase.verifyFalse(startsWith(vm.Status,"Cannot"));
            testCase.verifyEqual(size(vm.AcceptedRows,1),1);
        end

        function complexPythonFunctionAcceptsOpaqueInterface(testCase)
            folder=testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            path=fullfile(folder.Folder,'optimizer.py');
            testCase.writeText(path,["def optimize(loss, theta):"; ...
                "    for step in range(10):"; "        theta = update(loss, theta)"; "    return theta"]);
            registry=OceanMath.Symbolic.Registry();
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(OceanMath.Symbolic.Workspace(),registry);

            vm.importFile(string(path)); vm.selectSourceFunction(1);
            vm.acceptSelectedAbstract("f");

            testCase.verifyEqual(string(registry.Records(1).alias),"f");
            testCase.verifyEqual(string(registry.Records(1).record.classification),"opaque");
            testCase.verifyTrue(contains(vm.Status,"not expanded or computed through"));
            testCase.verifyTrue(contains(vm.SelectedFunctionDetail,"optimize(loss, theta)"));
        end

        function sourceReviewControlsArePresent(testCase)
            app=OceanTools.SymbolicWorkbench.launch('Visible',false);
            testCase.addTeardown(@()delete(app.View));
            testCase.verifyClass(app.Registry,'OceanMath.Symbolic.Registry');
            testCase.verifyNotEmpty(findall(app.View.Figure,'Tag','SymbolicSourceFunctionTable'));
            testCase.verifyNotEmpty(findall(app.View.Figure,'Tag','SymbolicFormulaTable'));
            testCase.verifyNotEmpty(findall(app.View.Figure,'Tag','AcceptedSymbolicLibrary'));
            testCase.verifyNotEmpty(findall(app.View.Figure,'Tag','SymbolicFunctionDetail'));
        end

        function importsBrowserExtractionPacketByRescanningSource(testCase)
            folder=testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            path=fullfile(folder.Folder,'affine.symbolic.json');
            packet=struct('kind','ocean-symbolic-extraction','schemaVersion','1.0', ...
                'source',struct('filename','affine.py','text',sprintf('def affine(x,w,b):\n    return w*x+b\n'),'language','python'), ...
                'candidates',struct([]),'manifest',struct());
            testCase.writeText(path,string(jsonencode(packet)));
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(OceanMath.Symbolic.Workspace(),OceanMath.Symbolic.Registry());

            vm.importFile(string(path));

            testCase.verifyEqual(size(vm.FunctionRows,1),1);
            testCase.verifyEqual(string(vm.FunctionRows{1,1}),"affine");
            testCase.verifyTrue(contains(vm.Status,"safely rescanned"));
            testCase.verifyEmpty(vm.Workspace.Definitions);
        end

        function documentationFormulaRequiresValidatedMapping(testCase)
            folder=testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            path=fullfile(folder.Folder,'entropy.m');
            testCase.writeText(path,["function h = entropy(p)"; "% H(X) = -sum(X)"; "h = -sum(p);"; "end"]);
            registry=OceanMath.Symbolic.Registry();
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(OceanMath.Symbolic.Workspace(),registry);
            vm.importFile(string(path)); vm.selectSourceFunction(1);

            vm.acceptSelectedFormula(2,"H","");
            testCase.verifyEmpty(registry.Records);
            testCase.verifyTrue(contains(vm.Status,"remains display-only"));
            vm.acceptSelectedFormula(2,"H","X=p");
            testCase.verifyEqual(string(registry.Records(1).alias),"H");
            testCase.verifyEqual(string(registry.Records(1).record.computationalSupport),"display-only");
        end

        function collisionAndRejectDecisionsAreVisible(testCase)
            folder=testCase.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture);
            registry=OceanMath.Symbolic.Registry();
            vm=OceanTools.SymbolicWorkbench.ViewModels.Main(OceanMath.Symbolic.Workspace(),registry);
            first=fullfile(folder.Folder,'first.m'); second=fullfile(folder.Folder,'second.m');
            testCase.writeText(first,["function y = f(x)"; "y=x+1;"; "end"]);
            testCase.writeText(second,["function y = f(x)"; "y=x+2;"; "end"]);
            vm.importFile(string(first)); vm.selectSourceFunction(1); vm.acceptSelectedAbstract("f");
            vm.importFile(string(second)); vm.selectSourceFunction(1); vm.acceptSelectedAbstract("f");
            testCase.verifyEqual(string(registry.Records(2).alias),"f_2");
            testCase.verifyTrue(contains(vm.Notices,"f_2"));

            third=fullfile(folder.Folder,'third.m');
            testCase.writeText(third,["function y = rejected(x)"; "y=x;"; "end"]);
            vm.importFile(string(third)); vm.selectSourceFunction(1); vm.rejectSelected();
            vm.acceptSelectedAbstract("rejected");
            testCase.verifyEqual(numel(registry.Records),2);
            testCase.verifyTrue(contains(vm.Status,"was rejected"));
            testCase.verifyEqual(string(vm.FunctionRows{1,3}),"review-rejected");
        end
    end

    methods (Static,Access=private)
        function writeText(path,lines)
            arguments
                path (1,:) char
                lines (:,1) string
            end
            identifier=fopen(path,'w','n','UTF-8');
            if identifier<0, error('OceanTest:Write','Cannot create fixture.'); end
            cleaner=onCleanup(@()fclose(identifier)); %#ok<NASGU>
            fprintf(identifier,'%s\n',char(join(lines,newline)));
        end
    end
end
