classdef SymbolicWorkbenchViewTest < OceanTest.BaseTestCase
    % Native UI construction and observable bindings, without a network call.
    methods (Test)
        function selectingBlockPreservesContracts(testCase)
            app=OceanTools.SymbolicWorkbench.launch('Visible',false);
            testCase.addTeardown(@() delete(app.View));
            app.ViewModel.gaussianExample();
            grid=findall(app.View.Figure,'Tag','SymbolicBlockTable');
            grid.CellSelectionCallback(grid,struct('Indices',[1 1]));
            domains=findall(app.View.Figure,'Tag','SymbolicBlockDomains');
            testCase.verifyEqual(string(domains.Value),"real, real, positive");
            testCase.verifyTrue(contains(app.ViewModel.ReportText,"user-selected-mathematical-example"));
            testCase.verifyTrue(contains(app.ViewModel.AIDraftText,"No AI draft"));
        end
        function constructsAndBinds(testCase)
            app=OceanTools.SymbolicWorkbench.launch('Visible',false);
            testCase.addTeardown(@() delete(app.View));
            testCase.verifyTrue(isgraphics(app.View.Figure));
            app.ViewModel.gaussianExample();
            drawnow;
            labels=findall(app.View.Figure,'Type','uilabel');
            texts=string(get(labels,'Text'));
            testCase.verifyTrue(any(contains(texts,"G")));
            testCase.verifyFalse(startsWith(app.ViewModel.Status,"Cannot"));
            app.ViewModel.addSymbol("x","real");
            testCase.verifyTrue(contains(app.ViewModel.Notices,"x_2"));
        end
        function deletionReleasesListeners(testCase)
            app=OceanTools.SymbolicWorkbench.launch('Visible',false);
            figure=app.View.Figure;
            delete(app.View);
            app.ViewModel.Expression="2+2";
            app.ViewModel.calculate();
            testCase.verifyTrue(contains(app.ViewModel.ReportText,"Result: 4"));
            testCase.verifyFalse(isgraphics(figure));
        end
    end
end
