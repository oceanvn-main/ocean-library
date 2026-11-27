function results = verify_symbolic_workbench()
%VERIFY_SYMBOLIC_WORKBENCH Run native workbench regressions and render previews.
arguments
end
root=fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(root);
files={fullfile(root,'+OceanTest','+OceanMath','SymbolicWorkspaceTest.m'), ...
    fullfile(root,'+OceanTest','+OceanMath','SymbolicSourceExtractorTest.m'), ...
    fullfile(root,'+OceanTest','+OceanMath','SymbolicRegistryTest.m'), ...
    fullfile(root,'+OceanTest','+OceanTools','SymbolicWorkbenchViewModelTest.m'), ...
    fullfile(root,'+OceanTest','+OceanTools','SymbolicAIDraftTest.m'), ...
    fullfile(root,'+OceanTest','+OceanTools','SymbolicWorkbenchViewTest.m'), ...
    fullfile(root,'+OceanTest','+OceanTools','SymbolicWorkbenchSourceUITest.m')};
results=runtests(files);
disp(table(results));
assertSuccess(results);
app=OceanTools.SymbolicWorkbench.launch();
cleanup=onCleanup(@() delete(app.View)); %#ok<NASGU>
app.ViewModel.gaussianExample();
drawnow;
exportapp(app.View.Figure,fullfile(fileparts(mfilename('fullpath')),'symbolic_workbench_preview.png'));
app.ViewModel.importFile(string(fullfile(root,'+OceanMath','+Information','Entropy.m')));
app.ViewModel.selectSourceFunction(1);
app.ViewModel.acceptSelectedAbstract("EntropyOp");
app.ViewModel.updateOperatorContracts("EntropyOp","real,unknown","vector,unknown","real","scalar");
app.ViewModel.composeOperator("EntropyOp","X,options","real,unknown","vector,unknown");
tabs=findall(app.View.Figure,'Type','uitab');
source=tabs(strcmp(get(tabs,'Title'),'Source symbolic interfaces'));
source.Parent.SelectedTab=source;
drawnow;
exportapp(app.View.Figure,fullfile(fileparts(mfilename('fullpath')),'symbolic_workbench_source_preview.png'));
end
