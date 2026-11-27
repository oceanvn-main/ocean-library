function app = launch(options)
%LAUNCH Open the native MATLAB scalar symbolic workbench.
%   APP = OceanTools.SymbolicWorkbench.launch() returns handles to the
%   Workspace, ViewModel and View. Closing the window releases UI listeners;
%   the returned Workspace remains available for programmatic inspection.
%
%   APP = OceanTools.SymbolicWorkbench.launch('Visible',false) constructs
%   the view without displaying it, for UI integration checks.

arguments
    options.Visible (1,1) logical = true
end

if isempty(which('sym')) || ~license('test', 'Symbolic_Toolbox')
    error('OceanTools:SymbolicWorkbench:MissingToolbox', ...
        'Symbolic Workbench requires an installed and licensed Symbolic Math Toolbox.');
end

workspace = OceanMath.Symbolic.Workspace();
registry = OceanMath.Symbolic.Registry();
viewModel = OceanTools.SymbolicWorkbench.ViewModels.Main(workspace,registry);
view = OceanTools.SymbolicWorkbench.Views.Main(viewModel);
app = struct('Workspace', workspace, 'Registry',registry,'ViewModel', viewModel, 'View', view);
if options.Visible
    view.show();
end
end
