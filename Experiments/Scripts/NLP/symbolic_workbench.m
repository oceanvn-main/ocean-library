function app = symbolic_workbench(options)
%SYMBOLIC_WORKBENCH Launch the native MATLAB symbolic expression explorer.
%   APP = SYMBOLIC_WORKBENCH() opens the workbench and returns its handles.
%   It imports source as text and performs explicitly requested scalar
%   symbolic operations using Symbolic Math Toolbox.

arguments
    options.Visible (1,1) logical = true
end

libraryRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(libraryRoot);
app = OceanTools.SymbolicWorkbench.launch('Visible', options.Visible);
end
